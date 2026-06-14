import asyncio
import hashlib
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, date, timezone

from ai_client import AIClient, BlockedException
from conversation_formatter import ConversationFormatter
from conversation_graph import ConversationMessage
from message_node import MessageNode
from open_telemetry import Telemetry
from redis_cache import RedisCache
from schemas import MemoryContext, DailySummaries, UserAliases
from store import Store
from user_resolver import UserResolver

logger = logging.getLogger(__name__)


MERGE_CONTEXT_PROMPT = """
Merge the factual memory with daily summaries from the past week for the user.

Guidelines:
- Prioritize factual information for accuracy
- Preserve specific events and conversations from recent days
- Identify patterns across the full week while maintaining detail
- Resolve conflicts intelligently, favoring factual data then more recent summaries
- Provide unified context with rich recent memory for personalized conversation
"""

BATCH_SUMMARIZE_DAILY_PROMPT = """
Analyze the provided chat messages and create concise daily summaries for each active user.

For each user, focus on:
- Notable events or experiences they mentioned
- Their mood and emotional state
- Important interactions and topics they discussed
- Behavioral patterns they exhibited
- Information revealed about them through their messages or messages from others

Identity Resolution:
- People in chat often address each other by real names, not Discord nicknames
- Use <also_known_as> mappings to connect real names in messages to the correct Discord user

Embeddings in Messages:
- Messages may contain <embedding type="image"> tags with descriptions of images that users posted
- These descriptions should be treated as if you saw the images yourself
- Messages may contain <embedding type="article"> tags with article content that users shared
- Include relevant details from shared images and articles when summarizing user behavior or interests

Keep each summary in the third person.
Return a list of summaries, one for each active user.
"""

EXTRACT_ALIASES_PROMPT = """
Extract all known real names, nicknames, and alternative names for this user from their factual memory.

Only extract names that clearly identify the same person. Do not extract generic descriptions, roles, or locations.

Examples:
- "He is Sergey" → ["Sergey"]
- "Also known as Медвед, real name is Pierre" → ["Медвед", "Pierre"]
- "Works at Google, lives in Berlin" → []
- "Her name is Sarah, friends call her Saz" → ["Sarah", "Saz"]
- "His username on Steam is darkslayer99" → ["darkslayer99"]
- "Его ник в плексе naruto" → ["naruto"]
"""

STALENESS_THRESHOLD = timedelta(hours=1)


@dataclass
class CachedDailySummary:
    """Wrapper for cached daily summaries with timestamp for staleness tracking."""

    summaries: dict[int, str]
    created_at: datetime


class MemoryManager:
    def __init__(
        self,
        telemetry: Telemetry,
        store: Store,
        summary_client: AIClient,
        alias_client: AIClient,
        merge_client: AIClient,
        user_resolver: UserResolver,
        redis_cache: RedisCache,
        timeout: float | None = 1.0,
    ):
        self._telemetry = telemetry
        self._store = store
        self._summary_client = summary_client
        self._alias_client = alias_client
        self._merge_client = merge_client
        self._user_resolver = user_resolver
        self._redis_cache = redis_cache
        self._timeout = timeout
        # In-process single-flight registries: concurrent callers for the same key await one
        # shared build instead of recomputing it. Valid because the bot is a single event loop.
        self._date_inflight: dict[tuple[int, date], asyncio.Task[dict[int, str]]] = {}
        self._member_inflight: dict[tuple[int, int], asyncio.Task[str | None]] = {}

    def _coalesce(self, registry: dict, key: tuple, factory: Callable[[], Awaitable]) -> asyncio.Task:
        """Return the in-flight task for ``key`` or start one, so concurrent callers share a
        single build (await-not-skip) rather than each recomputing it."""
        task = registry.get(key)
        if task is None or task.done():
            task = asyncio.create_task(factory())
            registry[key] = task
            task.add_done_callback(lambda t: self._discard(registry, key, t))
        return task

    def _discard(self, registry: dict, key: tuple, task: asyncio.Task) -> None:
        if registry.get(key) is task:
            registry.pop(key, None)
        if not task.cancelled():
            task.exception()  # retrieve to suppress "exception was never retrieved" warnings

    async def get_memories(self, guild_id: int, user_ids: list[int]) -> dict[int, str | None]:
        """Get memories for multiple users, bounding each member build by the timeout.

        Each member's memory is produced by a coalesced single-flight build raced against
        ``timeout``; if it doesn't finish in time, the last-known-good value
        (``memory_latest``, else raw facts) is served while the shielded build keeps running and
        warms the cache for the next read. Routing therefore never blocks on an LLM merge.
        """
        async with self._telemetry.async_create_span("get_memories") as span:
            span.set_attribute("guild_id", guild_id)
            span.set_attribute("user_count", len(user_ids))
            span.set_attribute("user_ids", str(user_ids))

            if not user_ids:
                return {}

            memories = await asyncio.gather(*[self._get_member_memory(guild_id, user_id) for user_id in user_ids])
            return dict(zip(user_ids, memories))

    async def _get_member_memory(self, guild_id: int, user_id: int) -> str | None:
        """Build a member's memory within the timeout, falling back to the last-known-good value."""
        task = self._coalesce(
            self._member_inflight, (guild_id, user_id), lambda: self._build_member_memory(guild_id, user_id)
        )

        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=self._timeout)
        except asyncio.TimeoutError:
            self._telemetry.metrics.memory_merges.add(
                1, {"guild_id": str(guild_id), "cache_outcome": "timeout_fallback", "outcome": "success"}
            )
            latest = await self._redis_cache.get_memory_latest(guild_id, user_id)
            if latest is not None:
                return latest
            return await self._store.get_user_facts(guild_id, user_id)

    async def get_memory(self, guild_id: int, user_id: int) -> str | None:
        """Get memory for a single user (backward compatibility wrapper)."""
        memories_dict = await self.get_memories(guild_id, [user_id])
        return memories_dict.get(user_id)

    async def _fetch_all_daily_summaries(self, guild_id: int, all_dates: list[date]) -> dict[date, dict[int, str]]:
        """Fetch daily summaries for all dates concurrently."""
        async with self._telemetry.async_create_span("fetch_all_daily_summaries") as span:
            # Fire concurrent calls to get all daily summaries
            daily_summary_results = await asyncio.gather(
                *[self._daily_summary(guild_id, date) for date in all_dates], return_exceptions=True
            )

            # Create a map of date -> result for easy lookup (defaults to empty dict for failed dates)
            daily_summaries_by_date = defaultdict(dict)
            for date, result in zip(all_dates, daily_summary_results):
                if isinstance(result, Exception):
                    logger.warning(f"Daily summary failed for {date}: {result}")
                    span.record_exception(result)
                    # defaultdict will automatically provide {} for this date
                else:
                    daily_summaries_by_date[date] = result

            return daily_summaries_by_date

    async def _build_member_memory(self, guild_id: int, user_id: int) -> str | None:
        """Synchronously build one member's memory: facts merged with the 7-day summary window.

        Correctness over latency — this may block on a synchronous daily-summary rebuild and an
        LLM merge. The daily fetch lives here (inside the timeout-bounded region) so its date builds,
        shared across members via the coalescing registry, are covered by the caller's timeout.
        """
        async with self._telemetry.async_create_span("build_member_memory") as span:
            span.set_attribute("guild_id", guild_id)
            span.set_attribute("user_id", user_id)

            today = datetime.now(timezone.utc).date()
            all_dates = [today] + [today - timedelta(days=i) for i in range(1, 7)]
            daily_summaries_by_date = await self._fetch_all_daily_summaries(guild_id, all_dates)

            facts = await self._store.get_user_facts(guild_id, user_id)
            user_daily_summaries = {
                day: batch[user_id] for day, batch in daily_summaries_by_date.items() if user_id in batch
            }
            return await self._create_user_memory(guild_id, user_id, facts, user_daily_summaries)

    async def _create_user_memory(
        self, guild_id: int, user_id: int, facts: str | None, daily_summaries: dict[date, str]
    ) -> str | None:
        """Process memory for a single user."""
        # If no memories exist at all, return None
        if not facts and not daily_summaries:
            return None

        # If only facts exist, return them directly
        if facts and not daily_summaries:
            return facts

        # If only one daily summary exists and no facts, return it directly
        if not facts and len(daily_summaries) == 1:
            return list(daily_summaries.values())[0]

        # Multiple sources exist (facts + daily summaries, or multiple daily summaries) - merge with AI
        try:
            merged = await self._merge_context(guild_id, user_id, facts, daily_summaries)
            return merged
        except Exception as e:
            logger.error(f"Failed to merge context for user {user_id} in guild {guild_id}: {e}", exc_info=True)
            # Record merge failure once
            self._telemetry.metrics.memory_merges.add(
                1, {"guild_id": str(guild_id), "cache_outcome": "miss", "outcome": "error"}
            )
            # Fallback hierarchy: facts first, then most recent daily summary
            if facts:
                return facts
            elif daily_summaries:
                most_recent_date = max(daily_summaries.keys())
                return daily_summaries[most_recent_date]
            return None

    async def _daily_summary(self, guild_id: int, for_date: date) -> dict[int, str]:
        """Return all users' summaries for a date, building synchronously (coalesced) on a miss.

        The current day is cached in Redis with a staleness window (a value fresher than
        STALENESS_THRESHOLD is reused); historical days are immutable in the DB. A miss — or a
        stale current-day entry — triggers a single-flight synchronous rebuild rather than
        serving an empty result, so a merge is never built on a known-incomplete summary set.
        """
        async with self._telemetry.async_create_span("daily_summary") as span:
            span.set_attribute("guild_id", guild_id)
            span.set_attribute("for_date", str(for_date))

            is_current_day = for_date == datetime.now(timezone.utc).date()

            if is_current_day:
                cached = await self._redis_cache.get_daily_summary(guild_id, for_date)
                if cached is not None:
                    summaries, created_at = cached
                    if datetime.now(timezone.utc) - created_at < STALENESS_THRESHOLD:
                        span.set_attribute("cache_hit", True)
                        self._telemetry.metrics.daily_summary_jobs.add(
                            1, {"guild_id": str(guild_id), "cache_outcome": "hit", "outcome": "success"}
                        )
                        return summaries
            else:
                if not await self._store.has_chat_messages_for_date(guild_id, for_date):
                    span.set_attribute("has_messages", False)
                    self._telemetry.metrics.daily_summary_jobs.add(1, {"guild_id": str(guild_id), "outcome": "success"})
                    return {}

                span.set_attribute("has_messages", True)
                db_summaries = await self._store.get_daily_summaries(guild_id, for_date)
                if db_summaries:
                    span.set_attribute("cache_hit", True)
                    self._telemetry.metrics.daily_summary_jobs.add(
                        1, {"guild_id": str(guild_id), "cache_outcome": "hit", "outcome": "success"}
                    )
                    return db_summaries

            span.set_attribute("cache_hit", False)
            self._telemetry.metrics.daily_summary_jobs.add(
                1, {"guild_id": str(guild_id), "cache_outcome": "miss", "outcome": "success"}
            )
            return await self._coalesce(
                self._date_inflight, (guild_id, for_date), lambda: self._build_date_summary(guild_id, for_date)
            )

    async def _build_date_summary(self, guild_id: int, for_date: date) -> dict[int, str]:
        """Generate a date's summaries and persist them. Current day → Redis (timestamped for
        staleness); historical → DB. The target is decided at execution time so a build that
        straddles midnight persists according to what the date has become."""
        async with self._telemetry.async_create_span("build_date_summary") as span:
            span.set_attribute("guild_id", guild_id)
            span.set_attribute("for_date", str(for_date))

            is_current_day = for_date == datetime.now(timezone.utc).date()
            try:
                summaries = await self._create_daily_summaries(guild_id, for_date)
            except BlockedException as blocked:
                span.record_exception(blocked)
                span.set_attribute("blocked_reason", blocked.reason)
                logger.warning("Daily summary blocked for guild %s on %s: %s", guild_id, for_date, blocked.reason)
                # Persist empty only for historical dates so a refused generation isn't retried
                # forever; the current day is left uncached so the next read attempts a rebuild.
                if not is_current_day:
                    await self._store.save_daily_summaries(guild_id, for_date, {})
                return {}

            if is_current_day:
                await self._redis_cache.set_daily_summary(guild_id, for_date, summaries, datetime.now(timezone.utc))
            else:
                await self._store.save_daily_summaries(guild_id, for_date, summaries)
            return summaries

    async def _create_daily_summaries(self, guild_id: int, for_date: date) -> dict[int, str]:
        """Generate daily summaries for all active users in a single API call. Pure generation method - no caching."""
        async with self._telemetry.async_create_span("create_daily_summaries") as span:
            span.set_attribute("guild_id", guild_id)
            span.set_attribute("for_date", str(for_date))

            messages = await self._store.get_chat_messages_for_date(guild_id, for_date)
            if not messages:
                empty_dict: dict[int, str] = {}
                return empty_dict

            span.set_attribute("message_count", len(messages))
            self._telemetry.metrics.daily_summary_messages.record(len(messages), {"guild_id": str(guild_id)})

            active_user_ids = list(set(msg.user_id for msg in messages))
            span.set_attribute("active_user_count", len(active_user_ids))

            conversation_messages = [
                ConversationMessage(
                    message_id=msg.message_id,
                    author_id=msg.user_id,
                    content=msg.message_text,
                    timestamp=str(msg.timestamp),
                    mentioned_user_ids=[],
                    reply_to_id=msg.reply_to_id,
                )
                for msg in messages
            ]

            formatter = ConversationFormatter(self._user_resolver)
            formatted_messages = await formatter.format_to_xml(guild_id, conversation_messages)

            aliases_map = await self._extract_aliases_for(guild_id, active_user_ids)

            # Create user list with names and aliases for context
            user_list = []
            for user_id in active_user_ids:
                user_name = await self._user_resolver.get_display_name(guild_id, user_id)
                aliases = aliases_map.get(user_id, [])
                also_known_as = f"<also_known_as>{', '.join(aliases)}</also_known_as>" if aliases else ""
                user_list.append(
                    f"<member><member_id>{user_id}</member_id><member_name>{user_name}</member_name>{also_known_as}</member>"
                )

            structured_prompt = f"""{BATCH_SUMMARIZE_DAILY_PROMPT}

<target_members>
{chr(10).join(user_list)}
</target_members>
<messages>
{formatted_messages}
</messages>"""

            response = await self._summary_client.generate_content(
                message=structured_prompt, response_schema=DailySummaries, temperature=0
            )

            # Convert list of UserSummary objects to dict[int, str]
            summaries_dict = {user_summary.user_id: user_summary.summary for user_summary in response.summaries}
            return summaries_dict

    async def _merge_context(
        self, guild_id: int, user_id: int, facts: str | None, daily_summaries: dict[date, str]
    ) -> str:
        """Merge factual memory with daily summaries from the past week using AI."""
        async with self._telemetry.async_create_span("merge_context") as span:
            span.set_attribute("guild_id", guild_id)
            span.set_attribute("user_id", user_id)
            span.set_attribute("daily_summaries_count", len(daily_summaries))

            # Create cache key based on content hash of all inputs
            facts_hash = hashlib.md5((facts or "").encode()).hexdigest()
            summaries_concat = "".join(f"{date}:{summary}" for date, summary in sorted(daily_summaries.items()))
            summaries_hash = hashlib.md5(summaries_concat.encode()).hexdigest()

            cached = await self._redis_cache.get_memory(guild_id, user_id, facts_hash, summaries_hash)
            if cached is not None:
                span.set_attribute("cache_hit", True)
                self._telemetry.metrics.memory_merges.add(
                    1, {"guild_id": str(guild_id), "cache_outcome": "hit", "outcome": "success"}
                )
                return cached

            span.set_attribute("cache_hit", False)
            user_nick = await self._user_resolver.get_display_name(guild_id, user_id)

            # Format daily summaries for prompt
            daily_summary_blocks = []
            if daily_summaries:
                dates = sorted(daily_summaries.keys(), reverse=True)  # Most recent first
                for date in dates:
                    summary = daily_summaries[date]
                    daily_summary_blocks.append(
                        f"<daily_summary>\n<date>{date}</date>\n<summary>{summary}</summary>\n</daily_summary>"
                    )

            daily_summaries_xml = (
                "\n".join(daily_summary_blocks) if daily_summary_blocks else "No daily summaries available."
            )

            # Create structured prompt with XML data
            structured_prompt = f"""{MERGE_CONTEXT_PROMPT}

<user_name>{user_nick}</user_name>
<factual_memory>{facts or "No factual information available."}</factual_memory>
<daily_summaries>
{daily_summaries_xml}
</daily_summaries>"""

            response = await self._merge_client.generate_content(
                message=structured_prompt, response_schema=MemoryContext, temperature=0
            )
            merged_context = response.context
            self._telemetry.metrics.memory_merges.add(
                1, {"guild_id": str(guild_id), "cache_outcome": "miss", "outcome": "success"}
            )

            await self._redis_cache.set_memory(guild_id, user_id, facts_hash, summaries_hash, merged_context)
            return merged_context

    async def _extract_aliases(self, user_facts: dict[int, str]) -> dict[int, list[str]]:
        """Extract known names/aliases from factual memory for identity resolution.

        Processes all users concurrently, each cached by hash(facts).
        """
        async with self._telemetry.async_create_span("extract_aliases") as span:
            span.set_attribute("user_count", len(user_facts))

            async def _extract_for_user(user_id: int, facts: str) -> tuple[int, list[str]]:
                facts_hash = hashlib.md5(facts.encode()).hexdigest()

                cached = await self._redis_cache.get_aliases(facts_hash)
                if cached is not None:
                    return user_id, cached

                response = await self._alias_client.generate_content(
                    message=facts,
                    prompt=EXTRACT_ALIASES_PROMPT,
                    response_schema=UserAliases,
                    temperature=0,
                )
                aliases = response.aliases

                await self._redis_cache.set_aliases(facts_hash, aliases)
                return user_id, aliases

            results = await asyncio.gather(
                *[_extract_for_user(uid, facts) for uid, facts in user_facts.items()],
                return_exceptions=True,
            )

            aliases_map: dict[int, list[str]] = {}
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Alias extraction failed: {result}")
                    span.record_exception(result)
                else:
                    uid, aliases = result
                    aliases_map[uid] = aliases

            return aliases_map

    async def _extract_aliases_for(self, guild_id: int, user_ids: list[int] | set[int]) -> dict[int, list[str]]:
        """Fetch each member's stored facts and extract their aliases for identity resolution."""
        user_facts = {}
        for user_id in user_ids:
            facts = await self._store.get_user_facts(guild_id, user_id)
            if facts:
                user_facts[user_id] = facts
        return await self._extract_aliases(user_facts) if user_facts else {}

    async def build_memory_prompt(self, guild_id: int, user_ids: set[int] | list[int]) -> str:
        """Build formatted memory blocks for LLM prompts with outer tags.

        Args:
            guild_id: Discord guild ID for context resolution
            user_ids: Set or list of user IDs to fetch memories for

        Returns:
            Complete XML members block (<members>...</members>), or empty string if no memories
        """
        if not user_ids:
            return ""

        async with self._telemetry.async_create_span("build_memory_prompt") as span:
            span.set_attribute("guild_id", guild_id)
            span.set_attribute("user_count", len(user_ids))
            span.set_attribute("user_ids", str(sorted(user_ids)))

            # Use batch interface for concurrent processing
            user_ids_list = list(user_ids)
            memories_dict = await self.get_memories(guild_id, user_ids_list)

            aliases_map = await self._extract_aliases_for(guild_id, user_ids_list)

            member_blocks = []
            for user_id in user_ids_list:
                memories = memories_dict.get(user_id)
                if memories:
                    display_name = await self._user_resolver.get_display_name(guild_id, user_id)
                    aliases = aliases_map.get(user_id, [])
                    also_known_as = f"\n<also_known_as>{', '.join(aliases)}</also_known_as>" if aliases else ""
                    member_block = f"""<memory>
<member_id>{user_id}</member_id>
<member_name>{display_name}</member_name>{also_known_as}
<facts>{memories}</facts>
</memory>"""
                    member_blocks.append(member_block)

            if not member_blocks:
                return ""

            memories_xml = "\n".join(member_blocks)
            return f"<memories>\n{memories_xml}\n</memories>"

    async def ingest_message(self, guild_id: int, message: MessageNode) -> None:
        await self._store.add_chat_message(
            guild_id,
            message.channel_id,
            message.id,
            message.author_id,
            message.content,
            message.created_at,
            message.reference_id,
        )
