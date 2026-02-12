import asyncio
import hashlib
import logging
from collections import defaultdict
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
        gemini_client: AIClient,
        gemma_client: AIClient,
        user_resolver: UserResolver,
        redis_cache: RedisCache,
    ):
        self._telemetry = telemetry
        self._store = store
        self._gemini_client = gemini_client  # For batch daily summaries
        self._gemma_client = gemma_client  # For historical summaries and context merging
        self._user_resolver = user_resolver
        self._redis_cache = redis_cache

    async def get_memories(self, guild_id: int, user_ids: list[int]) -> dict[int, str | None]:
        """Get memories for multiple users with concurrent processing."""
        async with self._telemetry.async_create_span("get_memories") as span:
            span.set_attribute("guild_id", guild_id)
            span.set_attribute("user_count", len(user_ids))
            span.set_attribute("user_ids", str(user_ids))

            if not user_ids:
                return {}

            today = datetime.now(timezone.utc).date()
            all_dates = [today] + [today - timedelta(days=i) for i in range(1, 7)]

            # Step 1: Get all daily summaries
            daily_summaries_by_date = await self._fetch_all_daily_summaries(guild_id, all_dates)

            # Step 2: Create combined memories with all daily summaries
            return await self._create_combined_memories(guild_id, user_ids, daily_summaries_by_date)

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

    async def _create_combined_memories(
        self, guild_id: int, user_ids: list[int], daily_summaries_by_date: dict[date, dict[int, str]]
    ) -> dict[int, str | None]:
        """Create combined memories for all users by merging facts with all daily summaries."""
        async with self._telemetry.async_create_span("create_combined_memories"):
            merge_tasks = []
            for user_id in user_ids:
                # Get facts (fast DB operation)
                facts = await self._store.get_user_facts(guild_id, user_id)

                # Extract user's daily summaries from all dates
                user_daily_summaries = {}
                for date, daily_batch in daily_summaries_by_date.items():
                    if user_id in daily_batch:
                        user_daily_summaries[date] = daily_batch[user_id]

                # Add merge task
                merge_tasks.append(self._create_user_memory(guild_id, user_id, facts, user_daily_summaries))

            # Concurrently merge contexts for all users
            memories = await asyncio.gather(*merge_tasks)

            # Build result dictionary
            result = {}
            for user_id, memory in zip(user_ids, memories):
                result[user_id] = memory

            return result

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
        """Get daily summaries for all users on a given date."""
        async with self._telemetry.async_create_span("daily_summary") as span:
            span.set_attribute("guild_id", guild_id)
            span.set_attribute("for_date", str(for_date))

            outcome = "success"

            # Determine if this is current day or historical
            current_date = datetime.now(timezone.utc).date()
            is_current_day = for_date == current_date

            if is_current_day:
                # Current day: Redis-backed staleness caching with async rebuild
                # Check Redis cache first
                cached = await self._redis_cache.get_daily_summary(guild_id, for_date)
                if cached is not None:
                    summaries, created_at = cached
                    now = datetime.now(timezone.utc)
                    age = now - created_at

                    if age < STALENESS_THRESHOLD:
                        # Fresh cache (< 1 hour)
                        span.set_attribute("cache_hit", True)
                        self._telemetry.metrics.daily_summary_jobs.add(
                            1, {"guild_id": str(guild_id), "cache_outcome": "hit", "outcome": outcome}
                        )
                        return summaries

                    # Stale (>= 1 hour) - return stale + trigger async rebuild
                    span.set_attribute("cache_hit", True)
                    self._telemetry.metrics.daily_summary_jobs.add(
                        1, {"guild_id": str(guild_id), "cache_outcome": "stale_hit", "outcome": outcome}
                    )
                    self._trigger_async_rebuild(guild_id, for_date)
                    return summaries

                span.set_attribute("cache_hit", False)

                # Cache miss - return empty + trigger async rebuild
                self._telemetry.metrics.daily_summary_jobs.add(
                    1,
                    {
                        "guild_id": str(guild_id),
                        "cache_outcome": "miss",
                        "outcome": outcome,
                    },
                )
                self._trigger_async_rebuild(guild_id, for_date)
                return {}
            else:
                # Historical day: Database-first approach (caching handled by Store)

                # Check if any messages exist for this date
                has_messages = await self._store.has_chat_messages_for_date(guild_id, for_date)
                if not has_messages:
                    span.set_attribute("has_messages", False)
                    empty_dict: dict[int, str] = {}
                    self._telemetry.metrics.daily_summary_jobs.add(1, {"guild_id": str(guild_id), "outcome": outcome})
                    return empty_dict

                span.set_attribute("has_messages", True)

                # Check database for existing summaries (Store handles caching)
                db_summaries = await self._store.get_daily_summaries(guild_id, for_date)
                if db_summaries:
                    span.set_attribute("cache_hit", True)
                    self._telemetry.metrics.daily_summary_jobs.add(
                        1, {"guild_id": str(guild_id), "cache_outcome": "hit", "outcome": outcome}
                    )
                    return db_summaries

                span.set_attribute("cache_hit", False)
                # Messages exist but no summaries = needs processing
                try:
                    batch_summaries = await self._create_daily_summaries(guild_id, for_date)
                except BlockedException as blocked:
                    outcome = "blocked"
                    span.record_exception(blocked)
                    span.set_attribute("blocked_reason", blocked.reason)
                    logger.warning(
                        "Daily summary blocked for guild %s on %s: %s",
                        guild_id,
                        for_date,
                        blocked.reason,
                    )
                    batch_summaries = {}

                self._telemetry.metrics.daily_summary_jobs.add(
                    1,
                    {
                        "guild_id": str(guild_id),
                        "cache_outcome": "miss",
                        "outcome": outcome,
                    },
                )

                await self._store.save_daily_summaries(guild_id, for_date, batch_summaries)
                return batch_summaries

    def _trigger_async_rebuild(self, guild_id: int, for_date: date) -> None:
        """Trigger fire-and-forget async rebuild of daily summary using Redis lock for deduplication."""
        asyncio.create_task(self._async_rebuild_daily_summary(guild_id, for_date))

    async def _async_rebuild_daily_summary(self, guild_id: int, for_date: date) -> None:
        """Background task to rebuild daily summary asynchronously with Redis-based locking."""
        acquired = await self._redis_cache.try_acquire_build_lock(guild_id, for_date)
        if not acquired:
            return

        async with self._telemetry.async_create_span("async_rebuild_daily_summary") as span:
            try:
                batch_summaries = await self._create_daily_summaries(guild_id, for_date)

                await self._redis_cache.set_daily_summary(
                    guild_id, for_date, batch_summaries, datetime.now(timezone.utc)
                )

                # Rebuild memories for affected users to warm context cache
                if batch_summaries:
                    await self.get_memories(guild_id, list(batch_summaries.keys()))

            except BlockedException as blocked:
                span.record_exception(blocked)
                span.set_attribute("blocked_reason", blocked.reason)
                logger.warning("Async rebuild blocked for guild %s on %s: %s", guild_id, for_date, blocked.reason)
            except Exception as e:
                span.record_exception(e)
                logger.error(f"Async rebuild failed for guild {guild_id} on {for_date}: {e}", exc_info=True)
            finally:
                await self._redis_cache.release_build_lock(guild_id, for_date)

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

            # Fetch facts and extract aliases for identity resolution
            user_facts = {}
            for user_id in active_user_ids:
                facts = await self._store.get_user_facts(guild_id, user_id)
                if facts:
                    user_facts[user_id] = facts

            aliases_map = await self._extract_aliases(user_facts) if user_facts else {}

            # Create user list with names and aliases for context
            user_list = []
            for user_id in active_user_ids:
                user_name = await self._user_resolver.get_display_name(guild_id, user_id)
                aliases = aliases_map.get(user_id, [])
                also_known_as = f"<also_known_as>{', '.join(aliases)}</also_known_as>" if aliases else ""
                user_list.append(
                    f"<user><user_id>{user_id}</user_id><nickname>{user_name}</nickname>{also_known_as}</user>"
                )

            structured_prompt = f"""{BATCH_SUMMARIZE_DAILY_PROMPT}

<target_users>
{chr(10).join(user_list)}
</target_users>
<messages>
{formatted_messages}
</messages>"""

            response = await self._gemini_client.generate_content(
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

            cached = await self._redis_cache.get_context(guild_id, user_id, facts_hash, summaries_hash)
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

            response = await self._gemma_client.generate_content(
                message=structured_prompt, response_schema=MemoryContext, temperature=0
            )
            merged_context = response.context
            self._telemetry.metrics.memory_merges.add(
                1, {"guild_id": str(guild_id), "cache_outcome": "miss", "outcome": "success"}
            )

            await self._redis_cache.set_context(guild_id, user_id, facts_hash, summaries_hash, merged_context)
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

                response = await self._gemma_client.generate_content(
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

    async def build_memory_prompt(self, guild_id: int, user_ids: set[int] | list[int]) -> str:
        """Build formatted memory blocks for LLM prompts with outer tags.

        Args:
            guild_id: Discord guild ID for context resolution
            user_ids: Set or list of user IDs to fetch memories for

        Returns:
            Complete XML memories block (<memories>...</memories>), or empty string if no memories
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

            memory_blocks = []
            for user_id in user_ids_list:
                memories = memories_dict.get(user_id)
                if memories:
                    display_name = await self._user_resolver.get_display_name(guild_id, user_id)
                    memory_block = f"""<memory>
<nickname>{display_name}</nickname>
<facts>{memories}</facts>
</memory>"""
                    memory_blocks.append(memory_block)

            if not memory_blocks:
                return ""

            memories_xml = "\n".join(memory_blocks)
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
