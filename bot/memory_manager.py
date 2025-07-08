import asyncio
import hashlib
import logging
from collections import defaultdict
from datetime import datetime, timedelta, date, timezone

from cachetools import LRUCache, TTLCache

from ai_client import AIClient
from message_node import MessageNode
from open_telemetry import Telemetry
from schemas import MemoryContext, DailySummaries
from store import Store
from user_resolver import UserResolver

logger = logging.getLogger(__name__)

SUMMARIZE_HISTORICAL_PROMPT = """
Analyze the provided daily summaries for the user and create a coherent historical behavioral summary.

Focus on:
- Recurring patterns and themes from recent history
- Overall mood trends over the period
- Significant events or behavioral changes
- Personality insights from consistent behaviors

Keep the summary in the third person and under 500 characters.
"""

MERGE_CONTEXT_PROMPT = """
Merge the factual memory with current day and historical behavioral summaries for the user.

Guidelines:
- Prioritize factual information for accuracy
- Balance current observations with historical patterns
- Resolve conflicts intelligently, favoring factual data
- Provide a unified context for personalized conversation
"""

BATCH_SUMMARIZE_DAILY_PROMPT = """
Analyze the provided chat messages and create concise daily summaries for each active user.

For each user, focus on:
- Notable events or experiences they mentioned
- Their mood and emotional state
- Important interactions and topics they discussed
- Behavioral patterns they exhibited
- Information revealed about them through their messages or messages from others

Embeddings in Messages:
- Messages may contain <embedding type="image"> tags with descriptions of images that users posted
- These descriptions should be treated as if you saw the images yourself
- Messages may contain <embedding type="article"> tags with article content that users shared
- Include relevant details from shared images and articles when summarizing user behavior or interests

Keep each summary in the third person and under 300 characters.
Return a list of summaries, one for each active user.
"""

class MemoryManager:
    def __init__(
        self,
        telemetry: Telemetry,
        store: Store,
        gemini_client: AIClient,
        gemma_client: AIClient,
        user_resolver: UserResolver,
    ):
        self._telemetry = telemetry
        self._store = store
        self._gemini_client = gemini_client  # For batch daily summaries
        self._gemma_client = gemma_client   # For historical summaries and context merging
        self._user_resolver = user_resolver

        # Caches
        self._current_day_batch_cache = TTLCache(maxsize=100, ttl=3600)  # 1-hour TTL for current day batch summaries
        self._week_summary_cache = LRUCache(maxsize=500)  # Permanent cache for week summaries
        self._context_cache = LRUCache(maxsize=500)  # Final context cache

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
            
            # Step 2: Create historical summaries for all users  
            historical_summaries = await self._create_historical_summaries_for_users(
                user_ids, daily_summaries_by_date
            )
            
            # Step 3: Create combined memories
            return await self._create_combined_memories(
                guild_id, user_ids, daily_summaries_by_date, historical_summaries
            )

    async def get_memory(self, guild_id: int, user_id: int) -> str | None:
        """Get memory for a single user (backward compatibility wrapper)."""
        memories_dict = await self.get_memories(guild_id, [user_id])
        return memories_dict.get(user_id)

    async def _fetch_all_daily_summaries(self, guild_id: int, all_dates: list[date]) -> dict[date, dict[int, str]]:
        """Fetch daily summaries for all dates concurrently."""
        async with self._telemetry.async_create_span("fetch_all_daily_summaries") as span:
            # Fire concurrent calls to get all daily summaries
            daily_summary_results = await asyncio.gather(*[
                self._daily_summary(guild_id, date) for date in all_dates
            ], return_exceptions=True)
            
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

    async def _create_historical_summaries_for_users(self, user_ids: list[int], daily_summaries_by_date: dict[date, dict[int, str]]) -> dict[int, str | None]:
        """Create historical summaries for all users concurrently."""
        async with self._telemetry.async_create_span("create_historical_summaries_for_users") as span:
            historical_tasks = []
            # Skip first item (today) and use rest for historical summaries
            historical_dates_items = list(daily_summaries_by_date.items())[1:]
            
            for user_id in user_ids:
                user_daily_summaries = {}
                for date, daily_batch in historical_dates_items:
                    if user_id in daily_batch:
                        user_daily_summaries[date] = daily_batch[user_id]
                
                historical_tasks.append(self._create_week_summary(user_daily_summaries))
            
            historical_results_raw = await asyncio.gather(*historical_tasks, return_exceptions=True)
            
            # Convert exceptions to None and log them
            historical_results = {}
            for i, result in enumerate(historical_results_raw):
                if isinstance(result, Exception):
                    logger.warning(f"Historical summary failed for user {user_ids[i]}: {result}")
                    span.record_exception(result)
                    historical_results[user_ids[i]] = None
                else:
                    historical_results[user_ids[i]] = result
            
            return historical_results

    async def _create_combined_memories(self, guild_id: int, user_ids: list[int], daily_summaries_by_date: dict[date, dict[int, str]], historical_summaries: dict[int, str | None]) -> dict[int, str | None]:
        """Create combined memories for all users by merging facts, current day, and historical summaries."""
        async with self._telemetry.async_create_span("create_combined_memories"):
            # Find today (most recent date in daily_summaries_by_date)
            if not daily_summaries_by_date:
                # If no daily summaries available, use current date
                today = datetime.now(timezone.utc).date()
            else:
                today = max(daily_summaries_by_date.keys())
            
            merge_tasks = []
            for user_id in user_ids:
                # Get facts (fast DB operation)
                facts = await self._store.get_user_facts(guild_id, user_id)
                historical_summary = historical_summaries.get(user_id)
                current_day_summary = daily_summaries_by_date[today].get(user_id)
                
                # Add merge task
                merge_tasks.append(
                    self._create_user_memory(guild_id, user_id, facts, current_day_summary, historical_summary)
                )
            
            # Concurrently merge contexts for all users
            memories = await asyncio.gather(*merge_tasks)
            
            # Build result dictionary
            result = {}
            for user_id, memory in zip(user_ids, memories):
                result[user_id] = memory
            
            return result

    async def _create_user_memory(self, guild_id: int, user_id: int, facts: str | None, 
                                 current_day: str | None, historical: str | None) -> str | None:
        """Process memory for a single user."""
        # If no memories exist at all, return None
        if not facts and not current_day and not historical:
            return None

        # If only one type exists, return it directly
        if facts and not current_day and not historical:
            return facts
        if current_day and not facts and not historical:
            return current_day
        if historical and not facts and not current_day:
            return historical
            
        # Multiple sources exist - try to merge them with AI, fall back to facts if it fails
        try:
            merged = await self._merge_context(guild_id, user_id, facts, current_day, historical)
            return merged
        except Exception as e:
            logger.warning(f"Failed to merge context for user {user_id} in guild {guild_id}: {e}", exc_info=True)
            # Fall back to facts if available, otherwise return what we have
            if facts:
                return facts
            elif current_day:
                return current_day
            elif historical:
                return historical
            return None

    async def _daily_summary(self, guild_id: int, for_date: date) -> dict[int, str]:
        """Get daily summaries for all users on a given date."""
        async with self._telemetry.async_create_span("daily_summary") as span:
            span.set_attribute("guild_id", guild_id)
            span.set_attribute("for_date", str(for_date))
            
            # Determine if this is current day or historical
            current_date = datetime.now(timezone.utc).date()
            is_current_day = for_date == current_date
            
            if is_current_day:
                # Current day: Use existing TTL cache behavior
                cache_key = (guild_id, for_date)
                
                # Check cache first
                if cache_key in self._current_day_batch_cache:
                    span.set_attribute("cache_hit", True)
                    return self._current_day_batch_cache[cache_key]
                
                span.set_attribute("cache_hit", False)
                
                # Generate batch summaries and cache the result
                batch_summaries = await self._create_daily_summaries(guild_id, for_date)
                self._current_day_batch_cache[cache_key] = batch_summaries
                
                return batch_summaries
            else:
                # Historical day: Database-first approach (caching handled by Store)
                
                # Check if any messages exist for this date
                has_messages = await self._store.has_chat_messages_for_date(guild_id, for_date)
                if not has_messages:
                    span.set_attribute("has_messages", False)
                    empty_dict: dict[int, str] = {}
                    return empty_dict
                
                span.set_attribute("has_messages", True)
                
                # Check database for existing summaries (Store handles caching)
                db_summaries = await self._store.get_daily_summaries(guild_id, for_date)
                if db_summaries:
                    return db_summaries
                
                # Messages exist but no summaries = needs processing
                batch_summaries = await self._create_daily_summaries(guild_id, for_date)
                
                # Save to database (Store handles caching)
                await self._store.save_daily_summaries(guild_id, for_date, batch_summaries)
                
                return batch_summaries

    async def _create_week_summary(self, daily_summaries: dict[date, str]) -> str | None:
        """Build historical summary from daily summaries dictionary."""
        async with self._telemetry.async_create_span("create_week_summary") as span:
            if not daily_summaries:
                return None
                
            # Create cache key from hash of all summary strings
            summaries_concat = "".join(daily_summaries.values())
            cache_key = hashlib.md5(summaries_concat.encode()).hexdigest()
            
            if cache_key in self._week_summary_cache:
                span.set_attribute("cache_hit", True)
                return self._week_summary_cache[cache_key]
            
            span.set_attribute("cache_hit", False)
            span.set_attribute("daily_summaries_count", len(daily_summaries))
            
            # Format summaries for prompt
            daily_summary_blocks = []
            dates = sorted(daily_summaries.keys(), reverse=True)  # Most recent first
            for date in dates:
                summary = daily_summaries[date]
                daily_summary_blocks.append(f"<daily_summary>\n<date>{date}</date>\n<summary>{summary}</summary>\n</daily_summary>")
            
            # Create date range string
            date_range = f"{min(dates)} to {max(dates)}" if len(dates) > 1 else str(dates[0])
            
            # Create structured prompt with XML data
            structured_prompt = f"""{SUMMARIZE_HISTORICAL_PROMPT}

<date_range>{date_range}</date_range>
<daily_summaries>
{"\\n".join(daily_summary_blocks)}
</daily_summaries>"""
            
            response = await self._gemma_client.generate_content(
                message=structured_prompt,
                response_schema=MemoryContext,
                temperature=0
            )
            historical_summary = response.context

            self._week_summary_cache[cache_key] = historical_summary
            return historical_summary

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
            
            # Get all unique user IDs from messages
            active_user_ids = list(set(msg.user_id for msg in messages))
            span.set_attribute("active_user_count", len(active_user_ids))
            
            # Format messages like general_query_generator does
            message_blocks = []
            for msg in messages:
                author_name = await self._user_resolver.get_display_name(guild_id, msg.user_id)
                content_with_names = await self._user_resolver.replace_user_mentions_with_names(msg.message_text, guild_id)
                message_block = f"""<message>
<timestamp>{msg.timestamp}</timestamp>
<author_id>{msg.user_id}</author_id>
<author>{author_name}</author>
<content>{content_with_names}</content>
</message>"""
                message_blocks.append(message_block)
            
            # Create user list with names for context
            user_list = []
            for user_id in active_user_ids:
                user_name = await self._user_resolver.get_display_name(guild_id, user_id)
                user_list.append(f"<user><user_id>{user_id}</user_id><name>{user_name}</name></user>")
            
            structured_prompt = f"""{BATCH_SUMMARIZE_DAILY_PROMPT}

<target_users>
{chr(10).join(user_list)}
</target_users>
<messages>
{chr(10).join(message_blocks)}
</messages>"""
            
            response = await self._gemini_client.generate_content(
                message=structured_prompt,
                response_schema=DailySummaries,
                temperature=0
            )
            
            # Convert list of UserSummary objects to dict[int, str]
            summaries_dict = {user_summary.user_id: user_summary.summary for user_summary in response.summaries}
            return summaries_dict

    async def _merge_context(self, guild_id: int, user_id: int, facts: str | None, current_day: str | None, historical: str | None) -> str:
        """Merge factual memory with current day and historical summaries using AI."""
        async with self._telemetry.async_create_span("merge_context") as span:
            span.set_attribute("guild_id", guild_id)
            span.set_attribute("user_id", user_id)
            
            # Create cache key based on content hash of all inputs
            facts_hash = hashlib.md5((facts or "").encode()).hexdigest()
            current_hash = hashlib.md5((current_day or "").encode()).hexdigest()
            historical_hash = hashlib.md5((historical or "").encode()).hexdigest()
            cache_key = (guild_id, user_id, facts_hash, current_hash, historical_hash)
            
            if cache_key in self._context_cache:
                span.set_attribute("cache_hit", True)
                return self._context_cache[cache_key]
            
            span.set_attribute("cache_hit", False)
            user_nick = await self._user_resolver.get_display_name(guild_id, user_id)
            
            # Create structured prompt with XML data
            structured_prompt = f"""{MERGE_CONTEXT_PROMPT}

<user_name>{user_nick}</user_name>
<factual_memory>{facts or "No factual information available."}</factual_memory>
<current_day_summary>{current_day or "No current day observations."}</current_day_summary>
<historical_summary>{historical or "No historical observations."}</historical_summary>"""
            
            response = await self._gemma_client.generate_content(
                message=structured_prompt,
                response_schema=MemoryContext,
                temperature=0
            )
            merged_context = response.context
                
            self._context_cache[cache_key] = merged_context
            return merged_context

    async def ingest_message(self, guild_id: int, message: MessageNode) -> None:
        await self._store.add_chat_message(guild_id, message.channel_id, message.id, message.author_id, message.content, message.created_at)