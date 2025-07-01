from datetime import datetime, timedelta, date, timezone
from typing import Awaitable, Callable
import logging
import hashlib
from cachetools import LRUCache, TTLCache
from ai_client import AIClient
from message_node import MessageNode
from open_telemetry import Telemetry
from schemas import MemoryContext
from store import Store
from user_resolver import UserResolver

logger = logging.getLogger(__name__)

SUMMARIZE_DAILY_PROMPT = """
Analyze the provided chat messages and create a concise daily summary focusing specifically on the target user.

Focus on these areas about the target user:
- Notable events or experiences they mentioned
- Their mood and emotional state
- Important interactions and topics they discussed
- Behavioral patterns they exhibited
- Information revealed about them through their messages or messages from others

Keep the summary in the third person and under 300 characters.
"""

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

class MemoryManager:
    def __init__(
        self,
        telemetry: Telemetry,
        store: Store,
        gemma_client: AIClient,
        user_resolver: UserResolver,
    ):
        self._telemetry = telemetry
        self._store = store
        self._gemma_client = gemma_client
        self._user_resolver = user_resolver

        # Caches
        self._current_day_cache = TTLCache(maxsize=100, ttl=3600)  # 1-hour TTL for current day
        self._historical_daily_cache = LRUCache(maxsize=1000)  # Permanent cache for historical days
        self._historical_summary_cache = LRUCache(maxsize=500)  # Permanent cache for historical summaries
        self._context_cache = LRUCache(maxsize=500)  # Final context cache

    async def get_memories(self, guild_id: int, user_id: int) -> str | None:
        async with self._telemetry.async_create_span("get_memories") as span:
            span.set_attribute("guild_id", guild_id)
            span.set_attribute("user_id", user_id)
            
            # Always get facts first - these are reliable and from DB
            facts = await self._store.get_user_facts(guild_id, user_id)
            
            # Try to get transient memory with best effort
            today = datetime.now(timezone.utc).date()
            current_day_summary = None
            historical_summary = None
            
            try:
                current_day_summary = await self._get_current_day_summary(guild_id, user_id, today)
            except Exception as e:
                logger.warning(f"Failed to get current day summary for user {user_id} in guild {guild_id}: {e}", exc_info=True)
                span.record_exception(e)
                
            try:
                historical_summary = await self._get_historical_summary(guild_id, user_id, today)
            except Exception as e:
                logger.warning(f"Failed to get historical summary for user {user_id} in guild {guild_id}: {e}", exc_info=True)
                span.record_exception(e)
            
            
            # If no memories exist at all, return None
            if not facts and not current_day_summary and not historical_summary:
                return None

            # If only one type exists, return it directly
            if facts and not current_day_summary and not historical_summary:
                return facts
            if current_day_summary and not facts and not historical_summary:
                return current_day_summary
            if historical_summary and not facts and not current_day_summary:
                return historical_summary
                
            # Multiple sources exist - try to merge them with AI, fall back to facts if it fails
            try:
                merged = await self._merge_context(guild_id, user_id, facts, current_day_summary, historical_summary)
                return merged
            except Exception as e:
                logger.warning(f"Failed to merge context for user {user_id} in guild {guild_id}: {e}", exc_info=True)
                span.record_exception(e)
                # Fall back to facts if available, otherwise return what we have
                if facts:
                    return facts
                elif current_day_summary:
                    return current_day_summary
                elif historical_summary:
                    return historical_summary
                return None

    async def _generate_daily_summary(self, guild_id: int, user_id: int, for_date: date) -> str | None:
        """Generate daily summary for given date without caching."""
        async with self._telemetry.async_create_span("generate_daily_summary") as span:
            span.set_attribute("guild_id", guild_id)
            span.set_attribute("user_id", user_id)
            span.set_attribute("for_date", str(for_date))
            
            messages = await self._store.get_chat_messages_for_date(guild_id, for_date)
            if not messages:
                return None
            
            span.set_attribute("message_count", len(messages))

            # Format messages like general_query_generator does
            message_blocks = []
            for msg in messages:
                author_name = await self._user_resolver.get_display_name(guild_id, msg.user_id)
                content_with_names = await self._user_resolver.replace_user_mentions_with_names(msg.message_text, guild_id)
                message_block = f"""<message>
<timestamp>{msg.timestamp}</timestamp>
<author>{author_name}</author>
<content>{content_with_names}</content>
</message>"""
                message_blocks.append(message_block)
            
            user_nick = await self._user_resolver.get_display_name(guild_id, user_id)
            structured_prompt = f"""{SUMMARIZE_DAILY_PROMPT}

<target_user_name>{user_nick}</target_user_name>
<target_user_id>{user_id}</target_user_id>
<messages>
{"\n".join(message_blocks)}
</messages>"""
            
            response = await self._gemma_client.generate_content(
                message=structured_prompt,
                response_schema=MemoryContext,
                temperature=0
            )
            return response.context

    async def _get_current_day_summary(self, guild_id: int, user_id: int, for_date: date) -> str | None:
        """Get current day summary with 1-hour TTL caching using hour buckets."""
        async with self._telemetry.async_create_span("get_current_day_summary") as span:
            span.set_attribute("guild_id", guild_id)
            span.set_attribute("user_id", user_id)
            span.set_attribute("for_date", str(for_date))
            
            current_hour = datetime.now(timezone.utc).hour
            hour_bucket = f"{for_date}-{current_hour:02d}"
            cache_key = (guild_id, user_id, hour_bucket)
            
            if cache_key in self._current_day_cache:
                span.set_attribute("cache_hit", True)
                return self._current_day_cache[cache_key]
            
            span.set_attribute("cache_hit", False)
            summary = await self._generate_daily_summary(guild_id, user_id, for_date)
            if summary:
                self._current_day_cache[cache_key] = summary
            return summary

    async def _get_historical_daily_summary(self, guild_id: int, user_id: int, for_date: date) -> str | None:
        """Get historical daily summary with permanent LRU caching."""
        async with self._telemetry.async_create_span("get_historical_daily_summary") as span:
            span.set_attribute("guild_id", guild_id)
            span.set_attribute("user_id", user_id)
            span.set_attribute("for_date", str(for_date))
            
            cache_key = (guild_id, user_id, for_date)
            if cache_key in self._historical_daily_cache:
                span.set_attribute("cache_hit", True)
                return self._historical_daily_cache[cache_key]
            
            span.set_attribute("cache_hit", False)
            summary = await self._generate_daily_summary(guild_id, user_id, for_date)
            if summary:
                self._historical_daily_cache[cache_key] = summary
            return summary

    async def _get_historical_summary(self, guild_id: int, user_id: int, current_date: date) -> str | None:
        """Get historical summary for days 2-7 with permanent LRU caching."""
        async with self._telemetry.async_create_span("get_historical_summary") as span:
            span.set_attribute("guild_id", guild_id)
            span.set_attribute("user_id", user_id)
            span.set_attribute("current_date", str(current_date))
            
            # Historical end date is day 2 (yesterday relative to current_date)
            historical_end_date = current_date - timedelta(days=1)
            cache_key = (guild_id, user_id, historical_end_date)
            
            if cache_key in self._historical_summary_cache:
                span.set_attribute("cache_hit", True)
                return self._historical_summary_cache[cache_key]
            
            span.set_attribute("cache_hit", False)
            daily_summaries = []
            start_date = None
            end_date = None
            
            # Get days 2-7 (6 days total, starting from yesterday)
            for i in range(6):
                target_date = historical_end_date - timedelta(days=i)
                daily_summary = await self._get_historical_daily_summary(guild_id, user_id, target_date)
                if daily_summary:
                    daily_summaries.append(f"<daily_summary>\n<date>{target_date}</date>\n<summary>{daily_summary}</summary>\n</daily_summary>")
                    if start_date is None:
                        start_date = target_date
                    end_date = target_date

            span.set_attribute("daily_summaries_count", len(daily_summaries))
            if not daily_summaries:
                return None

            user_nick = await self._user_resolver.get_display_name(guild_id, user_id)
            # Create date range string
            date_range = f"{end_date} to {start_date}" if start_date and end_date else "recent days"
            
            # Create structured prompt with XML data
            structured_prompt = f"""{SUMMARIZE_HISTORICAL_PROMPT}

<user_name>{user_nick}</user_name>
<date_range>{date_range}</date_range>
<daily_summaries>
{"\n".join(daily_summaries)}
</daily_summaries>"""
            
            response = await self._gemma_client.generate_content(
                message=structured_prompt,
                response_schema=MemoryContext,
                temperature=0
            )
            historical_summary = response.context

            self._historical_summary_cache[cache_key] = historical_summary
            return historical_summary

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