import logging
import re
from typing import Optional
from cachetools import LRUCache

import nextcord
from opentelemetry.trace import SpanKind, Status, StatusCode

from open_telemetry import Telemetry

logger = logging.getLogger(__name__)


class UserResolver:
    """Service for resolving user IDs to display names and vice-versa."""

    def __init__(self, telemetry: Telemetry):
        """Initializes the UserResolver with telemetry."""
        self._bot: Optional[nextcord.Client] = None
        self._display_name_cache = LRUCache(maxsize=500)
        self._user_id_cache = LRUCache(maxsize=500)
        self.telemetry = telemetry

    def set_bot_client(self, bot: nextcord.Client):
        """Sets the Discord client instance to fully enable the resolver."""
        if self._bot is not None:
            logger.warning("Bot client is already set in UserResolver.")
            return
        self._bot = bot
        self._display_name_cache.clear()
        self._user_id_cache.clear()
        logger.info("UserResolver initialized with bot client.")

    async def get_display_name(self, guild_id: int, user_id: int) -> str:
        async with self.telemetry.async_create_span("get_display_name", SpanKind.CLIENT) as span:
            span.set_attribute("guild_id", guild_id)
            span.set_attribute("user_id", user_id)
            
            cache_key = (guild_id, user_id)
            if cache_key in self._display_name_cache:
                span.set_attribute("cache_hit", True)
                return self._display_name_cache[cache_key]

            span.set_attribute("cache_hit", False)

            if self._bot is None:
                logger.error("UserResolver has not been initialized with the bot client.")
                fallback = f"User(ID:{user_id})"
                self._display_name_cache[cache_key] = fallback
                span.set_status(Status(StatusCode.ERROR, "Bot client not initialized"))
                return fallback

            guild = self._bot.get_guild(guild_id)
            if not guild:
                logger.warning(f"Could not find guild with ID {guild_id}")
                fallback = f"User(ID:{user_id})"
                self._display_name_cache[cache_key] = fallback
                span.set_status(Status(StatusCode.ERROR, f"Guild {guild_id} not found"))
                return fallback

            # Try getting member from cache first
            member = guild.get_member(user_id)
            if member:
                display_name = member.display_name
                span.set_attribute("resolution_method", "guild_member_cache")
                self._display_name_cache[cache_key] = display_name
                span.set_attribute("display_name", display_name)
                return display_name

            # Try fetching member from API
            try:
                member = await guild.fetch_member(user_id)
                display_name = member.display_name
                span.set_attribute("resolution_method", "guild_member_fetch")
                self._display_name_cache[cache_key] = display_name
                span.set_attribute("display_name", display_name)
                return display_name
            except nextcord.NotFound:
                pass  # Continue to user resolution
            except nextcord.HTTPException as e:
                logger.warning(f"HTTP error fetching member {user_id}: {e}")
                span.record_exception(e)
                # Fall through to user resolution

            # Try getting user from cache
            user = self._bot.get_user(user_id)
            if user:
                display_name = user.name
                span.set_attribute("resolution_method", "user_cache")
                self._display_name_cache[cache_key] = display_name
                span.set_attribute("display_name", display_name)
                return display_name

            # Try fetching user from API
            try:
                user = await self._bot.fetch_user(user_id)
                display_name = user.name
                span.set_attribute("resolution_method", "user_fetch")
                self._display_name_cache[cache_key] = display_name
                span.set_attribute("display_name", display_name)
                return display_name
            except nextcord.NotFound:
                logger.warning(f"User {user_id} not found")
                fallback = f"User(ID:{user_id})"
                self._display_name_cache[cache_key] = fallback
                span.set_status(Status(StatusCode.ERROR, f"User {user_id} not found"))
                return fallback

    async def resolve_user_id(self, guild_id: int, user_mention_or_name: str) -> Optional[int]:
        """Resolves a user mention, ID, or name/nickname to a user ID."""
        async with self.telemetry.async_create_span("resolve_user_id", SpanKind.CLIENT) as span:
            span.set_attribute("guild_id", guild_id)
            span.set_attribute("user_mention_or_name", user_mention_or_name)
            
            cache_key = (guild_id, user_mention_or_name)
            if cache_key in self._user_id_cache:
                span.set_attribute("cache_hit", True)
                cached_result = self._user_id_cache[cache_key]
                span.set_attribute("user_id", cached_result if cached_result is not None else "none")
                return cached_result

            span.set_attribute("cache_hit", False)

            if self._bot is None:
                logger.error("UserResolver has not been initialized with the bot client.")
                self._user_id_cache[cache_key] = None
                span.set_status(Status(StatusCode.ERROR, "Bot client not initialized"))
                return None

            # Handle Discord mention format <@user_id> or <@!user_id>
            discord_mention_match = re.match(r'<@!?(\d+)>', user_mention_or_name)
            if discord_mention_match:
                user_id = int(discord_mention_match.group(1))
                self._user_id_cache[cache_key] = user_id
                span.set_attribute("resolution_method", "discord_mention")
                span.set_attribute("user_id", user_id)
                return user_id

            # Handle raw user ID
            if user_mention_or_name.isdigit():
                user_id = int(user_mention_or_name)
                self._user_id_cache[cache_key] = user_id
                span.set_attribute("resolution_method", "raw_user_id")
                span.set_attribute("user_id", user_id)
                return user_id

            guild = self._bot.get_guild(guild_id)
            if not guild:
                logger.warning(f"Could not find guild with ID {guild_id} for user resolution")
                self._user_id_cache[cache_key] = None
                span.set_status(Status(StatusCode.ERROR, f"Guild {guild_id} not found"))
                return None

            # Search by name#discriminator in cache first
            member = guild.get_member_named(user_mention_or_name)
            if member:
                self._user_id_cache[cache_key] = member.id
                span.set_attribute("resolution_method", "member_named")
                span.set_attribute("user_id", member.id)
                return member.id

            # Fallback to searching cached members by name/nickname
            members = guild.members
            for m in members:
                if m.name.lower() == user_mention_or_name.lower() or (m.nick and m.nick.lower() == user_mention_or_name.lower()):
                    self._user_id_cache[cache_key] = m.id
                    span.set_attribute("resolution_method", "member_search")
                    span.set_attribute("user_id", m.id)
                    return m.id

            # If not found in cache, we could make API calls but that's expensive
            # For name-based resolution, we'll rely on the cached member list
            logger.warning(f"Could not resolve '{user_mention_or_name}' to a user in guild {guild_id}")
            self._user_id_cache[cache_key] = None
            span.set_status(Status(StatusCode.ERROR, f"Could not resolve '{user_mention_or_name}' to a user"))
            return None

    async def replace_user_mentions_with_names(self, text: str, guild_id: int) -> str:
        """Replace Discord user mentions <@user_id> in text with actual display names."""
        async with self.telemetry.async_create_span("replace_user_mentions", SpanKind.CLIENT) as span:
            span.set_attribute("guild_id", guild_id)
            span.set_attribute("text_length", len(text) if text else 0)
            
            if not text:
                span.set_attribute("mentions_count", 0)
                return text
                
            # Find all user mentions in the text
            mention_pattern = r'<@!?(\d+)>'
            mentions = re.findall(mention_pattern, text)
            
            span.set_attribute("mentions_count", len(mentions))
            
            if not mentions:
                return text
                
            # Replace each mention with the actual display name
            result = text
            for user_id_str in mentions:
                user_id = int(user_id_str)
                display_name = await self.get_display_name(guild_id, user_id)
                
                # Replace both <@user_id> and <@!user_id> formats
                result = re.sub(f'<@!?{user_id}>', display_name, result)
                
            span.set_attribute("result_length", len(result))
            return result