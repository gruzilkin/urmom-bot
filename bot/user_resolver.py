import logging
import re
from typing import Optional, Dict
from cachetools import LRUCache

import nextcord

logger = logging.getLogger(__name__)


class UserResolver:
    """Service for resolving user IDs to display names and vice-versa."""

    def __init__(self):
        """Initializes the UserResolver without a bot client."""
        self._bot: Optional[nextcord.Client] = None
        self._display_name_cache = LRUCache(maxsize=500)
        self._user_id_cache = LRUCache(maxsize=500)

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
        cache_key = (guild_id, user_id)
        if cache_key in self._display_name_cache:
            return self._display_name_cache[cache_key]

        if self._bot is None:
            logger.error("UserResolver has not been initialized with the bot client.")
            fallback = f"User(ID:{user_id})"
            self._display_name_cache[cache_key] = fallback
            return fallback

        try:
            guild = self._bot.get_guild(guild_id)
            display_name = f"User(ID:{user_id})"

            if guild:
                member = guild.get_member(user_id)
                if member:
                    display_name = member.display_name
                else:
                    try:
                        member = await guild.fetch_member(user_id)
                        display_name = member.display_name
                    except nextcord.NotFound:
                        user = self._bot.get_user(user_id)
                        if user:
                            display_name = user.name
                        else:
                            try:
                                user = await self._bot.fetch_user(user_id)
                                display_name = user.name
                            except nextcord.NotFound:
                                logger.warning(f"User {user_id} not found")
                    except nextcord.HTTPException as e:
                        logger.warning(f"HTTP error fetching member {user_id}: {e}")
                        user = self._bot.get_user(user_id)
                        if user:
                            display_name = user.name
            else:
                logger.warning(f"Could not find guild with ID {guild_id}")

            self._display_name_cache[cache_key] = display_name
            return display_name
        except Exception as e:
            logger.error(f"Error resolving user {user_id} in guild {guild_id}: {e}")
            fallback = f"User(ID:{user_id})"
            self._display_name_cache[cache_key] = fallback
            return fallback

    async def resolve_user_id(self, guild_id: int, user_mention_or_name: str) -> Optional[int]:
        """Resolves a user mention, ID, or name/nickname to a user ID."""
        cache_key = (guild_id, user_mention_or_name)
        if cache_key in self._user_id_cache:
            return self._user_id_cache[cache_key]

        if self._bot is None:
            logger.error("UserResolver has not been initialized with the bot client.")
            self._user_id_cache[cache_key] = None
            return None

        # Handle Discord mention format <@user_id> or <@!user_id>
        discord_mention_match = re.match(r'<@!?(\d+)>', user_mention_or_name)
        if discord_mention_match:
            user_id = int(discord_mention_match.group(1))
            self._user_id_cache[cache_key] = user_id
            return user_id

        # Handle raw user ID
        if user_mention_or_name.isdigit():
            user_id = int(user_mention_or_name)
            self._user_id_cache[cache_key] = user_id
            return user_id

        guild = self._bot.get_guild(guild_id)
        if not guild:
            logger.warning(f"Could not find guild with ID {guild_id} for user resolution")
            self._user_id_cache[cache_key] = None
            return None

        # Search by name#discriminator in cache first
        member = guild.get_member_named(user_mention_or_name)
        if member:
            self._user_id_cache[cache_key] = member.id
            return member.id

        # Fallback to searching cached members by name/nickname
        members = guild.members
        for m in members:
            if m.name.lower() == user_mention_or_name.lower() or (m.nick and m.nick.lower() == user_mention_or_name.lower()):
                self._user_id_cache[cache_key] = m.id
                return m.id

        # If not found in cache, we could make API calls but that's expensive
        # For name-based resolution, we'll rely on the cached member list
        logger.warning(f"Could not resolve '{user_mention_or_name}' to a user in guild {guild_id}")
        self._user_id_cache[cache_key] = None
        return None

    async def replace_user_mentions_with_names(self, text: str, guild_id: int) -> str:
        """Replace Discord user mentions <@user_id> in text with actual display names."""
        if not text:
            return text
            
        # Find all user mentions in the text
        mention_pattern = r'<@!?(\d+)>'
        mentions = re.findall(mention_pattern, text)
        
        if not mentions:
            return text
            
        # Replace each mention with the actual display name
        result = text
        for user_id_str in mentions:
            user_id = int(user_id_str)
            display_name = await self.get_display_name(guild_id, user_id)
            
            # Replace both <@user_id> and <@!user_id> formats
            result = re.sub(f'<@!?{user_id}>', display_name, result)
            
        return result