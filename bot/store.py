import psycopg
from psycopg import AsyncConnection
from dataclasses import dataclass
import logging
from cachetools import LRUCache, cachedmethod
from typing import Dict, Tuple
from datetime import datetime, date, timedelta

logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    guild_id: int
    channel_id: int
    message_id: int
    user_id: int
    message_text: str
    timestamp: datetime

@dataclass
class GuildConfig:
    guild_id: int
    archive_channel_id: int = 0
    delete_jokes_after_minutes: int = 0
    downvote_reaction_threshold: int = 0
    enable_country_jokes: bool = True

class Store:
    def __init__(self, host: str = "localhost", 
                 port: int = 5432,
                 user: str = "postgres",
                 password: str = "postgres",
                 database: str = "postgres",
                 weight_coef: float = 1.1):
        self.connection_params = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "dbname": database
        }
        self.conn: AsyncConnection | None = None
        self.weight_coef = weight_coef
        self._guild_configs = {}  # Add cache dictionary
        self._user_facts_cache = LRUCache(maxsize=500)

    async def _connect(self) -> AsyncConnection:
        conn = await psycopg.AsyncConnection.connect(**self.connection_params)
        return conn

    async def _ensure_connection(self) -> None:
        if self.conn is None or self.conn.closed:
            if self.conn is not None:
                try:
                    await self.conn.close()
                except:
                    pass
            self.conn = await self._connect()
    
    async def save(self, source_message_id: int, 
            joke_message_id: int,
            source_message_content: str,
            joke_message_content: str,
            reaction_count: int,
            source_language: str,
            joke_language: str) -> None:
        
        # Print joke data
        joke_data = {
            "source_message_id": source_message_id,
            "joke_message_id": joke_message_id,
            "source_message_content": source_message_content,
            "joke_message_content": joke_message_content,
            "reaction_count": reaction_count,
            "source_language": source_language,
            "joke_language": joke_language
        }
        logger.info(f"Store saving joke: {joke_data}")
        
        try:
            await self._ensure_connection()
            async with self.conn.cursor() as cur:
                # Insert both messages in a single statement
                await cur.execute(
                    """
                    INSERT INTO messages (message_id, content, language_code) 
                    VALUES (%s, %s, %s), (%s, %s, %s)
                    ON CONFLICT (message_id) 
                    DO UPDATE SET content = EXCLUDED.content,
                                language_code = EXCLUDED.language_code
                    """,
                    (source_message_id, source_message_content, source_language,
                     joke_message_id, joke_message_content, joke_language)
                )
                
                # Create or update relationship in jokes table
                await cur.execute(
                    """
                    INSERT INTO jokes (source_message_id, joke_message_id, reaction_count) 
                    VALUES (%s, %s, %s)
                    ON CONFLICT (source_message_id, joke_message_id) 
                    DO UPDATE SET reaction_count = EXCLUDED.reaction_count
                    """,
                    (source_message_id, joke_message_id, reaction_count)
                )
                
                await self.conn.commit()
        except Exception as e:
            await self.conn.rollback()
            raise e

    async def get_random_jokes(self, n: int, language: str) -> list[tuple[str, str]]:
        """Get multiple random jokes in a single query"""
        try:
            await self._ensure_connection()
            async with self.conn.cursor() as cur:
                params = [language, self.weight_coef, n]
                
                await cur.execute(
                    f"""
                    SELECT m1.content, m2.content
                    FROM jokes j
                    JOIN messages m1 ON j.source_message_id = m1.message_id
                    JOIN messages m2 ON j.joke_message_id = m2.message_id
                    WHERE m1.language_code = %s
                    ORDER BY random() * power(%s, j.reaction_count) DESC
                    LIMIT %s
                    """,
                    params
                )
                return await cur.fetchall()
        except Exception as e:
            print(f"Error fetching random jokes: {e}")
            return []

    async def get_guild_config(self, guild_id: int) -> GuildConfig:
        if guild_id not in self._guild_configs:
            await self._ensure_connection()
            async with self.conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO guild_configs (guild_id) VALUES (%s) ON CONFLICT DO NOTHING",
                    (guild_id,)
                )
                
                await cur.execute(
                    """
                    SELECT archive_channel_id, delete_jokes_after_minutes, 
                           downvote_reaction_threshold, enable_country_jokes 
                    FROM guild_configs 
                    WHERE guild_id = %s
                    """,
                    (guild_id,)
                )
                result = await cur.fetchone()
                await self.conn.commit()
                
                self._guild_configs[guild_id] = GuildConfig(
                    guild_id=guild_id,
                    archive_channel_id=result[0],
                    delete_jokes_after_minutes=result[1],
                    downvote_reaction_threshold=result[2],
                    enable_country_jokes=result[3]
                )
        
        return self._guild_configs[guild_id]

    async def save_guild_config(self, config: GuildConfig) -> None:
        await self._ensure_connection()
        async with self.conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE guild_configs 
                SET archive_channel_id = %s,
                    delete_jokes_after_minutes = %s,
                    downvote_reaction_threshold = %s,
                    enable_country_jokes = %s
                WHERE guild_id = %s
                """,
                (config.archive_channel_id, config.delete_jokes_after_minutes,
                 config.downvote_reaction_threshold, config.enable_country_jokes,
                 config.guild_id)
            )
            await self.conn.commit()
            # Update cache
            self._guild_configs[config.guild_id] = config

    async def get_user_facts(self, guild_id: int, user_id: int) -> str | None:
        """Retrieve current memory blob for a user with LRU caching."""
        cache_key = (guild_id, user_id)
        
        # Check cache first
        if cache_key in self._user_facts_cache:
            return self._user_facts_cache[cache_key]
        
        try:
            await self._ensure_connection()
            async with self.conn.cursor() as cur:
                await cur.execute(
                    "SELECT memory_blob FROM user_facts WHERE guild_id = %s AND user_id = %s",
                    (guild_id, user_id)
                )
                result = await cur.fetchone()
                memory_blob = result[0] if result else None
                
                # Cache the result
                self._user_facts_cache[cache_key] = memory_blob
                return memory_blob
        except Exception as e:
            logger.error(f"Error retrieving user facts: {e}")
            return None

    async def save_user_facts(self, guild_id: int, user_id: int, memory_blob: str) -> None:
        """Save or update memory blob for a user and invalidate cache."""
        try:
            await self._ensure_connection()
            async with self.conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO user_facts (guild_id, user_id, memory_blob, updated_at)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (guild_id, user_id)
                    DO UPDATE SET memory_blob = EXCLUDED.memory_blob, updated_at = CURRENT_TIMESTAMP
                    """,
                    (guild_id, user_id, memory_blob)
                )
                await self.conn.commit()
                
                # Invalidate the cache for this specific user
                cache_key = (guild_id, user_id)
                if cache_key in self._user_facts_cache:
                    del self._user_facts_cache[cache_key]
                logger.debug(f"Saved and invalidated user facts cache for user {user_id} in guild {guild_id}")
                    
        except Exception as e:
            logger.error(f"Error saving user facts: {e}")
            if self.conn:
                await self.conn.rollback()
            raise e


    async def get_chat_messages_for_date(self, guild_id: int, for_date: date) -> list[ChatMessage]:
        """Retrieves all chat messages for a guild on a specific date."""
        try:
            await self._ensure_connection()
            async with self.conn.cursor() as cur:
                # Calculate start and end timestamps for the date
                start_time = datetime.combine(for_date, datetime.min.time())
                end_time = start_time + timedelta(days=1)
                
                await cur.execute(
                    """
                    SELECT guild_id, channel_id, message_id, user_id, message_text, timestamp
                    FROM chat_history
                    WHERE guild_id = %s AND timestamp >= %s AND timestamp < %s
                    ORDER BY timestamp ASC
                    """,
                    (guild_id, start_time, end_time)
                )
                results = await cur.fetchall()
                return [ChatMessage(*row) for row in results]
        except Exception as e:
            logger.error(f"Error retrieving chat messages for date: {e}")
            return []

    async def add_chat_message(self, guild_id: int, channel_id: int, message_id: int, user_id: int, message_text: str, timestamp: datetime) -> None:
        """Adds a chat message to the chat_history table."""
        try:
            await self._ensure_connection()
            async with self.conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO chat_history (guild_id, channel_id, message_id, user_id, message_text, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (message_id) DO NOTHING
                    """,
                    (guild_id, channel_id, message_id, user_id, message_text, timestamp)
                )
                await self.conn.commit()
        except Exception as e:
            logger.error(f"Error adding chat message: {e}")
            if self.conn:
                await self.conn.rollback()
            raise e


    async def close(self) -> None:
        """Close the database connection."""
        try:
            if self.conn is not None:
                await self.conn.close()
        except Exception:
            # Ignore exceptions during cleanup
            pass
