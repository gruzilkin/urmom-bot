import psycopg
from psycopg import AsyncConnection
from dataclasses import dataclass
import logging
from typing import Any

logger = logging.getLogger(__name__)

@dataclass
class JokeRow:
    source_message_id: int
    joke_message_id: int
    source_content: str
    joke_content: str
    reaction_count: int

class WebStore:
    def __init__(self, telemetry: Any, host: str = "localhost", 
                 port: int = 5432,
                 user: str = "postgres",
                 password: str = "postgres",
                 database: str = "postgres"):
        self._telemetry = telemetry
        self.connection_params = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "dbname": database
        }
        self.conn: AsyncConnection | None = None

    async def _connect(self) -> AsyncConnection:
        conn = await psycopg.AsyncConnection.connect(**self.connection_params)
        return conn

    async def _ensure_connection(self) -> None:
        if self.conn is None or self.conn.closed:
            if self.conn is not None:
                try:
                    await self.conn.close()
                except Exception:
                    pass
            self.conn = await self._connect()

    async def get_jokes(self, limit: int = 50, offset: int = 0, search_query: str = "") -> list[JokeRow]:
        """Get paginated list of jokes with optional search filtering"""
        async with self._telemetry.async_create_span("get_jokes") as span:
            span.set_attribute("limit", limit)
            span.set_attribute("offset", offset)
            span.set_attribute("has_search", bool(search_query))
            span.set_attribute("query_length", len(search_query))
            
            try:
                await self._ensure_connection()
                async with self.conn.cursor() as cur:
                    search_pattern = f"%{search_query}%"
                    await cur.execute(
                        """
                        SELECT j.source_message_id, j.joke_message_id, 
                               m1.content as source_content, m2.content as joke_content,
                               j.reaction_count
                        FROM jokes j
                        JOIN messages m1 ON j.source_message_id = m1.message_id
                        JOIN messages m2 ON j.joke_message_id = m2.message_id
                        WHERE m1.content ILIKE %s OR m2.content ILIKE %s
                        ORDER BY j.source_message_id DESC
                        LIMIT %s OFFSET %s
                        """,
                        (search_pattern, search_pattern, limit, offset)
                    )
                    results = await cur.fetchall()
                    jokes = [JokeRow(*row) for row in results]
                    span.set_attribute("returned_count", len(jokes))
                    return jokes
            except Exception as e:
                logger.error(f"Error fetching jokes: {e}", exc_info=True)
                span.record_exception(e)
                return []

    async def get_jokes_count(self, search_query: str = "") -> int:
        """Get total count of jokes for pagination with optional search filtering"""
        async with self._telemetry.async_create_span("get_jokes_count") as span:
            span.set_attribute("has_search", bool(search_query))
            span.set_attribute("query_length", len(search_query))
            
            try:
                await self._ensure_connection()
                async with self.conn.cursor() as cur:
                    search_pattern = f"%{search_query}%"
                    await cur.execute(
                        """
                        SELECT COUNT(*)
                        FROM jokes j
                        JOIN messages m1 ON j.source_message_id = m1.message_id
                        JOIN messages m2 ON j.joke_message_id = m2.message_id
                        WHERE m1.content ILIKE %s OR m2.content ILIKE %s
                        """,
                        (search_pattern, search_pattern)
                    )
                    result = await cur.fetchone()
                    count = result[0] if result else 0
                    span.set_attribute("total_count", count)
                    return count
            except Exception as e:
                logger.error(f"Error getting jokes count: {e}", exc_info=True)
                span.record_exception(e)
                return 0

    async def get_message_content(self, message_id: int) -> str | None:
        """Get content of a specific message"""
        async with self._telemetry.async_create_span("get_message_content") as span:
            span.set_attribute("message_id", message_id)
            
            try:
                await self._ensure_connection()
                async with self.conn.cursor() as cur:
                    await cur.execute(
                        "SELECT content FROM messages WHERE message_id = %s",
                        (message_id,)
                    )
                    result = await cur.fetchone()
                    content = result[0] if result else None
                    span.set_attribute("found", content is not None)
                    return content
            except Exception as e:
                logger.error(f"Error getting message content: {e}", exc_info=True)
                span.record_exception(e)
                return None

    async def update_message_content(self, message_id: int, new_content: str) -> bool:
        """Update the content of a message"""
        async with self._telemetry.async_create_span("update_message_content") as span:
            span.set_attribute("message_id", message_id)
            span.set_attribute("content_length", len(new_content))
            
            try:
                await self._ensure_connection()
                async with self.conn.cursor() as cur:
                    await cur.execute(
                        "UPDATE messages SET content = %s WHERE message_id = %s",
                        (new_content, message_id)
                    )
                    rows_affected = cur.rowcount
                    await self.conn.commit()
                    
                    success = rows_affected > 0
                    span.set_attribute("success", success)
                    span.set_attribute("rows_affected", rows_affected)
                    return success
            except Exception as e:
                logger.error(f"Error updating message content: {e}", exc_info=True)
                span.record_exception(e)
                return False

    async def delete_joke(self, source_message_id: int, joke_message_id: int) -> bool:
        """Delete a joke pair and clean up all orphaned messages"""
        async with self._telemetry.async_create_span("delete_joke") as span:
            span.set_attribute("source_message_id", source_message_id)
            span.set_attribute("joke_message_id", joke_message_id)
            
            try:
                await self._ensure_connection()
                async with self.conn.cursor() as cur:
                    # Delete the joke relationship
                    await cur.execute(
                        "DELETE FROM jokes WHERE source_message_id = %s AND joke_message_id = %s",
                        (source_message_id, joke_message_id)
                    )
                    joke_deleted = cur.rowcount > 0
                    
                    # Clean up all orphaned messages in one query
                    await cur.execute(
                        """
                        DELETE FROM messages 
                        WHERE message_id NOT IN (
                            SELECT DISTINCT source_message_id FROM jokes
                            UNION
                            SELECT DISTINCT joke_message_id FROM jokes
                        )
                        """
                    )
                    orphans_deleted = cur.rowcount
                    
                    await self.conn.commit()
                    span.set_attribute("success", joke_deleted)
                    span.set_attribute("orphans_deleted", orphans_deleted)
                    return joke_deleted
            except Exception as e:
                logger.error(f"Error deleting joke: {e}", exc_info=True)
                span.record_exception(e)
                return False


    async def close(self) -> None:
        """Close the database connection."""
        try:
            if self.conn is not None:
                await self.conn.close()
        except Exception:
            pass