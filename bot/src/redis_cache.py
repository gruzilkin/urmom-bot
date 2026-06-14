import hashlib
import json
import logging
from datetime import date, datetime

import redis.asyncio as redis

from open_telemetry import Telemetry

logger = logging.getLogger(__name__)

# TTLs in seconds
_ONE_DAY = 86400
_SEVEN_DAYS = 7 * _ONE_DAY


class RedisCache:
    def __init__(self, host: str, port: int, telemetry: Telemetry) -> None:
        self._redis = redis.Redis(
            host=host,
            port=port,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        self._telemetry = telemetry

    # ── daily summaries ─────────────────────────────────────────────

    async def get_daily_summary(self, guild_id: int, for_date: date) -> tuple[dict[int, str], datetime] | None:
        async with self._telemetry.async_create_span("redis.get_daily_summary") as span:
            key = f"daily_summary:{guild_id}:{for_date}"
            span.set_attribute("key", key)
            try:
                raw = await self._redis.get(key)
                if raw is None:
                    span.set_attribute("hit", False)
                    return None
                data = json.loads(raw)
                summaries = {int(k): v for k, v in data["summaries"].items()}
                created_at = datetime.fromisoformat(data["created_at"])
                span.set_attribute("hit", True)
                return summaries, created_at
            except Exception as e:
                span.record_exception(e)
                logger.warning(f"Redis get_daily_summary failed: {e}")
                return None

    async def set_daily_summary(
        self, guild_id: int, for_date: date, summaries: dict[int, str], created_at: datetime
    ) -> None:
        async with self._telemetry.async_create_span("redis.set_daily_summary") as span:
            key = f"daily_summary:{guild_id}:{for_date}"
            span.set_attribute("key", key)
            try:
                data = {
                    "summaries": {str(k): v for k, v in summaries.items()},
                    "created_at": created_at.isoformat(),
                }
                await self._redis.set(key, json.dumps(data), ex=_ONE_DAY)
            except Exception as e:
                span.record_exception(e)
                logger.warning(f"Redis set_daily_summary failed: {e}")

    # ── merged memory ───────────────────────────────────────────────

    async def get_memory(self, guild_id: int, user_id: int, facts_hash: str, summaries_hash: str) -> str | None:
        async with self._telemetry.async_create_span("redis.get_memory") as span:
            key = f"memory:{guild_id}:{user_id}:{facts_hash}:{summaries_hash}"
            span.set_attribute("key", key)
            try:
                result = await self._redis.get(key)
                span.set_attribute("hit", result is not None)
                return result
            except Exception as e:
                span.record_exception(e)
                logger.warning(f"Redis get_memory failed: {e}")
                return None

    async def set_memory(self, guild_id: int, user_id: int, facts_hash: str, summaries_hash: str, memory: str) -> None:
        """Cache a merged memory under both its content-addressed key and the hash-independent latest key.

        The content-addressed key serves exact-input cache hits; the latest key gives readers a
        last-known-good value to fall back on while inputs change, so the read path never blocks.
        """
        async with self._telemetry.async_create_span("redis.set_memory") as span:
            key = f"memory:{guild_id}:{user_id}:{facts_hash}:{summaries_hash}"
            latest_key = f"memory_latest:{guild_id}:{user_id}"
            span.set_attribute("key", key)
            try:
                await self._redis.set(key, memory, ex=_ONE_DAY)
                await self._redis.set(latest_key, memory, ex=_SEVEN_DAYS)
            except Exception as e:
                span.record_exception(e)
                logger.warning(f"Redis set_memory failed: {e}")

    async def get_memory_latest(self, guild_id: int, user_id: int) -> str | None:
        async with self._telemetry.async_create_span("redis.get_memory_latest") as span:
            key = f"memory_latest:{guild_id}:{user_id}"
            span.set_attribute("key", key)
            try:
                result = await self._redis.get(key)
                span.set_attribute("hit", result is not None)
                return result
            except Exception as e:
                span.record_exception(e)
                logger.warning(f"Redis get_memory_latest failed: {e}")
                return None

    # ── articles ────────────────────────────────────────────────────

    def _hash_url(self, url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()

    async def get_article(self, url: str) -> str | None:
        async with self._telemetry.async_create_span("redis.get_article") as span:
            key = f"article:{self._hash_url(url)}"
            span.set_attribute("key", key)
            try:
                result = await self._redis.get(key)
                span.set_attribute("hit", result is not None)
                return result
            except Exception as e:
                span.record_exception(e)
                logger.warning(f"Redis get_article failed: {e}")
                return None

    async def set_article(self, url: str, content: str) -> None:
        async with self._telemetry.async_create_span("redis.set_article") as span:
            key = f"article:{self._hash_url(url)}"
            span.set_attribute("key", key)
            try:
                await self._redis.set(key, content, ex=_SEVEN_DAYS)
            except Exception as e:
                span.record_exception(e)
                logger.warning(f"Redis set_article failed: {e}")

    # ── attachments ─────────────────────────────────────────────────

    async def get_attachment(self, attachment_id: int) -> str | None:
        async with self._telemetry.async_create_span("redis.get_attachment") as span:
            key = f"attachment:{attachment_id}"
            span.set_attribute("key", key)
            try:
                result = await self._redis.get(key)
                span.set_attribute("hit", result is not None)
                return result
            except Exception as e:
                span.record_exception(e)
                logger.warning(f"Redis get_attachment failed: {e}")
                return None

    async def set_attachment(self, attachment_id: int, embedding_text: str) -> None:
        async with self._telemetry.async_create_span("redis.set_attachment") as span:
            key = f"attachment:{attachment_id}"
            span.set_attribute("key", key)
            try:
                await self._redis.set(key, embedding_text, ex=_SEVEN_DAYS)
            except Exception as e:
                span.record_exception(e)
                logger.warning(f"Redis set_attachment failed: {e}")

    # ── aliases ───────────────────────────────────────────────────────

    async def get_aliases(self, facts_hash: str) -> list[str] | None:
        async with self._telemetry.async_create_span("redis.get_aliases") as span:
            key = f"aliases:{facts_hash}"
            span.set_attribute("key", key)
            try:
                raw = await self._redis.get(key)
                if raw is None:
                    span.set_attribute("hit", False)
                    return None
                span.set_attribute("hit", True)
                return json.loads(raw)
            except Exception as e:
                span.record_exception(e)
                logger.error(f"Redis get_aliases failed: {e}", exc_info=True)
                return None

    async def set_aliases(self, facts_hash: str, aliases: list[str]) -> None:
        async with self._telemetry.async_create_span("redis.set_aliases") as span:
            key = f"aliases:{facts_hash}"
            span.set_attribute("key", key)
            try:
                await self._redis.set(key, json.dumps(aliases), ex=_SEVEN_DAYS)
            except Exception as e:
                span.record_exception(e)
                logger.error(f"Redis set_aliases failed: {e}", exc_info=True)

    # ── lifecycle ───────────────────────────────────────────────────

    async def close(self) -> None:
        await self._redis.aclose()
