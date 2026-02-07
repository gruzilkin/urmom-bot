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
_LOCK_TTL = 600  # 10 minutes


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

    # ── merged context ──────────────────────────────────────────────

    async def get_context(self, guild_id: int, user_id: int, facts_hash: str, summaries_hash: str) -> str | None:
        async with self._telemetry.async_create_span("redis.get_context") as span:
            key = f"ctx:{guild_id}:{user_id}:{facts_hash}:{summaries_hash}"
            span.set_attribute("key", key)
            try:
                result = await self._redis.get(key)
                span.set_attribute("hit", result is not None)
                return result
            except Exception as e:
                span.record_exception(e)
                logger.warning(f"Redis get_context failed: {e}")
                return None

    async def set_context(
        self, guild_id: int, user_id: int, facts_hash: str, summaries_hash: str, context: str
    ) -> None:
        async with self._telemetry.async_create_span("redis.set_context") as span:
            key = f"ctx:{guild_id}:{user_id}:{facts_hash}:{summaries_hash}"
            span.set_attribute("key", key)
            try:
                await self._redis.set(key, context, ex=_ONE_DAY)
            except Exception as e:
                span.record_exception(e)
                logger.warning(f"Redis set_context failed: {e}")

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

    # ── build lock ──────────────────────────────────────────────────

    async def try_acquire_build_lock(self, guild_id: int, for_date: date) -> bool:
        async with self._telemetry.async_create_span("redis.try_acquire_build_lock") as span:
            key = f"lock:daily:{guild_id}:{for_date}"
            span.set_attribute("key", key)
            try:
                acquired = bool(await self._redis.set(key, "1", nx=True, ex=_LOCK_TTL))
                span.set_attribute("acquired", acquired)
                return acquired
            except Exception as e:
                span.record_exception(e)
                logger.warning(f"Redis try_acquire_build_lock failed: {e}")
                return False

    async def release_build_lock(self, guild_id: int, for_date: date) -> None:
        async with self._telemetry.async_create_span("redis.release_build_lock") as span:
            key = f"lock:daily:{guild_id}:{for_date}"
            span.set_attribute("key", key)
            try:
                await self._redis.delete(key)
            except Exception as e:
                span.record_exception(e)
                logger.warning(f"Redis release_build_lock failed: {e}")

    # ── lifecycle ───────────────────────────────────────────────────

    async def close(self) -> None:
        await self._redis.aclose()
