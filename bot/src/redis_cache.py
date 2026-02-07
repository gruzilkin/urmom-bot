import hashlib
import json
import logging
from datetime import date, datetime

import redis.asyncio as redis

logger = logging.getLogger(__name__)

# TTLs in seconds
_ONE_DAY = 86400
_SEVEN_DAYS = 7 * _ONE_DAY
_LOCK_TTL = 600  # 10 minutes


class RedisCache:
    def __init__(self, host: str, port: int) -> None:
        self._redis = redis.Redis(
            host=host,
            port=port,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )

    # ── daily summaries ─────────────────────────────────────────────

    async def get_daily_summary(self, guild_id: int, for_date: date) -> tuple[dict[int, str], datetime] | None:
        try:
            raw = await self._redis.get(f"daily_summary:{guild_id}:{for_date}")
            if raw is None:
                return None
            data = json.loads(raw)
            summaries = {int(k): v for k, v in data["summaries"].items()}
            created_at = datetime.fromisoformat(data["created_at"])
            return summaries, created_at
        except Exception as e:
            logger.warning(f"Redis get_daily_summary failed: {e}")
            return None

    async def set_daily_summary(
        self, guild_id: int, for_date: date, summaries: dict[int, str], created_at: datetime
    ) -> None:
        try:
            data = {
                "summaries": {str(k): v for k, v in summaries.items()},
                "created_at": created_at.isoformat(),
            }
            await self._redis.set(
                f"daily_summary:{guild_id}:{for_date}",
                json.dumps(data),
                ex=_ONE_DAY,
            )
        except Exception as e:
            logger.warning(f"Redis set_daily_summary failed: {e}")

    # ── merged context ──────────────────────────────────────────────

    async def get_context(self, guild_id: int, user_id: int, facts_hash: str, summaries_hash: str) -> str | None:
        try:
            return await self._redis.get(f"ctx:{guild_id}:{user_id}:{facts_hash}:{summaries_hash}")
        except Exception as e:
            logger.warning(f"Redis get_context failed: {e}")
            return None

    async def set_context(
        self, guild_id: int, user_id: int, facts_hash: str, summaries_hash: str, context: str
    ) -> None:
        try:
            await self._redis.set(
                f"ctx:{guild_id}:{user_id}:{facts_hash}:{summaries_hash}",
                context,
                ex=_ONE_DAY,
            )
        except Exception as e:
            logger.warning(f"Redis set_context failed: {e}")

    # ── articles ────────────────────────────────────────────────────

    def _hash_url(self, url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()

    async def get_article(self, url: str) -> str | None:
        try:
            return await self._redis.get(f"article:{self._hash_url(url)}")
        except Exception as e:
            logger.warning(f"Redis get_article failed: {e}")
            return None

    async def set_article(self, url: str, content: str) -> None:
        try:
            await self._redis.set(f"article:{self._hash_url(url)}", content, ex=_SEVEN_DAYS)
        except Exception as e:
            logger.warning(f"Redis set_article failed: {e}")

    # ── attachments ─────────────────────────────────────────────────

    async def get_attachment(self, attachment_id: int) -> str | None:
        try:
            return await self._redis.get(f"attachment:{attachment_id}")
        except Exception as e:
            logger.warning(f"Redis get_attachment failed: {e}")
            return None

    async def set_attachment(self, attachment_id: int, embedding_text: str) -> None:
        try:
            await self._redis.set(f"attachment:{attachment_id}", embedding_text, ex=_SEVEN_DAYS)
        except Exception as e:
            logger.warning(f"Redis set_attachment failed: {e}")

    # ── build lock ──────────────────────────────────────────────────

    async def try_acquire_build_lock(self, guild_id: int, for_date: date) -> bool:
        try:
            return bool(await self._redis.set(f"lock:daily:{guild_id}:{for_date}", "1", nx=True, ex=_LOCK_TTL))
        except Exception as e:
            logger.warning(f"Redis try_acquire_build_lock failed: {e}")
            return False

    async def release_build_lock(self, guild_id: int, for_date: date) -> None:
        try:
            await self._redis.delete(f"lock:daily:{guild_id}:{for_date}")
        except Exception as e:
            logger.warning(f"Redis release_build_lock failed: {e}")

    # ── lifecycle ───────────────────────────────────────────────────

    async def close(self) -> None:
        await self._redis.aclose()
