from datetime import date, datetime


class NullRedisCache:
    """In-memory test double for RedisCache following the NullTelemetry pattern."""

    def __init__(self) -> None:
        self._daily_summaries: dict[str, tuple[dict[int, str], datetime]] = {}
        self._contexts: dict[str, str] = {}
        self._articles: dict[str, str] = {}
        self._attachments: dict[int, str] = {}
        self._aliases: dict[str, list[str]] = {}
        self._locks: set[str] = set()

    async def get_daily_summary(self, guild_id: int, for_date: date) -> tuple[dict[int, str], datetime] | None:
        key = f"{guild_id}:{for_date}"
        return self._daily_summaries.get(key)

    async def set_daily_summary(
        self, guild_id: int, for_date: date, summaries: dict[int, str], created_at: datetime
    ) -> None:
        key = f"{guild_id}:{for_date}"
        self._daily_summaries[key] = (summaries, created_at)

    async def get_context(self, guild_id: int, user_id: int, facts_hash: str, summaries_hash: str) -> str | None:
        key = f"{guild_id}:{user_id}:{facts_hash}:{summaries_hash}"
        return self._contexts.get(key)

    async def set_context(
        self, guild_id: int, user_id: int, facts_hash: str, summaries_hash: str, context: str
    ) -> None:
        key = f"{guild_id}:{user_id}:{facts_hash}:{summaries_hash}"
        self._contexts[key] = context

    async def get_article(self, url: str) -> str | None:
        return self._articles.get(url)

    async def set_article(self, url: str, content: str) -> None:
        self._articles[url] = content

    async def get_attachment(self, attachment_id: int) -> str | None:
        return self._attachments.get(attachment_id)

    async def set_attachment(self, attachment_id: int, embedding_text: str) -> None:
        self._attachments[attachment_id] = embedding_text

    async def get_aliases(self, facts_hash: str) -> list[str] | None:
        return self._aliases.get(facts_hash)

    async def set_aliases(self, facts_hash: str, aliases: list[str]) -> None:
        self._aliases[facts_hash] = aliases

    async def try_acquire_build_lock(self, guild_id: int, for_date: date) -> bool:
        key = f"{guild_id}:{for_date}"
        if key in self._locks:
            return False
        self._locks.add(key)
        return True

    async def release_build_lock(self, guild_id: int, for_date: date) -> None:
        key = f"{guild_id}:{for_date}"
        self._locks.discard(key)

    async def close(self) -> None:
        pass
