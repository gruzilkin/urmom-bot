import unittest
from unittest.mock import AsyncMock, MagicMock

from conversation_graph import ConversationMessage
from null_telemetry import NullTelemetry
from redis_cache import RedisCache
from research_handoff import (
    HANDOFF_TTL_SECONDS,
    ProcessorResult,
    ResearchHandoffService,
)

BOT_ID = 999


def _msg(message_id: int, author_id: int, reply_to_id: int | None = None) -> ConversationMessage:
    return ConversationMessage(
        message_id=message_id,
        author_id=author_id,
        content=f"message {message_id}",
        timestamp="2026-09-20 10:00:00",
        mentioned_user_ids=[],
        reply_to_id=reply_to_id,
    )


def _payload(note: str = "Found X") -> str:
    return f"Verified: {note} (https://example.com/x)."


class TestProcessorResult(unittest.TestCase):
    def test_handoff_defaults_to_none(self):
        # Processors that do not generate handoffs can return plain text results
        result = ProcessorResult(text="hello")
        self.assertIsNone(result.handoff)


class TestSave(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.redis_cache = MagicMock(spec=RedisCache)
        self.redis_cache.set_handoff = AsyncMock()
        self.service = ResearchHandoffService(redis_cache=self.redis_cache, telemetry=NullTelemetry())
        self.handoff = "Verified: found X (https://example.com/x)."

    async def test_stores_text_under_sent_message_with_ttl(self):
        await self.service.save(555, self.handoff)

        self.redis_cache.set_handoff.assert_awaited_once_with(555, self.handoff, HANDOFF_TTL_SECONDS)

    async def test_empty_handoff_is_not_stored(self):
        await self.service.save(555, None)
        await self.service.save(555, "")
        self.redis_cache.set_handoff.assert_not_called()

    async def test_storage_failure_is_swallowed(self):
        self.redis_cache.set_handoff.side_effect = RuntimeError("redis down")
        await self.service.save(555, self.handoff)  # must not raise


class TestLoadForConversation(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.redis_cache = MagicMock(spec=RedisCache)
        self.redis_cache.get_handoffs = AsyncMock(return_value={})
        self.service = ResearchHandoffService(redis_cache=self.redis_cache, telemetry=NullTelemetry())

    async def test_no_bot_messages_skips_storage(self):
        handoffs = await self.service.load_for_conversation([_msg(1, 5), _msg(2, 6)], BOT_ID)
        self.assertEqual(handoffs, {})
        self.redis_cache.get_handoffs.assert_not_called()

    async def test_only_bot_messages_are_looked_up(self):
        conversation = [_msg(1, BOT_ID), _msg(2, 5, reply_to_id=1), _msg(3, BOT_ID), _msg(4, 5)]
        self.redis_cache.get_handoffs.return_value = {1: _payload(note="first"), 3: _payload(note="third")}

        handoffs = await self.service.load_for_conversation(conversation, BOT_ID)

        self.redis_cache.get_handoffs.assert_awaited_once_with([1, 3])
        self.assertEqual(handoffs, {1: _payload(note="first"), 3: _payload(note="third")})

    async def test_missing_entries_yield_empty_mapping(self):
        handoffs = await self.service.load_for_conversation([_msg(1, BOT_ID), _msg(2, 5, reply_to_id=1)], BOT_ID)
        self.assertEqual(handoffs, {})

    async def test_retrieval_failure_degrades_to_empty_mapping(self):
        self.redis_cache.get_handoffs.side_effect = RuntimeError("redis down")
        handoffs = await self.service.load_for_conversation([_msg(1, BOT_ID), _msg(2, 5)], BOT_ID)
        self.assertEqual(handoffs, {})


class TestRedisCacheHandoffs(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cache = RedisCache.__new__(RedisCache)
        self.cache._redis = MagicMock()
        self.cache._redis.mget = AsyncMock()
        self.cache._redis.set = AsyncMock()
        self.cache._telemetry = NullTelemetry()

    async def test_get_handoffs_returns_hits_and_skips_misses(self):
        self.cache._redis.mget.return_value = [_payload(), None, "   "]

        found = await self.cache.get_handoffs([1, 2, 3])

        self.cache._redis.mget.assert_awaited_once_with(["handoff:1", "handoff:2", "handoff:3"])
        self.assertEqual(found, {1: _payload()})

    async def test_get_handoffs_with_no_ids_skips_redis(self):
        self.assertEqual(await self.cache.get_handoffs([]), {})
        self.cache._redis.mget.assert_not_called()

    async def test_get_handoffs_failure_returns_empty(self):
        self.cache._redis.mget.side_effect = ConnectionError("down")
        self.assertEqual(await self.cache.get_handoffs([1]), {})

    async def test_set_handoff_uses_message_key_and_expiry(self):
        await self.cache.set_handoff(7, _payload(), 123)

        self.cache._redis.set.assert_awaited_once_with("handoff:7", _payload(), ex=123)

    async def test_set_handoff_failure_is_swallowed(self):
        self.cache._redis.set.side_effect = ConnectionError("down")
        await self.cache.set_handoff(7, _payload(), 123)  # must not raise


if __name__ == "__main__":
    unittest.main()
