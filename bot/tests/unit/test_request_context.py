"""Unit tests for RequestContextBuilder."""

from datetime import datetime, timezone
import unittest
from unittest.mock import AsyncMock, Mock

from conversation_formatter import ConversationFormatter
from conversation_graph import ConversationMessage
from null_redis_cache import NullRedisCache
from null_telemetry import NullTelemetry
from request_context import RequestContextBuilder, collect_member_ids
from request_helpers import make_request, utc_store
from research_handoff import ResearchHandoffService

BOT_USER_ID = 999
GUILD_ID = 123


def message(message_id: int, author_id: int, content: str, mentioned: list[int] | None = None) -> ConversationMessage:
    return ConversationMessage(
        message_id=message_id,
        guild_id=GUILD_ID,
        author_id=author_id,
        content=content,
        timestamp=datetime(2026, 11, 2, 12, message_id, tzinfo=timezone.utc),
        mentioned_user_ids=mentioned or [],
    )


class TestCollectMemberIds(unittest.TestCase):
    def test_includes_authors_mentions_and_request_author(self) -> None:
        conversation = [message(1, 10, "hi <@20>", mentioned=[20]), message(2, 30, "yo")]
        request = make_request("question", author_id=40, guild_id=GUILD_ID)

        self.assertEqual(collect_member_ids(conversation, request), {10, 20, 30, 40})


class TestRequestContextBuilder(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.telemetry = NullTelemetry()
        user_resolver = Mock()
        user_resolver.replace_user_mentions_with_names = AsyncMock(side_effect=lambda content, guild_id: content)
        self.formatter = ConversationFormatter(user_resolver, utc_store())

        self.memory_manager = Mock()
        self.memory_manager.build_memory_prompt = AsyncMock(return_value="<memories>Lives in Tokyo</memories>")

        self.redis_cache = NullRedisCache()
        self.handoff_service = ResearchHandoffService(self.redis_cache, self.telemetry)

        self.builder = RequestContextBuilder(self.formatter, self.memory_manager, self.handoff_service, self.telemetry)

    async def test_builds_memories_for_all_members(self) -> None:
        conversation = [message(1, 10, "hi")]
        request = make_request("question", author_id=40, guild_id=GUILD_ID)

        context = await self.builder.build(request, conversation, BOT_USER_ID, extra_user_ids={50})

        self.memory_manager.build_memory_prompt.assert_awaited_once_with(GUILD_ID, {10, 40, 50})
        self.assertEqual(context.memories_block, "<memories>Lives in Tokyo</memories>")

    async def test_embeds_handoffs_of_bot_messages(self) -> None:
        await self.redis_cache.set_handoff(2, "Source: NTV, Nov 1", ttl_seconds=60)
        await self.redis_cache.set_handoff(1, "not a bot message", ttl_seconds=60)
        conversation = [message(1, 10, "what's new?"), message(2, BOT_USER_ID, "GMO update")]
        request = make_request("question", author_id=10, guild_id=GUILD_ID)

        context = await self.builder.build(request, conversation, BOT_USER_ID)

        self.assertIn("<research_handoff>Source: NTV, Nov 1</research_handoff>", context.conversation_block)
        self.assertNotIn("not a bot message", context.conversation_block)

    async def test_empty_conversation(self) -> None:
        request = make_request("question", author_id=10, guild_id=GUILD_ID)

        context = await self.builder.build(request, [], BOT_USER_ID)

        self.assertEqual(context.conversation_block, "")
        self.memory_manager.build_memory_prompt.assert_awaited_once_with(GUILD_ID, {10})


if __name__ == "__main__":
    unittest.main()
