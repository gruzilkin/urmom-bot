"""Unit tests for FamousPersonGenerator: the prompt carries memories, research handoffs, and shared instructions."""

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
import unittest
from unittest.mock import AsyncMock, Mock

from conversation_formatter import ConversationFormatter
from conversation_graph import ConversationMessage
from discord_formatting import DISCORD_FORMATTING_INSTRUCTIONS
from famous_person_generator import FamousPersonGenerator
from null_redis_cache import NullRedisCache
from null_telemetry import NullTelemetry
from request_context import RequestContextBuilder
from request_helpers import make_request, utc_store
from research_handoff import HANDOFF_CONTEXT_INSTRUCTIONS, ResearchHandoffService
from schemas import FamousParams

BOT_USER_ID = 999
GUILD_ID = 123
REQUEST_AUTHOR_ID = 42
MEMORIES_BLOCK = (
    "<memories>\n<memory><member_id>42</member_id><member_name>gruzilkin</member_name>"
    "<also_known_as>Медвед</also_known_as><facts>Lives in Tokyo</facts></memory>\n</memories>"
)


class TestFamousPersonGenerator(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.telemetry = NullTelemetry()

        user_resolver = Mock()
        user_resolver.replace_user_mentions_with_names = AsyncMock(side_effect=lambda content, guild_id: content)
        self.formatter = ConversationFormatter(user_resolver, utc_store())

        self.memory_manager = Mock()
        self.memory_manager.build_memory_prompt = AsyncMock(return_value=MEMORIES_BLOCK)

        redis_cache = NullRedisCache()
        await redis_cache.set_handoff(2, "Kumagai pay FY2025: about ¥40M (securities filing)", ttl_seconds=60)
        self.context_builder = RequestContextBuilder(
            self.formatter,
            self.memory_manager,
            ResearchHandoffService(redis_cache, self.telemetry),
            self.telemetry,
        )

        self.ai_client = Mock()
        self.ai_client.generate_content = AsyncMock()

        self.response_summarizer = Mock()
        self.response_summarizer.process_response = AsyncMock(side_effect=lambda text: text)

        self.request = make_request("what would Hesse say?", author_id=REQUEST_AUTHOR_ID, guild_id=GUILD_ID)
        self.conversation = [
            ConversationMessage(
                message_id=1,
                guild_id=GUILD_ID,
                author_id=REQUEST_AUTHOR_ID,
                content="how much does Kumagai make?",
                timestamp=datetime(2026, 11, 2, 8, 0, tzinfo=timezone.utc),
                mentioned_user_ids=[],
            ),
            ConversationMessage(
                message_id=2,
                guild_id=GUILD_ID,
                author_id=BOT_USER_ID,
                content="Very roughly ¥10M",
                timestamp=datetime(2026, 11, 2, 8, 1, tzinfo=timezone.utc),
                mentioned_user_ids=[],
                reply_to_id=1,
            ),
        ]

    async def _fetch_conversation(self) -> list[ConversationMessage]:
        return self.conversation

    def _assert_shared_context(self, prompt: str) -> None:
        self.assertIn("Lives in Tokyo", prompt)
        self.assertIn("<also_known_as>Медвед</also_known_as>", prompt)
        self.assertIn(
            "<research_handoff>Kumagai pay FY2025: about ¥40M (securities filing)</research_handoff>",
            prompt,
        )
        self.assertIn(HANDOFF_CONTEXT_INSTRUCTIONS, prompt)
        self.assertIn(DISCORD_FORMATTING_INSTRUCTIONS, prompt)
        member_ids = self.memory_manager.build_memory_prompt.call_args.args[1]
        self.assertIn(REQUEST_AUTHOR_ID, member_ids)

    async def _run_and_get_prompt(self, generate: Callable[[], Awaitable[str | None]]) -> str:
        await generate()
        self.ai_client.generate_content.assert_awaited_once()
        return self.ai_client.generate_content.call_args.kwargs["prompt"]

    async def test_famous_person_uses_shared_context(self) -> None:
        self.ai_client.generate_content.return_value = "Ah, my friend."
        generator = FamousPersonGenerator(
            self.ai_client, self.response_summarizer, self.telemetry, self.formatter, self.context_builder
        )
        params = FamousParams(famous_person="Hermann Hesse", language_code="en", language_name="English")

        prompt = await self._run_and_get_prompt(
            lambda: generator.handle_request(params, self.request, self._fetch_conversation, BOT_USER_ID)
        )

        self._assert_shared_context(prompt)
        self.assertNotIn("Florent", prompt)
        self.assertIn("Personalization:", prompt)


if __name__ == "__main__":
    unittest.main()
