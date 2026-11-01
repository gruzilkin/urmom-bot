import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

from null_telemetry import NullTelemetry

from ai_client import AIClient
from conversation_formatter import ConversationFormatter
from schemas import TaskHistorySummary
from store import ChatMessage, ScheduledTask, Store
from task_history_summarizer import TaskHistorySummarizer, trim_to_budget
from user_resolver import UserResolver

BOT_ID = 999
USER_ID = 300


def _msg(message_id: int, text: str, user_id: int = BOT_ID) -> ChatMessage:
    return ChatMessage(
        guild_id=100,
        channel_id=200,
        message_id=message_id,
        user_id=user_id,
        message_text=text,
        timestamp=datetime(2026, 5, 1, tzinfo=timezone.utc) + timedelta(days=message_id),
    )


def _task(chain: list[int]) -> ScheduledTask:
    now = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc)
    return ScheduledTask(
        task_id=1,
        guild_id=100,
        channel_id=200,
        creator_user_id=USER_ID,
        prompt="teach a new japanese word",
        cron_expression="0 9 * * *",
        timezone="Asia/Tokyo",
        next_run_at=now,
        last_run_at=None,
        created_at=now,
        updated_at=now,
        chain_message_ids=chain,
    )


class TestTrimToBudget(unittest.TestCase):
    def test_keeps_everything_within_budget(self):
        messages = [_msg(1, "aaa"), _msg(2, "bbb")]
        kept = trim_to_budget(messages, 10)
        self.assertEqual([m.message_id for m in kept], [1, 2])

    def test_drops_oldest_beyond_budget(self):
        messages = [_msg(1, "aaaa"), _msg(2, "bbbb"), _msg(3, "cccc")]
        kept = trim_to_budget(messages, 9)
        self.assertEqual([m.message_id for m in kept], [2, 3])

    def test_drops_newest_when_it_alone_exceeds_budget(self):
        messages = [_msg(1, "old"), _msg(2, "x" * 50)]
        self.assertEqual(trim_to_budget(messages, 10), [])

    def test_empty(self):
        self.assertEqual(trim_to_budget([], 10), [])


class TestTaskHistorySummarizer(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_ai = MagicMock(spec=AIClient)
        self.mock_ai.generate_content = AsyncMock(
            return_value=TaskHistorySummary(summary="- 猫 (cat)\n- 犬 (dog)", reason="vocabulary task")
        )
        self.mock_store = MagicMock(spec=Store)
        self.mock_store.get_guild_timezone = AsyncMock(return_value=ZoneInfo("Asia/Tokyo"))
        mock_user_resolver = MagicMock(spec=UserResolver)
        mock_user_resolver.replace_user_mentions_with_names = AsyncMock(side_effect=lambda content, guild_id: content)
        formatter = ConversationFormatter(mock_user_resolver, self.mock_store)
        self.summarizer = TaskHistorySummarizer(
            self.mock_ai, self.mock_store, formatter, NullTelemetry(), char_budget=1000
        )

    async def test_skips_when_few_previous_runs(self):
        self.mock_store.get_task_chain_messages = AsyncMock(
            return_value=[_msg(1, "request", user_id=USER_ID), _msg(2, "run 1"), _msg(3, "run 2")]
        )

        result = await self.summarizer.summarize(_task([1, 2, 3]), BOT_ID)

        self.assertIsNone(result)
        self.mock_ai.generate_content.assert_not_called()

    async def test_summarizes_chain_with_task_prompt(self):
        self.mock_store.get_task_chain_messages = AsyncMock(
            return_value=[_msg(1, "request", user_id=USER_ID), _msg(2, "猫"), _msg(3, "犬"), _msg(4, "鳥")]
        )

        result = await self.summarizer.summarize(_task([1, 2, 3, 4]), BOT_ID)

        self.assertEqual(result, "- 猫 (cat)\n- 犬 (dog)")
        self.mock_store.get_task_chain_messages.assert_awaited_once_with(1)
        kwargs = self.mock_ai.generate_content.await_args.kwargs
        self.assertIn("teach a new japanese word", kwargs["prompt"])
        self.assertIn(f"member_id {BOT_ID}", kwargs["prompt"])
        message = kwargs["message"]
        self.assertTrue(message.startswith("<conversation_history>"))
        self.assertLess(message.index("request"), message.index("鳥"))
        self.assertIn(f"<member_id>{USER_ID}</member_id>", message)
        self.assertIn("<timestamp>2026-05-02 Sat 09:00 +09:00</timestamp>", message)
        self.assertIs(kwargs["response_schema"], TaskHistorySummary)

    async def test_returns_none_on_ai_failure(self):
        self.mock_store.get_task_chain_messages = AsyncMock(return_value=[_msg(i, f"run {i}") for i in range(1, 5)])
        self.mock_ai.generate_content.side_effect = Exception("all clients failed")

        result = await self.summarizer.summarize(_task([1, 2, 3, 4]), BOT_ID)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
