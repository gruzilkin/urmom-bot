import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from null_telemetry import NullTelemetry

from ai_router import AiRouter
from general_query_generator import GeneralQueryGenerator
from research_handoff import ProcessorResult, ResearchHandoffService
from schedule_engine import CATCHUP_STALENESS, ScheduleEngine
from schemas import GeneralParams
from store import ScheduledTask, Store
from task_history_summarizer import TaskHistorySummarizer


def _make_task(
    task_id: int = 1,
    cron: str | None = "0 9 * * *",
    tz: str = "Asia/Tokyo",
    next_run_at: datetime | None = None,
    chain_message_ids: list[int] | None = None,
) -> ScheduledTask:
    now = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc)
    return ScheduledTask(
        task_id=task_id,
        guild_id=100,
        channel_id=200,
        creator_user_id=300,
        prompt="ping",
        cron_expression=cron,
        timezone=tz,
        next_run_at=next_run_at or now,
        last_run_at=None,
        created_at=now,
        updated_at=now,
        chain_message_ids=chain_message_ids or [],
    )


class TestScheduleEngine(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_store = MagicMock(spec=Store)
        self.mock_store.list_scheduled_tasks = AsyncMock(return_value=[])
        self.mock_store.update_task_next_run_at = AsyncMock()
        self.mock_store.delete_scheduled_task = AsyncMock(return_value=True)
        self.mock_store.record_task_run = AsyncMock()
        self.mock_store.list_active_user_ids = AsyncMock(return_value=[])

        self.mock_gqg = MagicMock(spec=GeneralQueryGenerator)
        self.mock_gqg.handle_request = AsyncMock(return_value=ProcessorResult(text="response text"))

        self.mock_handoff_service = MagicMock(spec=ResearchHandoffService)
        self.mock_handoff_service.save = AsyncMock()

        self.mock_router = MagicMock(spec=AiRouter)
        self.mock_router.extract_general_params = AsyncMock(
            return_value=GeneralParams(
                ai_backend="grok",
                temperature=0.7,
                cleaned_query="post a haiku",
                language_code="en",
                language_name="English",
            )
        )

        self.mock_summarizer = MagicMock(spec=TaskHistorySummarizer)
        self.mock_summarizer.summarize = AsyncMock(return_value=None)

        self.engine = ScheduleEngine(
            store=self.mock_store,
            telemetry=NullTelemetry(),
            general_query_generator=self.mock_gqg,
            handoff_service=self.mock_handoff_service,
            task_history_summarizer=self.mock_summarizer,
            chain_max_messages=100,
        )
        self.engine.ai_router = self.mock_router

        # Stub bot and channel fetcher
        self.mock_channel = MagicMock()
        self.mock_channel.send = AsyncMock(return_value=SimpleNamespace(id=555))
        self.mock_bot = MagicMock()
        self.mock_bot.get_channel = MagicMock(return_value=self.mock_channel)
        self.mock_bot.user = SimpleNamespace(id=999, name="bot")
        self.engine.bot = self.mock_bot
        self.engine._channel_conversation_fetcher = AsyncMock(return_value=[])

    # ---- _compute_next_after_fire ----

    def test_compute_next_after_fire_recurring(self):
        task = _make_task(cron="0 9 * * *", tz="Asia/Tokyo")
        now = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc)
        nxt = self.engine._compute_next_after_fire(task, now)
        self.assertIsNotNone(nxt)
        self.assertEqual(nxt.tzinfo, timezone.utc)
        self.assertGreater(nxt, now)

    def test_compute_next_after_fire_invalid_timezone_falls_back_to_utc(self):
        # cron "0 9 * * *" anchored at 12:00 UTC, with UTC fallback, should produce
        # tomorrow at 09:00 UTC. (If it had used Asia/Tokyo, it would be 00:00 UTC.)
        task = _make_task(cron="0 9 * * *", tz="Made/Up_Zone")
        now = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc)
        nxt = self.engine._compute_next_after_fire(task, now)
        self.assertEqual(nxt, datetime(2026, 5, 24, 9, 0, tzinfo=timezone.utc))

    # ---- _dispatch_due_task ----

    async def test_dispatch_one_off_deletes_row(self):
        task = _make_task(task_id=7, cron=None)
        now = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc)

        self.engine._execute = AsyncMock()

        await self.engine._dispatch_due_task(task, now)

        self.mock_store.delete_scheduled_task.assert_awaited_once_with(7)
        self.mock_store.update_task_next_run_at.assert_not_called()

    async def test_dispatch_skips_stale_recurring_firing(self):
        # Intended fire was > CATCHUP_STALENESS ago
        intended = datetime(2026, 5, 22, 0, 0, tzinfo=timezone.utc)
        now = intended + CATCHUP_STALENESS + timedelta(hours=1)
        task = _make_task(task_id=9, cron="0 9 * * *", next_run_at=intended)

        self.engine._execute = AsyncMock()

        await self.engine._dispatch_due_task(task, now)

        # Schedule advanced, but execute should NOT have been called
        self.mock_store.update_task_next_run_at.assert_awaited_once()
        self.engine._execute.assert_not_called()

    async def test_dispatch_one_off_fires_even_when_stale(self):
        # One-off tasks fire regardless of staleness (per spec)
        intended = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
        now = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc)
        task = _make_task(task_id=11, cron=None, next_run_at=intended)

        self.engine._execute = AsyncMock()

        await self.engine._dispatch_due_task(task, now)

        self.engine._execute.assert_called_once()

    # ---- _execute ----

    async def test_execute_skips_send_when_generator_returns_none(self):
        self.mock_gqg.handle_request.return_value = None
        task = _make_task()

        await self.engine._execute(task, intended_run_at=task.next_run_at, scheduled=True)

        self.mock_channel.send.assert_not_called()
        self.mock_store.record_task_run.assert_not_awaited()

    async def test_execute_handles_channel_unavailable(self):
        self.mock_bot.get_channel.return_value = None
        self.mock_bot.fetch_channel = AsyncMock(side_effect=Exception("404 not found"))
        task = _make_task()

        await self.engine._execute(task, intended_run_at=task.next_run_at, scheduled=True)

        # Should not crash; nothing sent
        self.mock_channel.send.assert_not_called()
        self.mock_gqg.handle_request.assert_not_called()

    async def test_fire_task_does_not_advance_schedule(self):
        task = _make_task(task_id=5)
        await self.engine.fire_task(task)

        # Neither the schedule-advance nor the one-off-delete path should fire for run_now
        self.mock_store.update_task_next_run_at.assert_not_called()
        self.mock_store.delete_scheduled_task.assert_not_called()
        # But the task should still be executed
        self.mock_gqg.handle_request.assert_awaited_once()

    # ---- _tick ----

    async def test_tick_no_due_tasks_returns_early(self):
        self.mock_store.list_scheduled_tasks.return_value = []
        await self.engine._tick()
        # No dispatching, no execution
        self.mock_store.update_task_next_run_at.assert_not_called()
        self.mock_store.delete_scheduled_task.assert_not_called()
        self.mock_gqg.handle_request.assert_not_called()

    async def test_tick_dispatches_each_due_task(self):
        t1 = _make_task(task_id=1, cron="0 9 * * *")
        t2 = _make_task(task_id=2, cron=None)  # one-off
        self.mock_store.list_scheduled_tasks.return_value = [t1, t2]

        self.engine._dispatch_due_task = AsyncMock()
        await self.engine._tick()

        self.assertEqual(self.engine._dispatch_due_task.await_count, 2)
        # Verify list call was filtered with due_before
        list_kwargs = self.mock_store.list_scheduled_tasks.await_args.kwargs
        self.assertIn("due_before", list_kwargs)
        self.assertEqual(list_kwargs["limit"], 20)

    # ---- start() ----

    async def test_start_launches_loop_and_is_idempotent(self):
        fetcher = AsyncMock(return_value=[])
        await self.engine.start(self.mock_bot, fetcher)
        self.assertIsNotNone(self.engine._loop_task)
        first_task = self.engine._loop_task

        # Second call should NOT replace the running task
        await self.engine.start(self.mock_bot, fetcher)
        self.assertIs(self.engine._loop_task, first_task)

        # Clean up the background loop
        self.engine._loop_task.cancel()
        try:
            await self.engine._loop_task
        except (asyncio.CancelledError, BaseException):
            pass

    # ---- _execute edge cases ----

    async def test_execute_returns_when_engine_not_started(self):
        self.engine.bot = None
        self.engine._channel_conversation_fetcher = None
        task = _make_task()

        await self.engine._execute(task, intended_run_at=task.next_run_at, scheduled=True)

        self.mock_gqg.handle_request.assert_not_called()
        self.mock_channel.send.assert_not_called()

    async def test_execute_swallows_generator_failure(self):
        self.mock_gqg.handle_request.side_effect = Exception("AI failed")
        task = _make_task()

        # Should not raise — outer catch logs and returns
        await self.engine._execute(task, intended_run_at=task.next_run_at, scheduled=True)

        self.mock_channel.send.assert_not_called()
        self.mock_store.record_task_run.assert_not_called()

    async def test_execute_fetches_channel_via_api_when_cache_misses(self):
        self.mock_bot.get_channel.return_value = None
        self.mock_bot.fetch_channel = AsyncMock(return_value=self.mock_channel)
        task = _make_task()

        await self.engine._execute(task, intended_run_at=task.next_run_at, scheduled=True)

        self.mock_bot.fetch_channel.assert_awaited_once_with(task.channel_id)
        self.mock_channel.send.assert_awaited_once_with("response text", suppress_embeds=True)

    async def test_execute_saves_handoff_against_sent_message(self):
        # The handoff must be associated with the message the engine actually sent,
        # not with the task or any triggering message.
        handoff = "Verified: X was introduced in version B (https://example.com/notes)."
        self.mock_gqg.handle_request.return_value = ProcessorResult(text="response text", handoff=handoff)
        task = _make_task()

        await self.engine._execute(task, intended_run_at=task.next_run_at, scheduled=True)

        self.mock_channel.send.assert_awaited_once_with("response text", suppress_embeds=True)
        self.mock_handoff_service.save.assert_awaited_once_with(555, handoff)
        self.mock_store.record_task_run.assert_awaited_once()

    async def test_execute_sends_task_as_request_from_creator(self):
        # The firing is answered as a request authored by the task creator, so
        # GeneralQueryGenerator loads the creator's memory and the guild's context.
        task = _make_task()

        await self.engine._execute(task, intended_run_at=task.next_run_at, scheduled=True)

        request = self.mock_gqg.handle_request.await_args.args[1]
        self.assertIsNone(request.message_id)
        self.assertEqual(request.author_id, task.creator_user_id)
        self.assertEqual(request.guild_id, task.guild_id)
        self.assertEqual(request.content, task.prompt)
        self.assertIs(self.mock_router.extract_general_params.await_args.args[0], request)

    # ---- reply chain ----

    async def test_execute_without_chain_posts_plain_message_from_latest_context(self):
        task = _make_task(chain_message_ids=[])

        await self.engine._execute(task, intended_run_at=task.next_run_at, scheduled=True)

        _params, request, fetch_conversation, _bot_user = self.mock_gqg.handle_request.await_args.args
        await fetch_conversation()
        self.engine._channel_conversation_fetcher.assert_awaited_once_with(task.channel_id, None)
        self.mock_channel.send.assert_awaited_once_with("response text", suppress_embeds=True)
        self.assertIsNone(request.reply_to_id)

    async def test_execute_replies_to_last_chain_message(self):
        task = _make_task(task_id=3, chain_message_ids=[10, 20, 30])

        await self.engine._execute(task, intended_run_at=task.next_run_at, scheduled=True)

        _params, request, fetch_conversation, _bot_user = self.mock_gqg.handle_request.await_args.args
        await fetch_conversation()
        self.engine._channel_conversation_fetcher.assert_awaited_once_with(task.channel_id, 30)
        reference = self.mock_channel.send.await_args.kwargs["reference"]
        self.assertEqual(reference.message_id, 30)
        self.assertEqual(reference.channel_id, task.channel_id)
        self.assertFalse(reference.fail_if_not_exists)
        self.assertEqual(request.reply_to_id, 30)

    async def test_execute_records_sent_message_in_chain(self):
        task = _make_task(task_id=3, chain_message_ids=[10])

        await self.engine._execute(task, intended_run_at=task.next_run_at, scheduled=True)

        task_id, _last_run_at, message_id, max_chain_ids = self.mock_store.record_task_run.await_args.args
        self.assertEqual(task_id, 3)
        self.assertEqual(message_id, 555)
        self.assertEqual(max_chain_ids, 100)

    async def test_execute_passes_task_history_to_generator(self):
        self.mock_summarizer.summarize.return_value = "- covered: 猫 (cat)"
        task = _make_task(chain_message_ids=[10, 20, 30, 40])

        await self.engine._execute(task, intended_run_at=task.next_run_at, scheduled=True)

        self.mock_summarizer.summarize.assert_awaited_once_with(task, 999)
        self.assertEqual(self.mock_gqg.handle_request.await_args.kwargs["task_history"], "- covered: 猫 (cat)")

    async def test_fire_task_also_extends_chain(self):
        task = _make_task(task_id=5, chain_message_ids=[10])

        await self.engine.fire_task(task)

        self.mock_store.record_task_run.assert_awaited_once()
        self.assertEqual(self.mock_channel.send.await_args.kwargs["reference"].message_id, 10)

    # ---- _dispatch_due_task error handling ----

    async def test_dispatch_store_failure_skips_execution(self):
        self.mock_store.update_task_next_run_at.side_effect = Exception("DB down")
        task = _make_task(task_id=5, cron="0 9 * * *")
        self.engine._execute = AsyncMock()

        await self.engine._dispatch_due_task(task, datetime.now(timezone.utc))

        self.engine._execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
