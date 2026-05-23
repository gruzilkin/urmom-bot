import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

from null_telemetry import NullTelemetry

from ai_client import AIClient
from schedule_handler import ScheduleHandler
from schemas import (
    ScheduleCreateParams,
    ScheduleEditParams,
    ScheduleListParams,
    ScheduleParams,
    ScheduleTaskResolution,
)
from store import GuildConfig, ScheduledTask, Store


def _make_task(
    task_id: int = 1,
    prompt: str = "do thing",
    cron: str | None = "0 9 * * 1",
    tz: str = "Asia/Tokyo",
    next_run_at: datetime | None = None,
) -> ScheduledTask:
    now = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc)
    return ScheduledTask(
        task_id=task_id,
        guild_id=100,
        channel_id=200,
        creator_user_id=300,
        prompt=prompt,
        cron_expression=cron,
        timezone=tz,
        next_run_at=next_run_at or (now + timedelta(days=1)),
        last_run_at=None,
        created_at=now,
        updated_at=now,
    )


class TestScheduleHandler(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.telemetry = NullTelemetry()
        self.mock_ai = MagicMock(spec=AIClient)
        self.mock_store = MagicMock(spec=Store)
        self.mock_engine = MagicMock()
        self.mock_engine.fire_task = AsyncMock()

        self.mock_store.get_guild_config = AsyncMock(
            return_value=GuildConfig(guild_id=100, default_timezone="Asia/Tokyo")
        )
        self.mock_store.create_scheduled_task = AsyncMock(return_value=42)
        self.mock_store.update_scheduled_task = AsyncMock(return_value=True)
        self.mock_store.delete_scheduled_task = AsyncMock(return_value=True)
        self.mock_store.list_scheduled_tasks = AsyncMock(return_value=[])

        self.handler = ScheduleHandler(
            ai_client=self.mock_ai,
            store=self.mock_store,
            telemetry=self.telemetry,
            schedule_engine=self.mock_engine,
        )

    # ---- _compute_next_run_at ----

    def test_compute_next_run_at_pure_recurring(self):
        tz = ZoneInfo("Asia/Tokyo")
        nxt = self.handler._compute_next_run_at(cron="0 9 * * 1", first_run_phrase=None, tz=tz)
        self.assertEqual(nxt.tzinfo, timezone.utc)
        # Should be a Monday 9am Berlin in the future
        self.assertGreater(nxt, datetime.now(timezone.utc))

    def test_compute_next_run_at_first_run_override(self):
        tz = ZoneInfo("Asia/Tokyo")
        nxt = self.handler._compute_next_run_at(cron="0 9 * * 1", first_run_phrase="in 5 minutes", tz=tz)
        self.assertEqual(nxt.tzinfo, timezone.utc)
        delta = nxt - datetime.now(timezone.utc)
        self.assertGreater(delta.total_seconds(), 0)
        self.assertLess(delta.total_seconds(), 10 * 60)

    def test_compute_next_run_at_one_off(self):
        tz = ZoneInfo("Asia/Tokyo")
        nxt = self.handler._compute_next_run_at(cron=None, first_run_phrase="in 30 minutes", tz=tz)
        self.assertEqual(nxt.tzinfo, timezone.utc)

    def test_compute_next_run_at_both_null_raises(self):
        tz = ZoneInfo("UTC")
        with self.assertRaises(ValueError):
            self.handler._compute_next_run_at(cron=None, first_run_phrase=None, tz=tz)

    def test_compute_next_run_at_bad_cron_raises(self):
        tz = ZoneInfo("UTC")
        with self.assertRaises(ValueError):
            self.handler._compute_next_run_at(cron="not a cron", first_run_phrase=None, tz=tz)

    def test_compute_next_run_at_past_phrase_raises(self):
        tz = ZoneInfo("UTC")
        with self.assertRaises(ValueError):
            self.handler._compute_next_run_at(cron=None, first_run_phrase="yesterday", tz=tz)

    # ---- _create ----

    async def test_create_success(self):
        self.mock_ai.generate_content = AsyncMock(
            return_value=ScheduleCreateParams(
                reason="Scheduled a weekly summary every Monday at 9am Asia/Tokyo.",
                prompt="post a weekly summary",
                cron_expression="0 9 * * 1",
                first_run_phrase=None,
                timezone="Asia/Tokyo",
            )
        )
        params = ScheduleParams(operation="create", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="schedule a weekly summary every Monday 9am",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )
        self.assertIn("task #42", response)
        self.mock_store.create_scheduled_task.assert_awaited_once()
        kwargs = self.mock_store.create_scheduled_task.await_args.kwargs
        self.assertEqual(kwargs["prompt"], "post a weekly summary")
        self.assertEqual(kwargs["cron_expression"], "0 9 * * 1")
        self.assertEqual(kwargs["timezone"], "Asia/Tokyo")

    async def test_create_llm_failure_returns_reason_without_persisting(self):
        self.mock_ai.generate_content = AsyncMock(
            return_value=ScheduleCreateParams(reason="I couldn't understand your schedule.")
        )
        params = ScheduleParams(operation="create", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="gibberish",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )
        self.assertEqual(response, "I couldn't understand your schedule.")
        self.mock_store.create_scheduled_task.assert_not_awaited()

    async def test_create_invalid_timezone_rejected(self):
        self.mock_ai.generate_content = AsyncMock(
            return_value=ScheduleCreateParams(
                reason="OK",
                prompt="do thing",
                cron_expression="0 9 * * *",
                first_run_phrase=None,
                timezone="Made/Up_Zone",
            )
        )
        params = ScheduleParams(operation="create", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="schedule",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )
        self.assertIn("timezone", response.lower())
        self.mock_store.create_scheduled_task.assert_not_awaited()

    async def test_create_invalid_cron_rejected(self):
        self.mock_ai.generate_content = AsyncMock(
            return_value=ScheduleCreateParams(
                reason="OK",
                prompt="do thing",
                cron_expression="not a cron",
                first_run_phrase=None,
                timezone="UTC",
            )
        )
        params = ScheduleParams(operation="create", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="schedule",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )
        self.assertIn("schedule", response.lower())
        self.mock_store.create_scheduled_task.assert_not_awaited()

    # ---- _list ----

    async def test_list_empty(self):
        self.mock_store.list_scheduled_tasks.return_value = []
        params = ScheduleParams(operation="list", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="what's scheduled?",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )
        self.assertIn("No scheduled tasks", response)
        # Should NOT call AI when empty
        self.mock_ai.generate_content.assert_not_called()

    async def test_list_returns_answer(self):
        self.mock_store.list_scheduled_tasks.return_value = [_make_task()]
        self.mock_ai.generate_content = AsyncMock(
            return_value=ScheduleListParams(answer="You have 1 task: weekly summary on Mondays.")
        )
        params = ScheduleParams(operation="list", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="what's scheduled?",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )
        self.assertEqual(response, "You have 1 task: weekly summary on Mondays.")

    # ---- _delete ----

    async def test_delete_resolved(self):
        task = _make_task(task_id=5)
        self.mock_store.list_scheduled_tasks.return_value = [task]
        self.mock_ai.generate_content = AsyncMock(
            return_value=ScheduleTaskResolution(task_id=5, reason="Deleted task 5.")
        )
        params = ScheduleParams(operation="delete", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="delete task 5",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )
        self.assertEqual(response, "Deleted task 5.")
        self.mock_store.delete_scheduled_task.assert_awaited_once_with(5)

    async def test_delete_unresolved(self):
        self.mock_store.list_scheduled_tasks.return_value = [_make_task(task_id=5)]
        self.mock_ai.generate_content = AsyncMock(
            return_value=ScheduleTaskResolution(reason="I couldn't find that task.")
        )
        params = ScheduleParams(operation="delete", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="delete the foo",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )
        self.assertEqual(response, "I couldn't find that task.")
        self.mock_store.delete_scheduled_task.assert_not_awaited()

    async def test_delete_task_belongs_to_different_channel_rejected(self):
        # LLM hallucinates a task_id that's not in the channel list
        self.mock_store.list_scheduled_tasks.return_value = [_make_task(task_id=5)]
        self.mock_ai.generate_content = AsyncMock(
            return_value=ScheduleTaskResolution(task_id=999, reason="Deleting...")
        )
        params = ScheduleParams(operation="delete", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="delete task 999",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )
        self.assertIn("doesn't belong", response)
        self.mock_store.delete_scheduled_task.assert_not_awaited()

    # ---- _run_now ----

    async def test_run_now_dispatches_to_engine(self):
        task = _make_task(task_id=5)
        self.mock_store.list_scheduled_tasks.return_value = [task]
        self.mock_ai.generate_content = AsyncMock(
            return_value=ScheduleTaskResolution(task_id=5, reason="Running task 5 now.")
        )
        params = ScheduleParams(operation="run_now", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="run task 5 now",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )
        self.assertEqual(response, "Running task 5 now.")
        # asyncio.create_task wraps fire_task; allow the scheduled task to run
        import asyncio

        await asyncio.sleep(0)
        self.mock_engine.fire_task.assert_awaited_once_with(task)

    # ---- _edit ----

    async def test_edit_success(self):
        existing = _make_task(task_id=5, prompt="old prompt", cron="0 9 * * 1")
        self.mock_store.list_scheduled_tasks.return_value = [existing]
        self.mock_ai.generate_content = AsyncMock(
            return_value=ScheduleEditParams(
                reason="Updated task 5's prompt.",
                task_id=5,
                prompt="new prompt",
                cron_expression="0 9 * * 1",
                first_run_phrase=None,
                timezone="Asia/Tokyo",
            )
        )
        params = ScheduleParams(operation="edit", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="change task 5's prompt to new prompt",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )
        self.assertEqual(response, "Updated task 5's prompt.")
        self.mock_store.update_scheduled_task.assert_awaited_once()
        kwargs = self.mock_store.update_scheduled_task.await_args.kwargs
        self.assertEqual(kwargs["task_id"], 5)
        self.assertEqual(kwargs["prompt"], "new prompt")

    async def test_edit_unresolved_task(self):
        self.mock_store.list_scheduled_tasks.return_value = [_make_task(task_id=5)]
        self.mock_ai.generate_content = AsyncMock(return_value=ScheduleEditParams(reason="Couldn't find the task."))
        params = ScheduleParams(operation="edit", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="edit the foo",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )
        self.assertEqual(response, "Couldn't find the task.")
        self.mock_store.update_scheduled_task.assert_not_awaited()

    # ---- _render_tasks_xml ----

    def test_render_tasks_xml_recurring(self):
        rendered = self.handler._render_tasks_xml([_make_task(task_id=1)])
        self.assertIn("<scheduled_tasks>", rendered)
        self.assertIn("<task_id>1</task_id>", rendered)
        self.assertIn("<cron_expression>0 9 * * 1</cron_expression>", rendered)
        self.assertIn("<timezone>Asia/Tokyo</timezone>", rendered)
        self.assertIn("<next_run_at>", rendered)

    def test_render_tasks_xml_one_off_emits_empty_cron_element(self):
        task = _make_task(task_id=1, cron=None, tz="UTC")
        rendered = self.handler._render_tasks_xml([task])
        # One-off tasks have a self-closing cron_expression element
        self.assertIn("<cron_expression/>", rendered)
        self.assertNotIn("<cron_expression>", rendered)

    def test_render_tasks_xml_empty_list(self):
        rendered = self.handler._render_tasks_xml([])
        self.assertEqual(rendered, "<scheduled_tasks/>")

    def test_render_tasks_xml_invalid_timezone_falls_back_to_isoformat(self):
        task = _make_task(task_id=1, tz="Made/Up_Zone")
        rendered = self.handler._render_tasks_xml([task])
        self.assertIn("<task_id>1</task_id>", rendered)
        # Falls back to ISO datetime when tz can't be resolved
        self.assertIn("<next_run_at>", rendered)
        # ISO datetime contains 'T'
        next_run_segment = rendered.split("<next_run_at>")[1].split("</next_run_at>")[0]
        self.assertIn("T", next_run_segment)

    # ---- discovery methods (used by AI router) ----

    def test_discovery_methods(self):
        from schemas import ScheduleParams as SP

        self.assertIs(self.handler.get_parameter_schema(), SP)
        self.assertIn("SCHEDULE", self.handler.get_route_description())
        no_context = self.handler.get_parameter_extraction_prompt()
        with_context = self.handler.get_parameter_extraction_prompt("<conversation>recent stuff</conversation>")
        self.assertIn("operation", no_context)
        self.assertIn("recent stuff", with_context)

    # ---- _compute_next_run_at edge case ----

    def test_compute_next_run_at_unparseable_phrase_raises(self):
        from zoneinfo import ZoneInfo as Z

        with self.assertRaises(ValueError):
            self.handler._compute_next_run_at(cron=None, first_run_phrase="asdkfjasdkfj qwerty", tz=Z("UTC"))

    # ---- _create edge cases ----

    async def test_create_with_only_prompt_returns_reason_without_persisting(self):
        # LLM returns prompt + tz but NO cron and NO phrase — invalid combination.
        self.mock_ai.generate_content = AsyncMock(
            return_value=ScheduleCreateParams(
                reason="Need a schedule.",
                prompt="do thing",
                cron_expression=None,
                first_run_phrase=None,
                timezone="UTC",
            )
        )
        params = ScheduleParams(operation="create", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="ambiguous request",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )
        self.assertEqual(response, "Need a schedule.")
        self.mock_store.create_scheduled_task.assert_not_awaited()

    async def test_create_invalid_guild_default_timezone_falls_back_to_utc(self):
        # Guild config has an invalid timezone string
        self.mock_store.get_guild_config.return_value = GuildConfig(guild_id=100, default_timezone="Made/Up")
        self.mock_ai.generate_content = AsyncMock(
            return_value=ScheduleCreateParams(
                reason="Scheduled.",
                prompt="do thing",
                cron_expression="0 9 * * *",
                first_run_phrase=None,
                timezone="UTC",
            )
        )
        params = ScheduleParams(operation="create", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="schedule",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )
        self.assertIn("task #42", response)

    # ---- _edit edge cases ----

    async def test_edit_empty_list(self):
        self.mock_store.list_scheduled_tasks.return_value = []
        params = ScheduleParams(operation="edit", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="edit task 5",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )
        self.assertIn("No scheduled tasks", response)
        self.mock_ai.generate_content.assert_not_called()

    async def test_edit_llm_returns_null_data_fields(self):
        existing = _make_task(task_id=5)
        self.mock_store.list_scheduled_tasks.return_value = [existing]
        self.mock_ai.generate_content = AsyncMock(return_value=ScheduleEditParams(reason="Edit unclear.", task_id=5))
        params = ScheduleParams(operation="edit", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="edit something",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )
        self.assertEqual(response, "Edit unclear.")
        self.mock_store.update_scheduled_task.assert_not_awaited()

    async def test_edit_resolved_to_wrong_channel_rejected(self):
        # LLM hallucinates a task_id that doesn't exist in this channel
        self.mock_store.list_scheduled_tasks.return_value = [_make_task(task_id=5)]
        self.mock_ai.generate_content = AsyncMock(
            return_value=ScheduleEditParams(
                reason="OK",
                task_id=999,
                prompt="new",
                cron_expression="0 9 * * *",
                first_run_phrase=None,
                timezone="UTC",
            )
        )
        params = ScheduleParams(operation="edit", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="edit task 999",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )
        self.assertIn("doesn't belong", response)
        self.mock_store.update_scheduled_task.assert_not_awaited()

    async def test_edit_invalid_timezone_rejected(self):
        self.mock_store.list_scheduled_tasks.return_value = [_make_task(task_id=5)]
        self.mock_ai.generate_content = AsyncMock(
            return_value=ScheduleEditParams(
                reason="OK",
                task_id=5,
                prompt="new prompt",
                cron_expression="0 9 * * *",
                first_run_phrase=None,
                timezone="Made/Up_Zone",
            )
        )
        params = ScheduleParams(operation="edit", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="edit task 5",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )
        self.assertIn("timezone", response.lower())
        self.mock_store.update_scheduled_task.assert_not_awaited()

    async def test_edit_invalid_cron_rejected(self):
        self.mock_store.list_scheduled_tasks.return_value = [_make_task(task_id=5)]
        self.mock_ai.generate_content = AsyncMock(
            return_value=ScheduleEditParams(
                reason="OK",
                task_id=5,
                prompt="new prompt",
                cron_expression="not a cron",
                first_run_phrase=None,
                timezone="UTC",
            )
        )
        params = ScheduleParams(operation="edit", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="edit task 5",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )
        self.assertIn("schedule", response.lower())
        self.mock_store.update_scheduled_task.assert_not_awaited()

    async def test_edit_invalid_guild_default_timezone_falls_back_to_utc(self):
        self.mock_store.get_guild_config.return_value = GuildConfig(guild_id=100, default_timezone="Made/Up")
        self.mock_store.list_scheduled_tasks.return_value = [_make_task(task_id=5)]
        self.mock_ai.generate_content = AsyncMock(
            return_value=ScheduleEditParams(
                reason="Updated.",
                task_id=5,
                prompt="new prompt",
                cron_expression="0 9 * * *",
                first_run_phrase=None,
                timezone="UTC",
            )
        )
        params = ScheduleParams(operation="edit", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="edit task 5",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )
        self.assertEqual(response, "Updated.")

    # ---- _delete / _run_now edge cases ----

    async def test_delete_empty_list(self):
        self.mock_store.list_scheduled_tasks.return_value = []
        params = ScheduleParams(operation="delete", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="delete task 5",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )
        self.assertIn("No scheduled tasks", response)
        self.mock_ai.generate_content.assert_not_called()

    async def test_run_now_empty_list(self):
        self.mock_store.list_scheduled_tasks.return_value = []
        params = ScheduleParams(operation="run_now", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="run task 5 now",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )
        self.assertIn("No scheduled tasks", response)
        self.mock_ai.generate_content.assert_not_called()

    async def test_run_now_unresolved(self):
        self.mock_store.list_scheduled_tasks.return_value = [_make_task(task_id=5)]
        self.mock_ai.generate_content = AsyncMock(return_value=ScheduleTaskResolution(reason="Not found."))
        params = ScheduleParams(operation="run_now", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="run the foo now",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )
        self.assertEqual(response, "Not found.")
        self.mock_engine.fire_task.assert_not_called()

    async def test_run_now_resolved_to_wrong_channel_rejected(self):
        self.mock_store.list_scheduled_tasks.return_value = [_make_task(task_id=5)]
        self.mock_ai.generate_content = AsyncMock(return_value=ScheduleTaskResolution(task_id=999, reason="OK"))
        params = ScheduleParams(operation="run_now", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="run task 999 now",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )
        self.assertIn("doesn't belong", response)
        self.mock_engine.fire_task.assert_not_called()


if __name__ == "__main__":
    unittest.main()
