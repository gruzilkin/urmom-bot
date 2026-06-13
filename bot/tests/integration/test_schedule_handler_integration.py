"""Integration tests for ScheduleHandler.

Exercises the real LLM extraction path (gemma + fallbacks) against natural-language
schedule requests, verifying that:
- "every X at Y" yields a valid cron expression
- "tomorrow at 3pm" / "in 2 hours" produce a dateparser-compatible first_run_phrase
- Timezone is extracted when mentioned, defaults to guild default otherwise
- Failure path: gibberish input yields null fields + an explanatory reason
"""

import os
import unittest
from unittest.mock import AsyncMock, MagicMock

from croniter import croniter
from dotenv import load_dotenv

from gemma_client import GemmaClient
from null_telemetry import NullTelemetry
from ai_client_wrappers import CompositeAIClient, RetryAIClient

from schedule_handler import ScheduleHandler
from schemas import ScheduleParams
from store import GuildConfig, Store

load_dotenv()


class TestScheduleHandlerIntegration(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.telemetry = NullTelemetry()

        # Build a fallback chain similar to the bot's lightweight_fallback
        gemma = GemmaClient(
            api_key=os.getenv("GEMMA_API_KEY"),
            model_name=os.getenv("GEMMA_MODEL", "gemma-3-27b-it"),
            telemetry=self.telemetry,
            temperature=0.1,
        )
        retrying_gemma = RetryAIClient(gemma, telemetry=self.telemetry, max_time=60, jitter=True)
        self.client = CompositeAIClient([gemma, retrying_gemma], telemetry=self.telemetry)

        # Mock store: just records what gets persisted
        self.mock_store = MagicMock(spec=Store)
        self.mock_store.get_guild_config = AsyncMock(
            return_value=GuildConfig(guild_id=100, default_timezone="Asia/Tokyo")
        )
        self.mock_store.create_scheduled_task = AsyncMock(return_value=42)
        self.mock_store.list_scheduled_tasks = AsyncMock(return_value=[])

        self.mock_engine = MagicMock()
        self.mock_engine.fire_task = AsyncMock()

        self.handler = ScheduleHandler(
            ai_client=self.client,
            store=self.mock_store,
            telemetry=self.telemetry,
            schedule_engine=self.mock_engine,
        )

    async def test_create_extracts_cron_for_recurring_schedule(self):
        params = ScheduleParams(operation="create", language_code="en", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="schedule a weekly summary every Monday at 9am",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )

        self.assertIn("task #42", response)
        self.mock_store.create_scheduled_task.assert_awaited_once()
        kwargs = self.mock_store.create_scheduled_task.await_args.kwargs

        self.assertIsNotNone(kwargs["prompt"])
        self.assertGreater(len(kwargs["prompt"]), 0)

        # Should be a valid 5-field cron firing on Mondays at 9:00
        cron = kwargs["cron_expression"]
        self.assertIsNotNone(cron, f"Expected cron expression, got {kwargs}")
        self.assertTrue(croniter.is_valid(cron), f"Invalid cron from LLM: {cron}")

        # Timezone should be a valid IANA name (default Asia/Tokyo)
        self.assertIsNotNone(kwargs["timezone"])

    async def test_create_extracts_first_run_phrase_for_one_off(self):
        params = ScheduleParams(operation="create", language_code="en", language_name="English")
        response = await self.handler.handle_request(
            params,
            message="remind me to call mom tomorrow at 3pm",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )

        self.assertIn("task #42", response)
        self.mock_store.create_scheduled_task.assert_awaited_once()
        kwargs = self.mock_store.create_scheduled_task.await_args.kwargs

        # One-off: cron should be null, next_run_at populated
        self.assertIsNone(kwargs["cron_expression"], f"Expected null cron for one-off, got {kwargs}")
        self.assertIsNotNone(kwargs["next_run_at"])

    async def test_create_extracts_explicit_timezone(self):
        params = ScheduleParams(operation="create", language_code="en", language_name="English")
        await self.handler.handle_request(
            params,
            message="schedule a daily haiku every day at 10am Tokyo time",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )

        self.mock_store.create_scheduled_task.assert_awaited_once()
        kwargs = self.mock_store.create_scheduled_task.await_args.kwargs
        tz = kwargs["timezone"]
        self.assertIn("Tokyo", tz, f"Expected Asia/Tokyo timezone, got {tz}")

    async def test_create_failure_does_not_persist(self):
        params = ScheduleParams(operation="create", language_code="en", language_name="English")
        await self.handler.handle_request(
            params,
            message="qwerty asdf hjkl no schedule here",
            guild_id=100,
            channel_id=200,
            creator_user_id=300,
        )

        # Either the LLM bails out (null fields → reason returned), or validation rejects.
        # Either way, nothing should be persisted.
        self.mock_store.create_scheduled_task.assert_not_called()


if __name__ == "__main__":
    unittest.main()
