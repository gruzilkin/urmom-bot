from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone

import nextcord
from croniter import croniter
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ai_router import AiRouter
from conversation_graph import ConversationMessage
from general_query_generator import GeneralQueryGenerator
from open_telemetry import Telemetry
from schemas import GeneralParams
from store import ScheduledTask, Store

logger = logging.getLogger(__name__)

CATCHUP_STALENESS = timedelta(hours=12)
TICK_INTERVAL_SECONDS = 60
ACTIVITY_WINDOW = timedelta(hours=24)


class ScheduleEngine:
    def __init__(
        self,
        store: Store,
        telemetry: Telemetry,
        general_query_generator: GeneralQueryGenerator,
    ) -> None:
        self.store = store
        self.telemetry = telemetry
        self.general_query_generator = general_query_generator
        # Set by the container after AiRouter construction; constructor injection
        # would be circular (engine → router → schedule_handler → engine).
        self.ai_router: AiRouter | None = None
        self.bot: nextcord.Client | None = None
        self._channel_conversation_fetcher: Callable[[int], Awaitable[list[ConversationMessage]]] | None = None
        self._loop_task: asyncio.Task | None = None

    async def start(
        self,
        bot: nextcord.Client,
        channel_conversation_fetcher: Callable[[int], Awaitable[list[ConversationMessage]]],
    ) -> None:
        """Start the per-minute tick loop. Idempotent."""
        if self._loop_task and not self._loop_task.done():
            logger.info("ScheduleEngine already running")
            return
        self.bot = bot
        self._channel_conversation_fetcher = channel_conversation_fetcher
        self._loop_task = asyncio.create_task(self._tick_loop())
        logger.info("ScheduleEngine started")

    async def _tick_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(TICK_INTERVAL_SECONDS)
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unexpected error in schedule engine tick loop")

    async def _tick(self) -> None:
        now = datetime.now(timezone.utc)
        tasks = await self.store.list_scheduled_tasks(due_before=now, limit=20)
        if not tasks:
            return

        logger.info(f"ScheduleEngine tick: {len(tasks)} due task(s)")
        for task in tasks:
            await self._dispatch_due_task(task, now)

    async def _dispatch_due_task(self, task: ScheduledTask, now: datetime) -> None:
        """For a single due task: advance next_run_at (recurring) or delete row (one-off)
        BEFORE dispatching execution."""
        intended = task.next_run_at

        try:
            if task.cron_expression:
                new_next_run_at = self._compute_next_after_fire(task, now)
                await self.store.update_task_next_run_at(task.task_id, new_next_run_at)
            else:
                await self.store.delete_scheduled_task(task.task_id)
        except Exception:
            logger.exception(f"Failed to advance task {task.task_id}; skipping firing")
            return

        if task.cron_expression and (now - intended) > CATCHUP_STALENESS:
            logger.info(
                f"Skipping stale firing for task {task.task_id}: "
                f"intended {intended.isoformat()}, drift {now - intended}"
            )
            return

        await self._execute(task, intended_run_at=intended, scheduled=True)

    def _compute_next_after_fire(self, task: ScheduledTask, now: datetime) -> datetime:
        """Compute next_run_at after a firing of a recurring task (croniter from now → UTC instant)."""
        try:
            tz = ZoneInfo(task.timezone)
        except ZoneInfoNotFoundError:
            logger.warning(f"Task {task.task_id} has invalid timezone '{task.timezone}'; falling back to UTC")
            tz = timezone.utc
        now_in_tz = now.astimezone(tz)
        itr = croniter(task.cron_expression, now_in_tz)
        nxt = itr.get_next(datetime)
        return nxt.astimezone(timezone.utc)

    async def fire_task(self, task: ScheduledTask) -> None:
        """Public entry used by run_now: execute the task once outside the schedule."""
        await self._execute(task, intended_run_at=datetime.now(timezone.utc), scheduled=False)

    async def _execute(
        self,
        task: ScheduledTask,
        intended_run_at: datetime,
        scheduled: bool,
    ) -> None:
        actual_run_at = datetime.now(timezone.utc)
        async with self.telemetry.async_create_span("schedule_fire_task") as span:
            span.set_attribute("task_id", task.task_id)
            span.set_attribute("guild_id", task.guild_id)
            span.set_attribute("channel_id", task.channel_id)
            span.set_attribute("intended_run_at", intended_run_at.isoformat())
            span.set_attribute("actual_run_at", actual_run_at.isoformat())
            span.set_attribute("drift_seconds", (actual_run_at - intended_run_at).total_seconds())
            span.set_attribute("scheduled", scheduled)

            if self.bot is None or self._channel_conversation_fetcher is None or self.ai_router is None:
                span.set_attribute("status", "engine_not_started")
                logger.error(f"Task {task.task_id}: engine not started or not wired; cannot fire")
                return

            channel = self.bot.get_channel(task.channel_id)
            if channel is None:
                try:
                    channel = await self.bot.fetch_channel(task.channel_id)
                except Exception as e:
                    span.set_attribute("status", "channel_unavailable")
                    logger.error(
                        f"Task {task.task_id}: channel {task.channel_id} unavailable: {e}",
                        exc_info=True,
                    )
                    return

            try:
                params: GeneralParams = await self.ai_router.extract_general_params(task.prompt)

                span.set_attribute("ai_backend", params.ai_backend)
                span.set_attribute("temperature", params.temperature)

                async def fetch_conversation() -> list[ConversationMessage]:
                    return await self._channel_conversation_fetcher(task.channel_id)

                active_user_ids = await self.store.list_active_user_ids(task.guild_id, actual_run_at - ACTIVITY_WINDOW)
                extra_user_ids = set(active_user_ids)
                span.set_attribute("extra_user_count", len(extra_user_ids))

                response = await self.general_query_generator.handle_request(
                    params,
                    fetch_conversation,
                    task.guild_id,
                    self.bot.user,
                    task.creator_user_id,
                    extra_user_ids=extra_user_ids,
                )

                if response is None:
                    span.set_attribute("status", "no_response")
                    logger.warning(f"Task {task.task_id}: generator returned None")
                    return

                await channel.send(response)
                await self.store.mark_task_last_run(task.task_id, actual_run_at)
                span.set_attribute("status", "success")
            except Exception as e:
                span.set_attribute("status", "error")
                logger.error(f"Task {task.task_id}: firing failed: {e}", exc_info=True)
