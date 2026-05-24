from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone as dt_timezone
from typing import TYPE_CHECKING

import dateparser
from croniter import croniter, CroniterBadCronError
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ai_client import AIClient
from open_telemetry import Telemetry
from schemas import (
    ScheduleCreateParams,
    ScheduleEditParams,
    ScheduleListParams,
    ScheduleParams,
    ScheduleTaskResolution,
)
from store import ScheduledTask, Store

if TYPE_CHECKING:
    from schedule_engine import ScheduleEngine

logger = logging.getLogger(__name__)


class ScheduleHandler:
    def __init__(
        self,
        ai_client: AIClient,
        store: Store,
        telemetry: Telemetry,
        schedule_engine: ScheduleEngine,
    ) -> None:
        self.ai_client = ai_client
        self.store = store
        self.telemetry = telemetry
        self.schedule_engine = schedule_engine

    def get_route_description(self) -> str:
        return """
        SCHEDULE: For managing scheduled tasks in this channel
        - For commands that create, list, edit, delete, or fire scheduled tasks
        - Examples:
          * "schedule a weekly summary every Monday 9am"
          * "remind me to call mom tomorrow at 3pm"
          * "what's scheduled in this channel?"
          * "change task 5's prompt to summarize the past two weeks"
          * "delete task 5"
          * "cancel the weekly summary"
          * "run task 5 now"

        Non-examples (NOT SCHEDULE - route elsewhere):
        - "what time is it?" (GENERAL)
        - "what's on my calendar?" (GENERAL; not about bot-managed tasks)
        - "remember that John likes pizza" (FACT)
        """

    def get_parameter_schema(self):
        return ScheduleParams

    def get_parameter_extraction_prompt(self, conversation_context: str = "") -> str:
        context_section = ""
        if conversation_context:
            context_section = f"""
<conversation_context>
Extract parameters from the LAST message.
Use earlier messages to resolve references like "this", "that", "it" in the last message.

{conversation_context}
</conversation_context>
"""

        return f"""
        Extract which schedule sub-operation the user wants.

        operation:
        - 'create' — user wants to add a new scheduled task
          (e.g., "schedule X", "remind me to Y", "every Monday do Z")
        - 'list' — user wants to inspect scheduled tasks
          (e.g., "what's scheduled?", "show my tasks", "list scheduled jobs")
        - 'edit' — user wants to modify an existing task
          (e.g., "change task 5's prompt", "make the weekly summary run on Fridays")
        - 'delete' — user wants to remove a task
          (e.g., "delete task 5", "cancel the weekly summary", "stop the morning poem")
        - 'run_now' — user wants to fire a task immediately
          (e.g., "run task 5 now", "fire the daily standup now")

        Identify the verb from the user's message only. Do not resolve which task they mean.
        {context_section}"""

    async def handle_request(
        self,
        params: ScheduleParams,
        message: str,
        guild_id: int,
        channel_id: int,
        creator_user_id: int,
    ) -> str:
        logger.info(f"Processing schedule request: {params.operation}")
        language_name = params.language_name or "English"

        async with self.telemetry.async_create_span("schedule_handle_request") as span:
            span.set_attribute("operation", params.operation)
            span.set_attribute("guild_id", guild_id)
            span.set_attribute("channel_id", channel_id)

            if params.operation == "create":
                return await self._create(message, guild_id, channel_id, creator_user_id, language_name)
            if params.operation == "list":
                return await self._list(message, guild_id, channel_id, language_name)
            if params.operation == "edit":
                return await self._edit(message, guild_id, channel_id, language_name)
            if params.operation == "delete":
                return await self._delete(message, guild_id, channel_id, language_name)
            if params.operation == "run_now":
                return await self._run_now(message, guild_id, channel_id, language_name)
            return f"Unknown schedule operation: {params.operation}"

    async def _create(
        self,
        message: str,
        guild_id: int,
        channel_id: int,
        creator_user_id: int,
        language_name: str,
    ) -> str:
        config = await self.store.get_guild_config(guild_id)
        default_tz_name = config.default_timezone

        try:
            default_tz = ZoneInfo(default_tz_name)
        except ZoneInfoNotFoundError:
            default_tz = ZoneInfo("UTC")
            default_tz_name = "UTC"

        now_in_tz = datetime.now(default_tz)
        prompt = self._build_create_prompt(default_tz_name, now_in_tz, language_name)

        result: ScheduleCreateParams = await self.ai_client.generate_content(
            message=message,
            prompt=prompt,
            temperature=0.0,
            response_schema=ScheduleCreateParams,
        )

        if not result.prompt or not result.timezone:
            return result.reason
        if not result.cron_expression and not result.first_run_phrase:
            return result.reason

        try:
            task_tz = ZoneInfo(result.timezone)
        except ZoneInfoNotFoundError:
            return f"Couldn't resolve timezone '{result.timezone}'. Use an IANA name like Asia/Tokyo."

        try:
            next_run_at = self._compute_next_run_at(
                cron=result.cron_expression,
                first_run_phrase=result.first_run_phrase,
                tz=task_tz,
            )
        except ValueError as e:
            return f"Couldn't parse the schedule: {e}"

        task_id = await self.store.create_scheduled_task(
            guild_id=guild_id,
            channel_id=channel_id,
            creator_user_id=creator_user_id,
            prompt=result.prompt,
            cron_expression=result.cron_expression,
            timezone=result.timezone,
            next_run_at=next_run_at,
        )

        return f"{result.reason} (task #{task_id})"

    async def _list(
        self,
        message: str,
        guild_id: int,
        channel_id: int,
        language_name: str,
    ) -> str:
        tasks = await self.store.list_scheduled_tasks(guild_id, channel_id)
        if not tasks:
            return "No scheduled tasks in this channel."

        prompt = self._build_list_prompt(tasks, language_name)
        result: ScheduleListParams = await self.ai_client.generate_content(
            message=message,
            prompt=prompt,
            temperature=0.0,
            response_schema=ScheduleListParams,
        )
        return result.answer

    async def _edit(
        self,
        message: str,
        guild_id: int,
        channel_id: int,
        language_name: str,
    ) -> str:
        tasks = await self.store.list_scheduled_tasks(guild_id, channel_id)
        if not tasks:
            return "No scheduled tasks in this channel to edit."

        config = await self.store.get_guild_config(guild_id)
        try:
            default_tz = ZoneInfo(config.default_timezone)
        except ZoneInfoNotFoundError:
            default_tz = ZoneInfo("UTC")
        now_in_tz = datetime.now(default_tz)

        prompt = self._build_edit_prompt(tasks, now_in_tz, language_name)
        result: ScheduleEditParams = await self.ai_client.generate_content(
            message=message,
            prompt=prompt,
            temperature=0.0,
            response_schema=ScheduleEditParams,
        )

        if result.task_id is None:
            return result.reason
        if not result.prompt or not result.timezone:
            return result.reason

        existing = next((t for t in tasks if t.task_id == result.task_id), None)
        if existing is None:
            return f"Task {result.task_id} doesn't belong to this channel."

        try:
            task_tz = ZoneInfo(result.timezone)
        except ZoneInfoNotFoundError:
            return f"Couldn't resolve timezone '{result.timezone}'."

        # Schedule is unchanged if the LLM copied cron + timezone verbatim from the XML
        # and provided no first_run_phrase. In that case keep existing.next_run_at —
        # recomputing would clobber a one-off's stored firing time, and for a recurring
        # task whose first firing was set via first_run_phrase it would silently skip
        # past that intended first run to the next cron occurrence.
        schedule_unchanged = (
            result.cron_expression == existing.cron_expression
            and result.timezone == existing.timezone
            and not result.first_run_phrase
        )

        if schedule_unchanged:
            new_cron = existing.cron_expression
            next_run_at = existing.next_run_at
        else:
            new_cron = result.cron_expression
            try:
                next_run_at = self._compute_next_run_at(
                    cron=result.cron_expression,
                    first_run_phrase=result.first_run_phrase,
                    tz=task_tz,
                )
            except ValueError as e:
                return f"Couldn't parse the new schedule: {e}"

        await self.store.update_scheduled_task(
            task_id=result.task_id,
            prompt=result.prompt,
            cron_expression=new_cron,
            timezone=result.timezone,
            next_run_at=next_run_at,
        )
        return result.reason

    async def _delete(
        self,
        message: str,
        guild_id: int,
        channel_id: int,
        language_name: str,
    ) -> str:
        tasks = await self.store.list_scheduled_tasks(guild_id, channel_id)
        if not tasks:
            return "No scheduled tasks in this channel to delete."

        prompt = self._build_resolution_prompt(tasks, "delete", language_name)
        result: ScheduleTaskResolution = await self.ai_client.generate_content(
            message=message,
            prompt=prompt,
            temperature=0.0,
            response_schema=ScheduleTaskResolution,
        )

        if result.task_id is None:
            return result.reason

        task = next((t for t in tasks if t.task_id == result.task_id), None)
        if task is None:
            return f"Task {result.task_id} doesn't belong to this channel."

        await self.store.delete_scheduled_task(result.task_id)
        return result.reason

    async def _run_now(
        self,
        message: str,
        guild_id: int,
        channel_id: int,
        language_name: str,
    ) -> str:
        tasks = await self.store.list_scheduled_tasks(guild_id, channel_id)
        if not tasks:
            return "No scheduled tasks in this channel to run."

        prompt = self._build_resolution_prompt(tasks, "run immediately", language_name)
        result: ScheduleTaskResolution = await self.ai_client.generate_content(
            message=message,
            prompt=prompt,
            temperature=0.0,
            response_schema=ScheduleTaskResolution,
        )

        if result.task_id is None:
            return result.reason

        task = next((t for t in tasks if t.task_id == result.task_id), None)
        if task is None:
            return f"Task {result.task_id} doesn't belong to this channel."

        asyncio.create_task(self.schedule_engine.fire_task(task))
        return result.reason

    def _compute_next_run_at(
        self,
        cron: str | None,
        first_run_phrase: str | None,
        tz: ZoneInfo,
    ) -> datetime:
        """Compute the next firing as a UTC instant.

        Three legal combinations:
        - cron set, phrase null   → croniter(cron, now_tz).get_next()
        - cron set, phrase set    → dateparser(phrase) (first run override)
        - cron null, phrase set   → dateparser(phrase) (one-off)
        """
        now_tz = datetime.now(tz)

        if cron:
            try:
                itr = croniter(cron, now_tz)
            except (CroniterBadCronError, ValueError) as e:
                raise ValueError(f"invalid cron expression '{cron}': {e}") from e
        else:
            itr = None

        if first_run_phrase:
            parsed = dateparser.parse(
                first_run_phrase,
                settings={
                    "TIMEZONE": str(tz),
                    "RETURN_AS_TIMEZONE_AWARE": True,
                    "RELATIVE_BASE": now_tz.replace(tzinfo=None),
                    "PREFER_DATES_FROM": "future",
                },
            )
            if parsed is None:
                raise ValueError(f"couldn't parse time phrase '{first_run_phrase}'")
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=tz)
            if parsed <= now_tz:
                raise ValueError(f"parsed time {parsed.isoformat()} is in the past")
            return parsed.astimezone(dt_timezone.utc)

        if itr is not None:
            nxt = itr.get_next(datetime)
            return nxt.astimezone(dt_timezone.utc)

        raise ValueError("both cron_expression and first_run_phrase are null")

    def _render_tasks_xml(self, tasks: list[ScheduledTask]) -> str:
        """Render tasks as XML for inclusion in LLM prompts.

        Field elements match the names the LLM is expected to produce when editing
        (task_id, prompt, cron_expression, timezone), so the LLM can copy unchanged
        fields verbatim. next_run_at is read-only context.
        """
        if not tasks:
            return "<scheduled_tasks/>"

        parts = ["<scheduled_tasks>"]
        for t in tasks:
            try:
                tz = ZoneInfo(t.timezone)
                next_run = t.next_run_at.astimezone(tz).strftime("%Y-%m-%d %H:%M %Z")
            except ZoneInfoNotFoundError:
                next_run = t.next_run_at.isoformat()
            cron_el = (
                f"<cron_expression>{t.cron_expression}</cron_expression>" if t.cron_expression else "<cron_expression/>"
            )
            parts.append(
                f"  <task>\n"
                f"    <task_id>{t.task_id}</task_id>\n"
                f"    <prompt>{t.prompt}</prompt>\n"
                f"    {cron_el}\n"
                f"    <timezone>{t.timezone}</timezone>\n"
                f"    <next_run_at>{next_run}</next_run_at>\n"
                f"  </task>"
            )
        parts.append("</scheduled_tasks>")
        return "\n".join(parts)

    def _build_create_prompt(self, default_tz_name: str, now_in_tz: datetime, language_name: str) -> str:
        return f"""
Extract schedule details for a new scheduled task from the user's request.

Current datetime: {now_in_tz.strftime("%Y-%m-%d %H:%M %Z")}
Guild default timezone (IANA): {default_tz_name}

Fields:
- prompt: the instruction the bot will execute when fired. Strip scheduling words.
- cron_expression: 5-field cron for recurring schedules (e.g., "0 9 * * 1"). Null for one-off.
- first_run_phrase: time expression for one-off tasks, or an explicit first-run anchor on a
  recurring task. Null when the recurring schedule's first firing is the next cron occurrence.
  Always write first_run_phrase field in English regardless of input language.

  Use these forms:
    - "in 2 hours", "in 30 minutes", "in 5 days", "in 1 week"
    - "tomorrow", "tomorrow at 3pm", "tomorrow 9am"
    - "Monday 9am", "Monday at 9am", "Friday 5pm", "Sunday at noon"
    - "3pm", "9am", "noon", "midnight", "15:30"
    - "June 15 at 9am", "Dec 1 2pm", "2026-06-15 09:00"
    - "next week", "next month"

  Never produce these forms:
    - "next Monday", "next Friday", "next Monday 9am" — use bare "Monday", "Monday 9am"
    - "tomorrow morning", "tomorrow evening", "tonight", "this evening", "end of week" —
      use explicit clock times like "tomorrow at 8am", "tomorrow at 7pm"
    - bare ordinals like "the 1st at 9am" — use "June 1 at 9am"
- timezone: IANA name. If the user did not specify one, use {default_tz_name}.
- reason: confirmation in {language_name} echoing the resolved schedule.

If the request cannot be parsed, set prompt / cron_expression / first_run_phrase / timezone
to null and explain in reason.

Examples:
- "schedule a weekly summary every Monday 9am"
  → prompt: "post a weekly summary of the channel",
    cron_expression: "0 9 * * 1", first_run_phrase: null, timezone: "{default_tz_name}",
    reason: "Scheduled a weekly summary every Monday at 9:00 ({default_tz_name})."
- "remind me to call mom tomorrow at 3pm"
  → prompt: "remind us to call mom",
    cron_expression: null, first_run_phrase: "tomorrow at 3pm",
    timezone: "{default_tz_name}",
    reason: "I'll fire once tomorrow at 3pm ({default_tz_name})."
- "every day at 10am Tokyo starting tomorrow, post haiku"
  → prompt: "post a haiku",
    cron_expression: "0 10 * * *", first_run_phrase: "tomorrow at 10am",
    timezone: "Asia/Tokyo",
    reason: "Scheduled a daily haiku at 10:00 Asia/Tokyo, first firing tomorrow."
- "напомни мне позвонить маме завтра в 3 дня"
  → prompt: "напомни нам позвонить маме",
    cron_expression: null, first_run_phrase: "tomorrow at 3pm",
    timezone: "{default_tz_name}",
    reason: "Напомню один раз завтра в 15:00 ({default_tz_name})."

Respond in {language_name}.
"""

    def _build_list_prompt(self, tasks: list[ScheduledTask], language_name: str) -> str:
        return f"""
Answer the user's question about scheduled tasks in their channel.

{self._render_tasks_xml(tasks)}

Reference tasks by their integer task_id. If the question can't be answered from this data,
say so. Respond in {language_name}.
"""

    def _build_edit_prompt(self, tasks: list[ScheduledTask], now_in_tz: datetime, language_name: str) -> str:
        return f"""
Edit an existing scheduled task based on the user's change request.

Current datetime: {now_in_tz.strftime("%Y-%m-%d %H:%M %Z")}

{self._render_tasks_xml(tasks)}

Steps:
1. Identify which task the user wants to edit; set task_id to that integer.
2. Return the full updated task fields. Copy unchanged fields verbatim from the matching
   <task> element. Modify only the fields the user asked to change. An empty
   <cron_expression/> means the task is one-off and cron_expression must remain null.
3. cron_expression / first_run_phrase / timezone follow the same rules as for creating a task.

If the task or change cannot be identified, set task_id and all data fields to null and
explain in reason.

reason: confirmation of the change or explanation of failure, in {language_name}.
"""

    def _build_resolution_prompt(self, tasks: list[ScheduledTask], action: str, language_name: str) -> str:
        return f"""
Identify which task the user wants to {action}.

{self._render_tasks_xml(tasks)}

Set task_id to the integer ID of the matching <task> entry. If no match, set task_id to null
and explain in reason.

reason: confirmation of the action or explanation if not found, in {language_name}.
"""
