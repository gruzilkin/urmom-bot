"""
Shared Pydantic schemas for structured LLM outputs.

This module contains reusable schemas that can be used with different AI clients
for structured output generation.
"""

from typing import Literal
from pydantic import BaseModel, Field


class YesNo(BaseModel):
    """Schema for YES/NO responses."""

    answer: Literal["YES", "NO"] = Field(description="Return exactly YES or NO, uppercase")


class FamousParams(BaseModel):
    """Parameters for famous person impersonation requests."""

    famous_person: str = Field(
        description="Name of the famous person to impersonate (celebrity, fictional character, or historical figure)"
    )
    language_code: str | None = Field(
        default=None,
        description="ISO 639-1 language code (e.g., 'en', 'ru', 'de') - populated after parameter extraction",
    )
    language_name: str | None = Field(
        default=None,
        description=(
            "Full name of the detected language (e.g., 'English', 'Russian', 'German')"
            " - populated after parameter extraction"
        ),
    )


class GeneralParams(BaseModel):
    """Parameters for general AI query requests."""

    ai_backend: Literal["gemini_flash", "grok", "gemma", "codex", "deepseek"]
    temperature: float = Field(ge=0.0, le=1.0)
    cleaned_query: str = Field(description="User's request with 'BOT' mentions and routing instructions removed")
    language_code: str | None = Field(
        default=None,
        description="ISO 639-1 language code (e.g., 'en', 'ru', 'de') - populated after parameter extraction",
    )
    language_name: str | None = Field(
        default=None,
        description=(
            "Full name of the detected language (e.g., 'English', 'Russian', 'German')"
            " - populated after parameter extraction"
        ),
    )


class FactParams(BaseModel):
    """Parameters for memory fact operations."""

    operation: Literal["remember", "forget"] = Field(
        description="Memory operation type: 'remember' to store a fact, 'forget' to remove a fact"
    )
    member_id: int | None = Field(
        default=None,
        description="Numeric Discord member ID of the member the fact is about, or null if it cannot be determined",
    )
    fact_content: str = Field(description="The specific fact to remember or forget about the user")
    language_code: str | None = Field(
        default=None,
        description="ISO 639-1 language code (e.g., 'en', 'ru', 'de') - populated after parameter extraction",
    )
    language_name: str | None = Field(
        default=None,
        description=(
            "Full name of the detected language (e.g., 'English', 'Russian', 'German')"
            " - populated after parameter extraction"
        ),
    )


class ScheduleParams(BaseModel):
    """Parameters for scheduled task management requests.

    Router tier 2 extraction: identifies which sub-operation the user wants.
    Task identification and per-operation payload extraction happen later inside
    the schedule handler, where the channel's task list is available as context.
    """

    operation: Literal["create", "list", "edit", "delete", "run_now"] = Field(
        description=(
            "Schedule sub-operation: 'create' to add a new task, 'list' to query existing tasks,"
            " 'edit' to modify an existing task, 'delete' to remove a task,"
            " 'run_now' to fire a task immediately outside its schedule."
        )
    )
    language_code: str | None = Field(
        default=None,
        description="ISO 639-1 language code (e.g., 'en', 'ru', 'de') - populated after parameter extraction",
    )
    language_name: str | None = Field(
        default=None,
        description=(
            "Full name of the detected language (e.g., 'English', 'Russian', 'German')"
            " - populated after parameter extraction"
        ),
    )


class MemoryUpdate(BaseModel):
    """Schema for memory update operations."""

    updated_memory: str = Field(description="The updated memory blob after incorporating new information")
    confirmation_message: str = Field(description="Brief confirmation message for the user in their language")


class MemoryForget(BaseModel):
    """Schema for memory forget operations."""

    updated_memory: str = Field(description="The updated memory blob after removing information")
    fact_found: bool = Field(description="Whether the specified fact was found and removed")
    confirmation_message: str = Field(description="Brief confirmation message for the user in their language")


class ScheduleCreateParams(BaseModel):
    """LLM output for creating a new scheduled task.

    On a successful parse: prompt and timezone are populated, plus at least one of
    cron_expression / first_run_phrase. On failure (request unparseable, ambiguous,
    or contradictory): all data fields are null and reason explains why.
    """

    reason: str = Field(
        description="User-facing outcome message describing success or failure, in the user's language."
    )
    prompt: str | None = Field(
        default=None,
        description="Instruction the bot will execute when the task fires.",
    )
    cron_expression: str | None = Field(
        default=None,
        description=(
            "5-field cron expression for recurring schedules (e.g., '0 9 * * 1' for every Monday 9am)."
            " Null for one-off tasks."
        ),
    )
    first_run_phrase: str | None = Field(
        default=None,
        description=(
            "Natural-language time expression. MUST be written in English regardless of the"
            " user's input language (e.g., 'tomorrow at 3pm', 'in 2 hours', 'Monday 9am')."
            " Required for one-off tasks, or for an explicit first-run anchor on a recurring"
            " task; otherwise null."
        ),
    )
    timezone: str | None = Field(
        default=None,
        description="IANA timezone name (e.g., 'Asia/Tokyo', 'Asia/Dubai', 'America/New_York').",
    )


class ScheduleListParams(BaseModel):
    """LLM output for the list operation — freeform answer to the user's question
    given the channel's task list."""

    answer: str = Field(
        description=("Response to the user's request about the channel's scheduled tasks, in the user's language.")
    )


class ScheduleEditParams(BaseModel):
    """LLM output for editing an existing scheduled task.

    Resolves the target task and produces the updated task fields in a single call.
    On a successful parse: task_id resolved against the channel's task list, plus the
    full updated task fields (unchanged fields carried forward verbatim from the existing
    task). On failure (task not found, or change request unparseable): task_id and data
    fields are null and reason explains why.
    """

    reason: str = Field(
        description="User-facing outcome message describing success or failure, in the user's language."
    )
    task_id: int | None = Field(
        default=None,
        description=(
            "ID of the task referenced by the user, resolved against the channel's task list."
            " Null if no matching task was found."
        ),
    )
    prompt: str | None = Field(
        default=None,
        description="Instruction the bot will execute when the task fires.",
    )
    cron_expression: str | None = Field(
        default=None,
        description=(
            "5-field cron expression for recurring schedules (e.g., '0 9 * * 1' for every Monday 9am)."
            " Null for one-off tasks."
        ),
    )
    first_run_phrase: str | None = Field(
        default=None,
        description=(
            "Natural-language time expression. MUST be written in English regardless of the"
            " user's input language (e.g., 'tomorrow at 3pm', 'in 2 hours', 'Monday 9am')."
            " Required for one-off tasks, or for an explicit first-run anchor on a recurring"
            " task; otherwise null."
        ),
    )
    timezone: str | None = Field(
        default=None,
        description="IANA timezone name (e.g., 'Asia/Tokyo', 'Asia/Dubai', 'America/New_York').",
    )


class ScheduleTaskResolution(BaseModel):
    """LLM output for delete and run_now: resolves a task reference from the user's
    message against the channel's task list."""

    task_id: int | None = Field(
        default=None,
        description=(
            "ID of the task referenced by the user, resolved against the channel's task list."
            " Null if no matching task was found."
        ),
    )
    reason: str = Field(
        description="User-facing outcome message describing success or failure, in the user's language."
    )


class RouteSelection(BaseModel):
    """Schema for AI router route selection (first tier)."""

    route: Literal["FAMOUS", "GENERAL", "FACT", "SCHEDULE", "NONE", "NOTSURE"] = Field(description="Route decision")
    reason: str = Field(description="Brief reason for choosing this route")


class MemoryContext(BaseModel):
    """Schema for memory manager context generation."""

    context: str = Field(
        description="Merged context combining facts, current day observations, and historical patterns"
    )


class UserSummary(BaseModel):
    """Schema for individual user summary."""

    user_id: int = Field(description="Discord user ID")
    summary: str = Field(description="Daily summary for the user")


class DailySummaries(BaseModel):
    """Schema for batch daily summary generation."""

    summaries: list[UserSummary] = Field(description="List of daily summaries for all active users")


class WisdomResponse(BaseModel):
    """Schema for wisdom generation response."""

    answer: str = Field(description="The street-smart, humorous wisdom one-liner to deliver to the user")
    reason: str = Field(
        description="Explanation of the observation, style choice, and why this wisdom fits the conversation"
    )


class DevilsAdvocateResponse(BaseModel):
    """Schema for devil's advocate counter-argument response."""

    answer: str = Field(description="The analytical counter-argument to deliver to the user")
    reason: str = Field(
        description="Explanation of the logical strategy and reasoning approach used for this counter-argument"
    )


class UserAliases(BaseModel):
    """Schema for extracting known names/aliases from factual memory."""

    aliases: list[str] = Field(description="Known real names, nicknames, or alternative names for the user")
