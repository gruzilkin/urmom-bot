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
        description="Full name of the detected language (e.g., 'English', 'Russian', 'German') - populated after parameter extraction",
    )


class GeneralParams(BaseModel):
    """Parameters for general AI query requests."""

    ai_backend: Literal["gemini_flash", "grok", "claude", "gemma", "codex"] = Field(
        description="AI backend to use: gemini_flash for general questions, grok for creative tasks, claude for technical work, codex for research, gemma only if explicitly requested"
    )
    temperature: float = Field(
        description="Response creativity level: 0.1-0.3 for factual/precise, 0.4-0.6 for balanced, 0.7-0.9 for creative",
        ge=0.0,
        le=1.0,
    )
    cleaned_query: str = Field(description="User's request with 'BOT' mentions and routing instructions removed")
    language_code: str | None = Field(
        default=None,
        description="ISO 639-1 language code (e.g., 'en', 'ru', 'de') - populated after parameter extraction",
    )
    language_name: str | None = Field(
        default=None,
        description="Full name of the detected language (e.g., 'English', 'Russian', 'German') - populated after parameter extraction",
    )


class FactParams(BaseModel):
    """Parameters for memory fact operations."""

    operation: Literal["remember", "forget"] = Field(
        description="Memory operation type: 'remember' to store a fact, 'forget' to remove a fact"
    )
    user_mention: str = Field(
        description="User reference: Discord mention like '<@123456>' or nickname like 'gruzilkin'"
    )
    fact_content: str = Field(description="The specific fact to remember or forget about the user")
    language_code: str | None = Field(
        default=None,
        description="ISO 639-1 language code (e.g., 'en', 'ru', 'de') - populated after parameter extraction",
    )
    language_name: str | None = Field(
        default=None,
        description="Full name of the detected language (e.g., 'English', 'Russian', 'German') - populated after parameter extraction",
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


class RouteSelection(BaseModel):
    """Schema for AI router route selection (first tier)."""

    route: Literal["FAMOUS", "GENERAL", "FACT", "NONE", "NOTSURE"] = Field(description="Route decision")
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
