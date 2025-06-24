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
    famous_person: str = Field(description="Name of the famous person to impersonate")


class GeneralParams(BaseModel):
    """Parameters for general AI query requests."""
    ai_backend: Literal["gemini_flash", "grok", "claude", "gemma"] = Field(description="AI backend to use")
    temperature: float = Field(description="Temperature 0.0-1.0", ge=0.0, le=1.0)
    cleaned_query: str = Field(description="The query with routing instructions removed")
