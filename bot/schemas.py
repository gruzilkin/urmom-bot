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
