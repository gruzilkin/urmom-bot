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


class FactParams(BaseModel):
    """Parameters for memory fact operations."""
    operation: Literal["remember", "forget"] = Field(description="Type of memory operation")
    user_mention: str = Field(description="User being referenced (e.g., '@username' or 'gruzilkin')")
    fact_content: str = Field(description="The fact to remember/forget")


class MemoryUpdate(BaseModel):
    """Schema for memory update operations."""
    updated_memory: str = Field(description="The updated memory blob after incorporating new information")


class MemoryForget(BaseModel):
    """Schema for memory forget operations."""
    updated_memory: str = Field(description="The updated memory blob after removing information")
    fact_found: bool = Field(description="Whether the specified fact was found and removed")
