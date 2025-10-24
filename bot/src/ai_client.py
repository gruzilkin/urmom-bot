from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel


class BlockedException(RuntimeError):
    """Raised when an AI provider refuses to generate content."""

    def __init__(self, *, reason: str):
        super().__init__(reason)
        self.reason = reason

T = TypeVar("T", bound=BaseModel)

class AIClient(ABC):
    @abstractmethod
    async def generate_content(
        self,
        message: str,
        prompt: str = None,
        samples: list[tuple[str, str]] | None = None,
        enable_grounding: bool = False,
        response_schema: type[T] | None = None,
        temperature: float | None = None,
        image_data: bytes | None = None,
        image_mime_type: str | None = None,
    ) -> str | T:
        pass
