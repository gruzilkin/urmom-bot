from abc import ABC, abstractmethod
from typing import List, Tuple

class AIClient(ABC):
    @abstractmethod
    async def generate_content(self, message: str, prompt: str = None, samples: List[Tuple[str, str]] = None) -> str:
        pass

    @abstractmethod
    async def is_joke(self, original_message: str, response_message: str) -> bool:
        pass