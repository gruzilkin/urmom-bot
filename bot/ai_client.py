from abc import ABC, abstractmethod
from typing import List, Tuple, Type, TypeVar, Union, Optional
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class AIClient(ABC):
    @abstractmethod
    async def generate_content(self, message: str, prompt: str = None, samples: List[Tuple[str, str]] = None, **kwargs) -> Union[str, T]:
        pass