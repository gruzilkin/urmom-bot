from abc import ABC, abstractmethod
from typing import List, Tuple

class AIClient(ABC):
    @abstractmethod
    async def generate_content(self, message: str, prompt: str = None, samples: List[Tuple[str, str]] = None) -> str:
        pass

    @abstractmethod
    async def is_joke(self, original_message: str, response_message: str) -> bool:
        pass
    
    @abstractmethod
    async def generate_famous_person_response(self, conversation: List[Tuple[str, str]], person: str, original_message: str = "") -> str:
        """
        Generate a response in the style of a famous person based on the conversation context.
        
        Args:
            conversation (List[Tuple[str, str]]): List of (username, message) tuples
            person (str): The name of the famous person
            original_message (str): The original user request with bot mention removed
            
        Returns:
            str: A response in the style of the famous person
        """
        pass
        
    @abstractmethod
    async def is_famous_person_request(self, message: str) -> str | None:
        """
        Check if a message is asking what a famous person would say.
        
        Args:
            message (str): The message to check
            
        Returns:
            str | None: The name of the famous person if it's a request, None otherwise
        """
        pass