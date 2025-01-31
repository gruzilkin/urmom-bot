import re

from ai_client import AIClient
from gemini_client import GeminiClient
from store import Store

class JokeGenerator:
    def __init__(self, ai_client: AIClient, store: Store, sample_count: int = 10):
        self.ai_client = ai_client
        self.store = store
        self.sample_count = sample_count
        self.base_prompt = """You are a chatbot that receives a message and you should generate a ur mom joke.
        ur mom joke follows the pattern of replacing the subject or the object in a phrase with \"ur mom\" without adding much extra details but you can leave out irrelevant parts.
        Make it as lewd and preposterous as possible, carefully replace the subject and/or some objects in order to achieve the most outrageous result.
        Make sure that the joke is grammatically correct, check for subject-verb agreement, update pronouns after replacing subjects and objects.
        """

    async def generate_joke(self, content: str) -> str:
        sample_jokes = self.store.get_random_jokes(self.sample_count)
        response = await self.ai_client.generate_content(
            message=content,
            prompt=self.base_prompt,
            samples=sample_jokes
        )
        return response

    async def generate_country_joke(self, message: str, country: str) -> str:
        prompt = f"You are a chat bot and you need to turn a user message into a country joke. Your response should only contain the joke itself and it should start with 'In {country}'. Apply stereotypes and cliches about the country."
        response = await self.ai_client.generate_content(message=message, prompt=prompt)
        return response
