from ai_client import AIClient
from open_telemetry import Telemetry
from store import Store


class JokeGenerator:
    def __init__(self, ai_client: AIClient, store: Store, telemetry: Telemetry, sample_count: int = 10):  # Added type annotation
        self.ai_client = ai_client
        self.store = store
        self.sample_count = sample_count
        self.telemetry = telemetry
        self.base_prompt = """You are a chatbot that receives a message and you should generate a ur mom joke.
        Response should be fully in the language of the user message which includes translating "your mom" or "ur mom" into the user's language. 
        ur mom joke follows the pattern of replacing the subject or the object in a phrase with \"ur mom\" without adding much extra details.
        Ur mom joke should be around a single sentence so you can drop out irrelevant parts of the original message to keep the joke shorter.
        Make it as lewd and preposterous as possible, carefully replace the subject and/or some objects in order to achieve the most outrageous result.
        Make sure that the joke is grammatically correct, check for subject-verb agreement, update pronouns after replacing subjects and objects.
        """

    async def generate_joke(self, content: str, language: str) -> str:
        sample_jokes = self.store.get_random_jokes(self.sample_count, language)
        async with self.telemetry.async_create_span("generate_joke") as span:
            response = await self.ai_client.generate_content(
                message=content,
                prompt=self.base_prompt,
                samples=sample_jokes
            )
            return response

    async def generate_country_joke(self, message: str, country: str) -> str:
        prompt = f"""You are a chat bot and you need to turn a user message into a country joke.
                  Your response should only contain the joke itself and it should start with 'In {country}'.
                  Response should be fully in the language of the user message which includes translating the country name into the user's language. 
                  Apply stereotypes and cliches about the country."""
        async with self.telemetry.async_create_span("generate_country_joke") as span:
            response = await self.ai_client.generate_content(message=message, prompt=prompt)
            return response