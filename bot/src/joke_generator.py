import logging
from ai_client import AIClient
from open_telemetry import Telemetry
from store import Store
from language_detector import LanguageDetector
from schemas import YesNo
from typing import Dict
from opentelemetry.trace import SpanKind


logger = logging.getLogger(__name__)


class JokeGenerator:
    def __init__(
        self,
        joke_writer_client: AIClient,
        joke_classifier_client: AIClient,
        store: Store,
        telemetry: Telemetry,
        language_detector: LanguageDetector,
        sample_count: int = 10,
    ):
        self._joke_writer_client = joke_writer_client
        self._joke_classifier_client = joke_classifier_client
        self.store = store
        self.sample_count = sample_count
        self.telemetry = telemetry
        self.language_detector = language_detector
        self._joke_cache: Dict[int, bool] = {}  # message_id -> bool cache

    async def generate_joke(self, content: str, language: str) -> str:
        # Convert language code to language name
        language_name = await self.language_detector.get_language_name(language)
        
        sample_jokes = await self.store.get_random_jokes(self.sample_count)
        
        # Format sample jokes as XML examples
        examples_xml = ""
        if sample_jokes:
            examples_xml = "<examples>"
            for message, joke in sample_jokes:
                examples_xml += f"<example><message>{message}</message><joke>{joke}</joke></example>"
            examples_xml += "</examples>"
        
        # Create the prompt using format string
        russian_note = " In Russian, use the slang form 'твоя мамка'." if language == "ru" else ""
        prompt = f"""You are a chatbot that receives a message and you should generate a ur mom joke.        
        Extract the most interesting or funny concept from the message and create a punchy one-liner around it. Don't echo back entire sentences - distill the message into its most essential, impactful element for maximum comedic effect.
        ur mom joke follows the pattern of replacing the subject or the object with "ur mom".{russian_note}
        Make it as lewd and preposterous as possible, carefully replace the subject and/or some objects in order to achieve the most outrageous result.
        Make sure that the joke is grammatically correct, check for subject-verb agreement, update pronouns after replacing subjects and objects.
        
        Reply in {language_name}. Return only the joke, no meta commentary or explanation.
        
        {examples_xml}"""
        
        async with self.telemetry.async_create_span("generate_joke"):
            response = await self._joke_writer_client.generate_content(
                message=content,
                prompt=prompt
            )
            return response

    async def generate_country_joke(self, message: str, country: str) -> str:
        prompt = f"""You are a chat bot and you need to turn a user message into a country joke.
                  Your response should only contain the joke itself and it should start with 'In {country}'.
                  Response should be fully in the language of the user message which includes translating the country name into the user's language. 
                  Apply stereotypes and cliches about the country."""
        async with self.telemetry.async_create_span("generate_country_joke"):
            response = await self._joke_writer_client.generate_content(
                message=message, prompt=prompt
            )
            return response

    async def is_joke(self, original_message: str, response_message: str, message_id: int = None) -> bool:
        """
        Determine if a response message is a joke to the original message.
        Includes caching to avoid redundant AI calls.
        """
        # Check cache first if message_id is provided
        if message_id is not None and message_id in self._joke_cache:
            return self._joke_cache[message_id]

        async with self.telemetry.async_create_span("is_joke", kind=SpanKind.INTERNAL):
            # Format messages in XML for clarity
            message = f"""<messages>
<original>{original_message}</original>
<response>{response_message}</response>
</messages>"""

            prompt = """Determine if the response is clearly intended as a joke or humorous remark directed at the original message.

Only answer YES if the response is obviously humorous, a clear joke, or deliberate wordplay. Regular conversation, even if slightly playful or witty, should be NO."""

            logger.info("Checking if message is a joke:")
            logger.info(f"Original: {original_message}")
            logger.info(f"Response: {response_message}")

            response = await self._joke_classifier_client.generate_content(
                message=message,
                prompt=prompt,
                response_schema=YesNo
            )

            result = response.answer == "YES"

            # Cache the result if message_id is provided
            if message_id is not None:
                self._joke_cache[message_id] = result

            logger.info(f"AI response: {response.answer}")
            logger.info(f"Is joke: {result}")
            return result

    async def save_joke(self, source_message_id: int, source_message_content: str, 
                                    joke_message_id: int, joke_message_content: str, 
                                    reaction_count: int) -> None:
        """
        Save a joke to the store with language detection.
        """
        async with self.telemetry.async_create_span("save_joke"):
            # Detect languages using the language detection service
            source_lang = await self.language_detector.detect_language(source_message_content)
            joke_lang = await self.language_detector.detect_language(joke_message_content)
            
            await self.store.save(
                source_message_id=source_message_id,
                joke_message_id=joke_message_id,
                source_message_content=source_message_content,
                joke_message_content=joke_message_content,
                reaction_count=reaction_count,
                source_language=source_lang,
                joke_language=joke_lang
            )
            
            logger.info(f"Saved joke: {joke_message_content} (reactions: {reaction_count})")
