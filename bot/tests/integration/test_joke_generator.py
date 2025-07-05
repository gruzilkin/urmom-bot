import unittest
import os
from unittest.mock import Mock, AsyncMock
from dotenv import load_dotenv
from joke_generator import JokeGenerator
from gemini_client import GeminiClient
from gemma_client import GemmaClient
from store import Store
from language_detector import LanguageDetector
from tests.null_telemetry import NullTelemetry

load_dotenv()

class TestJokeGenerator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Check for API key and model names
        api_key = os.getenv('GEMINI_API_KEY')
        flash_model = os.getenv('GEMINI_FLASH_MODEL')
        gemma_model = os.getenv('GEMINI_GEMMA_MODEL')
        
        if not api_key:
            self.skipTest("GEMINI_API_KEY environment variable not set")
        if not flash_model:
            self.skipTest("GEMINI_FLASH_MODEL environment variable not set")
        if not gemma_model:
            self.skipTest("GEMINI_GEMMA_MODEL environment variable not set")
            
        ai_client = GeminiClient(
            api_key=api_key,
            model_name=flash_model,
            temperature=0.1,  # Fixed temperature for test stability
            telemetry=NullTelemetry()
        )
        
        # Language detector should always use Gemma
        gemma_client = GemmaClient(
            api_key=api_key,
            model_name=gemma_model,
            telemetry=NullTelemetry()
        )
        
        self.store = Mock(spec=Store)
        self.store.get_random_jokes = AsyncMock(return_value=[
            ("The switch is hard to use", "ur mom is hard to use"),
            ("I need more space in my room", "I need more space in ur mom"),
            ("This game is too expensive", "ur mom is too expensive"),
            ("The graphics are amazing", "ur mom's graphics are amazing"),
            ("I can't handle this level", "I can't handle ur mom")
        ])
        
        # Instantiate LanguageDetector with Gemma
        language_detector = LanguageDetector(
            ai_client=gemma_client,  # Always use Gemma for language detection
            telemetry=NullTelemetry()
        )

        self.joke_generator = JokeGenerator(
            ai_client=ai_client,
            store=self.store,
            telemetry=NullTelemetry(),
            language_detector=language_detector
        )

    async def test_generate_joke(self):
        # Test input
        test_message = "tldr, it's basically switch 1 with some extra coloured plastic"
        
        # Generate joke
        result = await self.joke_generator.generate_joke(test_message, "en")
        print(f"Generated joke: {result}")

    async def test_generate_country_joke(self):
        # Test input
        test_message = "I like to eat"
        country = "France"
        
        # Generate joke
        result = await self.joke_generator.generate_country_joke(test_message, country)
        print(f"Generated country joke: {result}")

if __name__ == '__main__':
    unittest.main()