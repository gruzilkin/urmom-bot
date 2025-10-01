import unittest
import os
from unittest.mock import Mock, AsyncMock
from dotenv import load_dotenv
from joke_generator import JokeGenerator
from gemini_client import GeminiClient
from gemma_client import GemmaClient
from grok_client import GrokClient
from store import Store
from language_detector import LanguageDetector
from null_telemetry import NullTelemetry

load_dotenv()

class TestJokeGenerator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Check if paid tests are enabled
        enable_paid_tests = os.getenv('ENABLE_PAID_TESTS', '').lower() == 'true'

        # Set up main AI client based on ENABLE_PAID_TESTS flag
        if enable_paid_tests:
            # Use Grok for paid tests
            grok_api_key = os.getenv('GROK_API_KEY')
            grok_model = os.getenv('GROK_MODEL')

            if not grok_api_key:
                self.skipTest("GROK_API_KEY environment variable not set")
            if not grok_model:
                self.skipTest("GROK_MODEL environment variable not set")

            ai_client = GrokClient(
                api_key=grok_api_key,
                model_name=grok_model,
                temperature=0.1,  # Fixed temperature for test stability
                telemetry=NullTelemetry()
            )
        else:
            # Use Gemini for free tests (default)
            gemini_api_key = os.getenv('GEMINI_API_KEY')
            flash_model = os.getenv('GEMINI_FLASH_MODEL')

            if not gemini_api_key:
                self.skipTest("GEMINI_API_KEY environment variable not set")
            if not flash_model:
                self.skipTest("GEMINI_FLASH_MODEL environment variable not set")

            ai_client = GeminiClient(
                api_key=gemini_api_key,
                model_name=flash_model,
                temperature=0.1,  # Fixed temperature for test stability
                telemetry=NullTelemetry()
            )

        # Language detector always uses Gemma
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        gemma_model = os.getenv('GEMINI_GEMMA_MODEL')

        if not gemini_api_key:
            self.skipTest("GEMINI_API_KEY environment variable not set (needed for language detection)")
        if not gemma_model:
            self.skipTest("GEMINI_GEMMA_MODEL environment variable not set")

        gemma_client = GemmaClient(
            api_key=gemini_api_key,
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

    async def test_is_joke_realistic_pairs(self):
        """Test is_joke with realistic Discord message pairs to validate prompt effectiveness"""
        test_cases = [
            # (original, response, expected_result, description)
            # YES cases - Clear jokes
            ("I haven't had that weird rash again", "TWSS", True, "TWSS joke"),
            ("I'm tired today", "That's what ur mom said last night", True, "ur mom joke"),
            ("This code is so confusing", "Just like your face", True, "insult humor"),
            ("How do I exit vim?", "You don't. Vim exits you.", True, "vim joke"),
            # NO cases - Not jokes (these were common false positives)
            ("Not enough to convince florent", "No shit...", False, "casual agreement"),
            ("This is terrible", "Tell me about it", False, "commiseration"),
            ("I'm going to the store", "Have fun, don't get lost", False, "friendly banter"),
            ("I can't figure this out", "Have you tried reading the docs?", False, "helpful suggestion"),
        ]

        for original, response, expected, description in test_cases:
            with self.subTest(description=description, original=original, response=response):
                result = await self.joke_generator.is_joke(original, response)
                print(f"  [{description}] Original: '{original}' -> Response: '{response}' -> Result: {result} (expected: {expected})")
                self.assertEqual(result, expected, f"Failed for {description}: expected {expected}, got {result}")

if __name__ == '__main__':
    unittest.main()