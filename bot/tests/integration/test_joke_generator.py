import os
import unittest
from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock

from dotenv import load_dotenv

from gemini_client import GeminiClient
from gemma_client import GemmaClient
from grok_client import GrokClient
from joke_generator import JokeGenerator
from language_detector import LanguageDetector
from null_telemetry import NullTelemetry
from ollama_client import OllamaClient
from store import Store

load_dotenv()


@dataclass(frozen=True)
class JokeClientProfile:
    """Container for joke generator client configuration."""

    name: str
    client: object


class TestJokeGenerator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.telemetry = NullTelemetry()
        self.profiles: list[JokeClientProfile] = []

        enable_paid_tests = os.getenv("ENABLE_PAID_TESTS", "").lower() == "true"

        gemini_api_key = os.getenv("GEMINI_API_KEY")
        gemma_model = os.getenv("GEMINI_GEMMA_MODEL")
        if not gemini_api_key:
            self.skipTest(
                "GEMINI_API_KEY environment variable not set (needed for language detection)"
            )
        if not gemma_model:
            self.skipTest(
                "GEMINI_GEMMA_MODEL environment variable not set (needed for language detection)"
            )

        self.gemma_client = GemmaClient(
            api_key=gemini_api_key,
            model_name=gemma_model,
            telemetry=self.telemetry,
        )
        self.language_detector = LanguageDetector(
            ai_client=self.gemma_client,
            telemetry=self.telemetry,
        )

        if enable_paid_tests:
            grok_api_key = os.getenv("GROK_API_KEY")
            grok_model = os.getenv("GROK_MODEL")
            if not grok_api_key:
                self.skipTest("GROK_API_KEY environment variable not set")
            if not grok_model:
                self.skipTest("GROK_MODEL environment variable not set")

            grok_client = GrokClient(
                api_key=grok_api_key,
                model_name=grok_model,
                temperature=0.1,
                telemetry=self.telemetry,
            )
            self.profiles.append(JokeClientProfile(name="grok", client=grok_client))
        else:
            flash_model = os.getenv("GEMINI_FLASH_MODEL")
            if not flash_model:
                self.skipTest("GEMINI_FLASH_MODEL environment variable not set")

            gemini_client = GeminiClient(
                api_key=gemini_api_key,
                model_name=flash_model,
                temperature=0.1,
                telemetry=self.telemetry,
            )
            self.profiles.append(
                JokeClientProfile(name="gemini_flash", client=gemini_client)
            )

        ollama_api_key = os.getenv("OLLAMA_API_KEY")
        if ollama_api_key:
            kimi_model = os.getenv("OLLAMA_KIMI_MODEL", "kimi-k2:1t-cloud")
            kimi_client = OllamaClient(
                api_key=ollama_api_key,
                model_name=kimi_model,
                telemetry=self.telemetry,
                temperature=0.1,
            )
            self.profiles.append(
                JokeClientProfile(name="ollama_kimi", client=kimi_client)
            )

        if not self.profiles:
            self.skipTest("No joke generator AI clients configured for integration tests")

        self.joke_seed_data = [
            ("The switch is hard to use", "ur mom is hard to use"),
            ("I need more space in my room", "I need more space in ur mom"),
            ("This game is too expensive", "ur mom is too expensive"),
            ("The graphics are amazing", "ur mom's graphics are amazing"),
            ("I can't handle this level", "I can't handle ur mom"),
        ]

    def _build_joke_generator(self, ai_client) -> JokeGenerator:
        store = Mock(spec=Store)
        store.get_random_jokes = AsyncMock(return_value=list(self.joke_seed_data))

        return JokeGenerator(
            joke_writer_client=ai_client,
            joke_classifier_client=ai_client,
            store=store,
            telemetry=self.telemetry,
            language_detector=self.language_detector,
        )

    async def test_generate_joke(self):
        test_message = "tldr, it's basically switch 1 with some extra coloured plastic"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                joke_generator = self._build_joke_generator(profile.client)
                result = await joke_generator.generate_joke(test_message, "en")
                self.assertIsInstance(result, str)
                self.assertGreater(len(result), 0)
                print(f"[{profile.name}] Generated joke: {result}")

    async def test_generate_country_joke(self):
        test_message = "I like to eat"
        country = "France"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                joke_generator = self._build_joke_generator(profile.client)
                result = await joke_generator.generate_country_joke(
                    test_message, country
                )
                self.assertIsInstance(result, str)
                self.assertGreater(len(result), 0)
                print(f"[{profile.name}] Generated country joke: {result}")

    async def test_is_joke_realistic_pairs(self):
        """Test is_joke with realistic Discord message pairs to validate prompt effectiveness"""
        test_cases = [
            ("I haven't had that weird rash again", "TWSS", True, "TWSS joke"),
            (
                "I'm tired today",
                "That's what ur mom said last night",
                True,
                "ur mom joke",
            ),
            ("This code is so confusing", "Just like your face", True, "insult humor"),
            ("How do I exit vim?", "You don't. Vim exits you.", True, "vim joke"),
            ("Not enough to convince florent", "No shit...", False, "casual agreement"),
            ("This is terrible", "Tell me about it", False, "commiseration"),
            ("I'm going to the store", "Have fun, don't get lost", False, "banter"),
            (
                "I can't figure this out",
                "Have you tried reading the docs?",
                False,
                "helpful suggestion",
            ),
        ]

        for profile in self.profiles:
            joke_generator = self._build_joke_generator(profile.client)
            for original, response, expected, description in test_cases:
                with self.subTest(
                    profile=profile.name,
                    description=description,
                    original=original,
                    response=response,
                ):
                    result = await joke_generator.is_joke(original, response)
                    print(
                        f"[{profile.name}] [{description}] Original: '{original}' -> "
                        f"Response: '{response}' -> Result: {result} "
                        f"(expected: {expected})"
                    )
                    self.assertEqual(
                        result,
                        expected,
                        f"Failed for {profile.name} / {description}: expected {expected}, got {result}",
                    )


if __name__ == "__main__":
    unittest.main()
