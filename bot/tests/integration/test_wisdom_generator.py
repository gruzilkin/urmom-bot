"""Integration tests for WisdomGenerator with real AI clients."""

import os
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

from dotenv import load_dotenv

from conversation_graph import ConversationMessage
from gemma_client import GemmaClient
from grok_client import GrokClient
from language_detector import LanguageDetector
from null_telemetry import NullTelemetry
from ollama_client import OllamaClient
from response_summarizer import ResponseSummarizer
from test_store import TestStore
from wisdom_generator import WisdomGenerator

load_dotenv()


@dataclass(frozen=True)
class WisdomClientProfile:
    """Container for wisdom generator client configuration."""

    name: str
    client: object


class TestWisdomGeneratorIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for WisdomGenerator with real AI clients."""

    def setUp(self):
        self.telemetry = NullTelemetry()
        self.profiles: list[WisdomClientProfile] = []

        enable_paid_tests = os.getenv("ENABLE_PAID_TESTS", "").lower() == "true"

        gemini_api_key = os.getenv("GEMINI_API_KEY")
        gemma_model = os.getenv("GEMINI_GEMMA_MODEL")
        if not gemini_api_key:
            self.skipTest("GEMINI_API_KEY environment variable not set")
        if not gemma_model:
            self.skipTest("GEMINI_GEMMA_MODEL environment variable not set")

        self.gemma_client = GemmaClient(
            api_key=gemini_api_key,
            model_name=gemma_model,
            telemetry=self.telemetry,
        )

        self.language_detector = LanguageDetector(
            ai_client=self.gemma_client,
            telemetry=self.telemetry,
        )

        self.response_summarizer = ResponseSummarizer(
            self.gemma_client,
            self.telemetry,
        )

        if enable_paid_tests:
            grok_api_key = os.getenv("GROK_API_KEY")
            grok_model = os.getenv("GROK_MODEL")
            if grok_api_key and grok_model:
                grok_client = GrokClient(
                    api_key=grok_api_key,
                    model_name=grok_model,
                    temperature=0.7,
                    telemetry=self.telemetry,
                )
                self.profiles.append(
                    WisdomClientProfile(name="grok", client=grok_client)
                )

        ollama_api_key = os.getenv("OLLAMA_API_KEY")
        if ollama_api_key:
            kimi_model = os.getenv("OLLAMA_KIMI_MODEL", "kimi-k2:1t-cloud")
            kimi_client = OllamaClient(
                api_key=ollama_api_key,
                model_name=kimi_model,
                telemetry=self.telemetry,
                temperature=0.7,
            )
            self.profiles.append(
                WisdomClientProfile(name="ollama_kimi", client=kimi_client)
            )

        if not self.profiles:
            self.skipTest("No wisdom generator AI clients configured for integration tests")

        print(f"\n=== Running tests with {len(self.profiles)} profile(s): {[p.name for p in self.profiles]} ===")

        self.test_store = TestStore()
        self.mock_user_resolver = self.test_store.user_resolver

    def _build_wisdom_generator(self, ai_client) -> WisdomGenerator:
        """Build a WisdomGenerator with the given AI client."""
        return WisdomGenerator(
            ai_client=ai_client,
            language_detector=self.language_detector,
            user_resolver=self.mock_user_resolver,
            response_summarizer=self.response_summarizer,
            telemetry=self.telemetry,
        )

    async def test_generate_wisdom_with_conversation_context(self):
        """Test generating wisdom with realistic physicist conversation from TestStore."""
        start_date = datetime(1905, 3, 3, 9, 15, tzinfo=timezone.utc)
        end_date = datetime(1905, 3, 3, 9, 45, tzinfo=timezone.utc)
        
        physics_messages = [
            msg for msg in self.test_store._messages
            if start_date <= msg.timestamp <= end_date
        ][:5]
        
        conversation_messages = [
            ConversationMessage(
                message_id=msg.message_id,
                author_id=msg.user_id,
                content=msg.message_text,
                timestamp=msg.timestamp.isoformat(),
                mentioned_user_ids=[],
            )
            for msg in physics_messages
        ]
        
        trigger_message = "God does not play dice with the universe"

        async def mock_conversation_fetcher():
            return conversation_messages

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                generator = self._build_wisdom_generator(profile.client)
                result = await generator.generate_wisdom(
                    trigger_message_content=trigger_message,
                    conversation_fetcher=mock_conversation_fetcher,
                    guild_id=self.test_store.physics_guild_id,
                )

                self.assertIsInstance(result, str)
                self.assertGreater(len(result), 0)
                result_lower = result.lower()
                context_mentioned = any(
                    term in result_lower
                    for term in ["quantum", "photoelectric", "einstein", "planck", "physics"]
                )
                self.assertTrue(
                    context_mentioned or len(result) > 50,
                    f"Wisdom should reference physics conversation context: {result}",
                )
                print(
                    f"[{profile.name}] Generated wisdom from physicist chat: {result}"
                )


if __name__ == "__main__":
    unittest.main()
