"""
Integration tests for GeneralQueryGenerator.

Tests the handle_request method with conversation context and length constraints.
Uses unittest.IsolatedAsyncioTestCase for async testing as per project standards.
"""

import os
import re
import unittest
from unittest.mock import Mock, AsyncMock

from dotenv import load_dotenv

from conversation_formatter import ConversationFormatter
from conversation_graph import ConversationMessage
from gemini_client import GeminiClient
from gemma_client import GemmaClient
from general_query_generator import GeneralQueryGenerator
from response_summarizer import ResponseSummarizer
from schemas import GeneralParams
from null_telemetry import NullTelemetry

load_dotenv()


class TestGeneralQueryGeneratorIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for GeneralQueryGenerator."""

    def setUp(self):
        """Set up test dependencies."""
        self.telemetry = NullTelemetry()

        gemini_api_key = os.getenv("GEMINI_API_KEY")
        gemini_model = os.getenv("GEMINI_FLASH_MODEL")
        gemma_api_key = os.getenv("GEMMA_API_KEY")
        gemma_model = os.getenv("GEMMA_MODEL")

        if not gemini_api_key:
            self.skipTest("GEMINI_API_KEY environment variable not set")
        if not gemini_model:
            self.skipTest("GEMINI_FLASH_MODEL environment variable not set")
        if not gemma_api_key:
            self.skipTest("GEMMA_API_KEY environment variable not set")
        if not gemma_model:
            self.skipTest("GEMMA_MODEL environment variable not set")

        self.gemini_client = GeminiClient(
            api_key=gemini_api_key,
            model_name=gemini_model,
            telemetry=self.telemetry,
            temperature=0.1,
        )

        self.gemma_client = GemmaClient(
            api_key=gemma_api_key,
            model_name=gemma_model,
            telemetry=self.telemetry,
            temperature=0.1,
        )

        self.response_summarizer = ResponseSummarizer(self.gemma_client, self.telemetry)

        self.mock_store = Mock()
        self.mock_store.get_user_facts = AsyncMock(return_value=None)
        self.mock_user_resolver = Mock()
        self.mock_user_resolver.get_display_name = AsyncMock(return_value="TestUser")
        self.mock_user_resolver.replace_user_mentions_with_names = AsyncMock(
            side_effect=lambda text, guild_id: text
        )

        self.mock_bot_user = Mock()
        self.mock_bot_user.name = "urmom-bot"
        self.mock_bot_user.id = 99999

        self.conversation_formatter = ConversationFormatter(self.mock_user_resolver)

        self.mock_memory_manager = Mock()
        self.mock_memory_manager.build_memory_prompt = AsyncMock(return_value="")

        self.generator = GeneralQueryGenerator(
            client_selector=lambda _: self.gemini_client,
            response_summarizer=self.response_summarizer,
            telemetry=self.telemetry,
            store=self.mock_store,
            conversation_formatter=self.conversation_formatter,
            memory_manager=self.mock_memory_manager,
        )

    async def test_handle_request_with_conversation_context(self):
        """Test handle_request passes conversation context to the LLM."""

        async def mock_conversation_fetcher():
            return [
                ConversationMessage(
                    message_id=100001,
                    author_id=1000,
                    content="I love my pet dragon named Sparkles",
                    timestamp="2024-01-01 11:55:00",
                    mentioned_user_ids=[],
                ),
                ConversationMessage(
                    message_id=100002,
                    author_id=2000,
                    content="That's so cool!",
                    timestamp="2024-01-01 11:58:00",
                    mentioned_user_ids=[],
                    reply_to_id=100001,
                ),
            ]

        params = GeneralParams(
            ai_backend="gemini_flash",
            temperature=0.5,
            cleaned_query="What do you know about pets from our conversation?",
            language_code="en",
            language_name="English",
        )

        result = await self.generator.handle_request(
            params,
            mock_conversation_fetcher,
            guild_id=12345,
            bot_user=self.mock_bot_user,
        )

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        result_lower = result.lower()
        context_mentioned = "sparkles" in result_lower or "dragon" in result_lower
        self.assertTrue(
            context_mentioned,
            f"Response should mention context from conversation: {result}",
        )

    async def test_handle_request_respects_length_limit(self):
        """Test handle_request keeps responses reasonably sized for Discord chat."""

        async def mock_conversation_fetcher():
            return []

        params = GeneralParams(
            ai_backend="gemini_flash",
            temperature=0.3,
            cleaned_query="Write a detailed essay about climate change, its causes, effects, and solutions",
            language_code="en",
            language_name="English",
        )

        result = await self.generator.handle_request(
            params,
            mock_conversation_fetcher,
            guild_id=12345,
            bot_user=self.mock_bot_user,
        )

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertLessEqual(
            len(result),
            2000,
            f"Response should be under 2000 characters but was {len(result)}: {result}",
        )

    async def test_translation_request_produces_target_language_response(self):
        """Test that translation requests produce responses in the target language."""

        async def mock_conversation_fetcher():
            return []

        params = GeneralParams(
            ai_backend="gemini_flash",
            temperature=0.3,
            cleaned_query="translate 'Hello, how are you?' into Russian",
            language_code="en",
            language_name="English",
        )

        result = await self.generator.handle_request(
            params,
            mock_conversation_fetcher,
            guild_id=12345,
            bot_user=self.mock_bot_user,
        )

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

        contains_cyrillic = bool(re.search(r"[а-яё]", result.lower()))
        self.assertTrue(
            contains_cyrillic, f"Response should contain Russian text but got: {result}"
        )


if __name__ == "__main__":
    unittest.main()
