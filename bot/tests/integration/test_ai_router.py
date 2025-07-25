"""
Integration tests for AiRouter.

Tests the route_request method to ensure it correctly processes user messages,
especially those requiring perspective-shifting and query cleaning.
"""

import os
import unittest
from unittest.mock import Mock, AsyncMock
from dotenv import load_dotenv

from ai_router import AiRouter
from gemma_client import GemmaClient
from general_query_generator import GeneralQueryGenerator
from famous_person_generator import FamousPersonGenerator
from fact_handler import FactHandler
from language_detector import LanguageDetector
from null_telemetry import NullTelemetry

load_dotenv()


class TestAiRouterIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for AiRouter."""

    def setUp(self):
        """Set up test dependencies."""
        self.telemetry = NullTelemetry()

        # API Keys and Models
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        gemma_model = os.getenv('GEMINI_GEMMA_MODEL')

        if not all([gemini_api_key, gemma_model]):
            self.skipTest("Missing GEMINI_API_KEY or GEMINI_GEMMA_MODEL environment variables.")

        # The router only needs one real AI client to make its decision
        self.gemma_client = GemmaClient(api_key=gemini_api_key, model_name=gemma_model, telemetry=self.telemetry)

        # Create mock dependencies for generators
        mock_user_resolver = Mock()
        mock_store = Mock()
        
        # The generators are only used for their `get_route_description` method,
        # which doesn't use any internal dependencies. We can pass None for them.
        self.famous_generator = FamousPersonGenerator(ai_client=None, response_summarizer=None, telemetry=self.telemetry, user_resolver=mock_user_resolver)
        self.general_generator = GeneralQueryGenerator(
            gemini_flash=None,
            grok=None,
            claude=None,
            gemma=None,
            response_summarizer=None,
            telemetry=self.telemetry,
            store=mock_store,
            user_resolver=mock_user_resolver,
            memory_manager=AsyncMock()
        )
        self.fact_handler = FactHandler(ai_client=None, store=None, telemetry=self.telemetry, user_resolver=mock_user_resolver)

        # The language detector is a required dependency
        self.language_detector = LanguageDetector(ai_client=self.gemma_client, telemetry=self.telemetry)

        # The component under test
        self.router = AiRouter(
            ai_client=self.gemma_client,
            telemetry=self.telemetry,
            language_detector=self.language_detector,
            famous_generator=self.famous_generator,
            general_generator=self.general_generator,
            fact_handler=self.fact_handler
        )

    async def test_route_request_with_perspective_shift(self):
        """
        Test that the router correctly rephrases a third-person request
        into a direct, second-person query.
        """
        user_message = "let's ask BOT to use grok and be creative to tell me a joke"
        expected_cleaned_query = "tell me a joke"

        # Act
        route, params = await self.router.route_request(user_message)

        # Assert
        self.assertEqual(route, "GENERAL")
        self.assertIsNotNone(params)
        self.assertEqual(params.ai_backend, "grok")
        self.assertEqual(params.cleaned_query.lower(), expected_cleaned_query.lower())
        self.assertGreaterEqual(params.temperature, 0.7, "Temperature should be high for a 'creative' request")

    async def test_route_request_with_direct_command(self):
        """
        Test that the router correctly handles a direct command to the bot,
        stripping out only the bot's name and routing hints.
        """
        user_message = "BOT, ask claude to write a technical blog post, be very detailed"
        expected_cleaned_query = "write a technical blog post"

        # Act
        route, params = await self.router.route_request(user_message)

        # Assert
        self.assertEqual(route, "GENERAL")
        self.assertIsNotNone(params)
        self.assertEqual(params.ai_backend, "claude")
        self.assertEqual(params.cleaned_query.lower(), expected_cleaned_query.lower())
        self.assertLessEqual(params.temperature, 0.3, "Temperature should be low for a 'detailed' request")

    async def test_route_request_memory_remember(self):
        """
        Test that the router correctly identifies and routes memory remember commands.
        """
        user_message = "Bot remember that <@1333878858138652682> works at TechCorp"

        # Act
        route, params = await self.router.route_request(user_message)

        # Assert
        self.assertEqual(route, "FACT")
        self.assertIsNotNone(params)
        self.assertEqual(params.operation, "remember")
        self.assertEqual(params.user_mention, "1333878858138652682")
        self.assertIn("TechCorp", params.fact_content)

    async def test_route_request_memory_forget(self):
        """
        Test that the router correctly identifies and routes memory forget commands.
        """
        user_message = "Bot forget that gruzilkin likes pizza"

        # Act
        route, params = await self.router.route_request(user_message)

        # Assert
        self.assertEqual(route, "FACT")
        self.assertIsNotNone(params)
        self.assertEqual(params.operation, "forget")
        self.assertEqual(params.user_mention, "gruzilkin")
        self.assertIn("pizza", params.fact_content.lower())
        self.assertIn("like", params.fact_content.lower())


if __name__ == '__main__':
    unittest.main()
