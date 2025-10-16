"""
Unit tests for AiRouter.

Tests the NOTSURE fallback behavior using CompositeAIClient with mocked dependencies.
"""

import unittest
from unittest.mock import Mock, AsyncMock

from ai_router import AiRouter
from ai_client_wrappers import CompositeAIClient
from schemas import RouteSelection, GeneralParams
from null_telemetry import NullTelemetry


class TestAiRouterUnit(unittest.IsolatedAsyncioTestCase):
    """Unit tests for AiRouter with mocked dependencies."""

    def setUp(self):
        """Set up test dependencies with mocks."""
        self.telemetry = NullTelemetry()
        
        # Mock AI clients
        self.primary_client = Mock()
        self.fallback_client = Mock()
        
        # Mock language detector
        self.language_detector = Mock()
        self.language_detector.detect_language = AsyncMock(return_value="en")
        self.language_detector.get_language_name = AsyncMock(return_value="English")
        
        # Mock generators - only need route descriptions for prompts
        self.famous_generator = Mock()
        self.famous_generator.get_route_description.return_value = "FAMOUS: For impersonating famous people"
        
        self.general_generator = Mock()
        self.general_generator.get_route_description.return_value = "GENERAL: For general AI queries"
        self.general_generator.get_parameter_schema.return_value = GeneralParams
        self.general_generator.get_parameter_extraction_prompt.return_value = "Extract parameters"
        
        self.fact_handler = Mock()
        self.fact_handler.get_route_description.return_value = "FACT: For memory operations"
        
        # The router client will be a composite client that handles the NOTSURE fallback.
        router_client = CompositeAIClient(
            [self.primary_client, self.fallback_client],
            telemetry=self.telemetry,
            is_bad_response=lambda r: hasattr(r, 'route') and r.route == "NOTSURE"
        )

        # Create router with mocked dependencies
        self.router = AiRouter(
            ai_client=router_client,
            telemetry=self.telemetry,
            language_detector=self.language_detector,
            famous_generator=self.famous_generator,
            general_generator=self.general_generator,
            fact_handler=self.fact_handler
        )

    async def test_notsure_triggers_fallback(self):
        """
        Test that when primary client returns NOTSURE, fallback client is called
        and its response is used as the final result.
        """
        self.primary_client.generate_content = AsyncMock(side_effect=[
            RouteSelection(route="NOTSURE", reason="Message is ambiguous"),
            GeneralParams(ai_backend="gemini_flash", temperature=0.5, cleaned_query="test query")
        ])

        self.fallback_client.generate_content = AsyncMock(
            return_value=RouteSelection(route="GENERAL", reason="Definitively a general query")
        )

        route, params = await self.router.route_request("What is the weather like?")

        self.fallback_client.generate_content.assert_called_once()
        self.assertEqual(route, "GENERAL")
        self.assertEqual(self.primary_client.generate_content.call_count, 2)

    async def test_no_fallback_for_definitive_route(self):
        """
        Test that when primary client returns a definitive route (not NOTSURE),
        fallback client is NOT called.
        """
        self.primary_client.generate_content = AsyncMock(side_effect=[
            RouteSelection(route="GENERAL", reason="Clear general query"),
            GeneralParams(ai_backend="gemini_flash", temperature=0.5, cleaned_query="test query")
        ])

        route, params = await self.router.route_request("What is the weather like?")

        self.fallback_client.generate_content.assert_not_called()
        self.assertEqual(route, "GENERAL")
        self.assertEqual(self.primary_client.generate_content.call_count, 2)

    async def test_fallback_with_none_route(self):
        """
        Test that NOTSURE fallback works correctly when result is NONE
        (which doesn't require parameter extraction).
        """
        self.primary_client.generate_content = AsyncMock(
            return_value=RouteSelection(route="NOTSURE", reason="Unclear intent")
        )

        self.fallback_client.generate_content = AsyncMock(
            return_value=RouteSelection(route="NONE", reason="Just an acknowledgment")
        )

        route, params = await self.router.route_request("ok")

        self.fallback_client.generate_content.assert_called_once()
        self.assertEqual(route, "NONE")
        self.assertIsNone(params)
        self.assertEqual(self.primary_client.generate_content.call_count, 1)


if __name__ == '__main__':
    unittest.main()
