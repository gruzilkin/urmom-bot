"""
Unit tests for AiRouter.

Tests the routing logic with mocked dependencies to verify fallback behavior
and confidence-based routing without requiring actual AI API calls.
"""

import unittest
from unittest.mock import Mock, AsyncMock

from ai_router import AiRouter
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
        
        # Create router with mocked dependencies
        self.router = AiRouter(
            ai_client=self.primary_client,
            telemetry=self.telemetry,
            language_detector=self.language_detector,
            famous_generator=self.famous_generator,
            general_generator=self.general_generator,
            fact_handler=self.fact_handler,
            fallback_client=self.fallback_client
        )

    async def test_notsure_triggers_fallback(self):
        """
        Test that when primary client returns NOTSURE, fallback client is called
        and its response is used as the final result.
        """
        # Arrange - primary client: first call returns NOTSURE, second call does parameter extraction
        self.primary_client.generate_content = AsyncMock(side_effect=[
            RouteSelection(route="NOTSURE", reason="Message is ambiguous"),  # Route selection
            GeneralParams(ai_backend="gemini_flash", temperature=0.5, cleaned_query="test query")  # Parameter extraction
        ])
        
        # Arrange - fallback client returns definitive route
        self.fallback_client.generate_content = AsyncMock(
            return_value=RouteSelection(route="GENERAL", reason="Definitively a general query")
        )
        
        # Act
        route, params = await self.router.route_request("What is the weather like?")
        
        # Assert - fallback client was called for route selection
        self.fallback_client.generate_content.assert_called_once()
        
        # Assert - final route comes from fallback client
        self.assertEqual(route, "GENERAL")

    async def test_no_fallback_for_definitive_route(self):
        """
        Test that when primary client returns a definitive route (not NOTSURE),
        fallback client is NOT called.
        """
        # Arrange - primary client returns definitive route
        self.primary_client.generate_content = AsyncMock(side_effect=[
            RouteSelection(route="GENERAL", reason="Clear general query"),  # Route selection
            GeneralParams(ai_backend="gemini_flash", temperature=0.5, cleaned_query="test query")  # Parameter extraction
        ])
        
        # Act
        route, params = await self.router.route_request("What is the weather like?")
        
        # Assert - fallback client was NOT called
        self.fallback_client.generate_content.assert_not_called()
        
        # Assert - final route comes from primary client
        self.assertEqual(route, "GENERAL")
        
        # Assert - primary client was called for route selection and parameter extraction
        self.assertEqual(self.primary_client.generate_content.call_count, 2)

    async def test_fallback_with_none_route(self):
        """
        Test that NOTSURE fallback works correctly when result is NONE
        (which doesn't require parameter extraction).
        """
        # Arrange - primary returns NOTSURE, fallback returns NONE
        self.primary_client.generate_content = AsyncMock(
            return_value=RouteSelection(route="NOTSURE", reason="Unclear intent")
        )
        
        self.fallback_client.generate_content = AsyncMock(
            return_value=RouteSelection(route="NONE", reason="Just an acknowledgment")
        )
        
        # Act
        route, params = await self.router.route_request("ok")
        
        # Assert - fallback was called
        self.fallback_client.generate_content.assert_called_once()
        
        # Assert - final result is NONE with no parameters
        self.assertEqual(route, "NONE")
        self.assertIsNone(params)
        
        # Assert - primary client only called once (no parameter extraction for NONE)
        self.assertEqual(self.primary_client.generate_content.call_count, 1)


if __name__ == '__main__':
    unittest.main()