"""
Integration tests for GeneralQueryGenerator.

Tests the handle_request method with conversation context and length constraints.
Uses unittest.IsolatedAsyncioTestCase for async testing as per project standards.
"""

import os
import unittest
from unittest.mock import AsyncMock
from dotenv import load_dotenv
from general_query_generator import GeneralQueryGenerator
from gemini_client import GeminiClient
from grok_client import GrokClient
from schemas import GeneralParams
from tests.null_telemetry import NullTelemetry

load_dotenv()


class TestGeneralQueryGeneratorIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for GeneralQueryGenerator."""
    
    def setUp(self):
        """Set up test dependencies."""
        
        self.telemetry = NullTelemetry()
        
        # Check for API keys and model names
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        gemini_model = os.getenv('GEMINI_FLASH_MODEL')
        grok_api_key = os.getenv('GROK_API_KEY')
        grok_model = os.getenv('GROK_MODEL')
        
        if not gemini_api_key:
            self.skipTest("GEMINI_API_KEY environment variable not set")
        if not gemini_model:
            self.skipTest("GEMINI_FLASH_MODEL environment variable not set")
        if not grok_api_key:
            self.skipTest("GROK_API_KEY environment variable not set")
        if not grok_model:
            self.skipTest("GROK_MODEL environment variable not set")
            
        self.gemini_client = GeminiClient(
            api_key=gemini_api_key,
            model_name=gemini_model,
            temperature=0.1,
            telemetry=self.telemetry
        )
        
        self.grok_client = GrokClient(
            api_key=grok_api_key,
            model_name=grok_model,
            temperature=0.1,
            telemetry=self.telemetry
        )
        
        self.generator = GeneralQueryGenerator(
            gemini_flash=self.gemini_client,
            grok=self.grok_client,
            claude=None,
            telemetry=self.telemetry
        )
    
    async def test_handle_request_with_conversation_context(self):
        """Test handle_request passes conversation context to the LLM."""
        # Mock conversation that mentions a specific item
        async def mock_conversation_fetcher():
            return [
                ("alice", "I love my pet dragon named Sparkles"),
                ("bob", "That's so cool!")
            ]
        
        params = GeneralParams(
            ai_backend="gemini_flash",
            temperature=0.5,
            cleaned_query="Tell me about Alice's pet"
        )
        
        result = await self.generator.handle_request(params, mock_conversation_fetcher)
        
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        # Verify context was passed - response should mention the dragon or its name
        result_lower = result.lower()
        context_mentioned = "sparkles" in result_lower or "dragon" in result_lower
        self.assertTrue(context_mentioned, f"Response should mention context from conversation: {result}")
    
    async def test_handle_request_respects_length_limit(self):
        """Test handle_request keeps responses reasonably sized for Discord chat."""
        # Mock conversation fetcher
        async def mock_conversation_fetcher():
            return []
        
        params = GeneralParams(
            ai_backend="gemini_flash",
            temperature=0.3,
            cleaned_query="Write a detailed essay about climate change, its causes, effects, and solutions"
        )
        
        result = await self.generator.handle_request(params, mock_conversation_fetcher)
        
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        # Verify response is reasonably sized for Discord chat (Discord limit is 2000 chars)
        self.assertLessEqual(len(result), 2000, f"Response should be under 2000 characters but was {len(result)}: {result}")


if __name__ == '__main__':
    unittest.main()