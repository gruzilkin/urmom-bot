"""
Integration tests for GeneralQueryGenerator.

Tests the handle_request method with conversation context and length constraints.
Uses unittest.IsolatedAsyncioTestCase for async testing as per project standards.
"""

import os
import unittest
from unittest.mock import Mock

from dotenv import load_dotenv

from conversation_graph import ConversationMessage
from gemini_client import GeminiClient
from gemma_client import GemmaClient
from general_query_generator import GeneralQueryGenerator
from grok_client import GrokClient
from response_summarizer import ResponseSummarizer
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
        gemma_model = os.getenv('GEMINI_GEMMA_MODEL')
        grok_api_key = os.getenv('GROK_API_KEY')
        grok_model = os.getenv('GROK_MODEL')
        
        if not gemini_api_key:
            self.skipTest("GEMINI_API_KEY environment variable not set")
        if not gemini_model:
            self.skipTest("GEMINI_FLASH_MODEL environment variable not set")
        if not gemma_model:
            self.skipTest("GEMINI_GEMMA_MODEL environment variable not set")
        if not grok_api_key:
            self.skipTest("GROK_API_KEY environment variable not set")
        if not grok_model:
            self.skipTest("GROK_MODEL environment variable not set")
            
        self.gemini_client = GeminiClient(
            api_key=gemini_api_key,
            model_name=gemini_model,
            telemetry=self.telemetry,
            temperature=0.1
        )
        
        self.gemma_client = GemmaClient(
            api_key=gemini_api_key,
            model_name=gemma_model,
            telemetry=self.telemetry,
            temperature=0.1
        )
        
        self.grok_client = GrokClient(
            api_key=grok_api_key,
            model_name=grok_model,
            telemetry=self.telemetry,
            temperature=0.1
        )
        
        # Create response summarizer
        self.response_summarizer = ResponseSummarizer(self.gemma_client, self.telemetry)
        
        # Create mock dependencies
        self.mock_store = Mock()
        self.mock_store.get_user_facts = Mock(return_value=None)
        self.mock_user_resolver = Mock()
        self.mock_user_resolver.get_display_name = Mock(return_value="TestUser")
        
        self.generator = GeneralQueryGenerator(
            gemini_flash=self.gemini_client,
            grok=self.grok_client,
            claude=None,
            gemma=self.gemma_client,
            response_summarizer=self.response_summarizer,
            telemetry=self.telemetry,
            store=self.mock_store,
            user_resolver=self.mock_user_resolver
        )
    
    async def test_handle_request_with_conversation_context(self):
        """Test handle_request passes conversation context to the LLM."""
        # Mock conversation that mentions a specific item
        async def mock_conversation_fetcher():
            return [
                ConversationMessage(
                    author_id=1000,
                    content="I love my pet dragon named Sparkles",
                    timestamp="2024-01-01 11:55:00",
                    mentioned_user_ids=[]
                ),
                ConversationMessage(
                    author_id=2000,
                    content="That's so cool!",
                    timestamp="2024-01-01 11:58:00",
                    mentioned_user_ids=[]
                )
            ]
        
        params = GeneralParams(
            ai_backend="gemini_flash",
            temperature=0.5,
            cleaned_query="Tell me about Alice's pet"
        )
        
        result = await self.generator.handle_request(params, mock_conversation_fetcher, guild_id=12345)
        
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
        
        result = await self.generator.handle_request(params, mock_conversation_fetcher, guild_id=12345)
        
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        # Verify response is reasonably sized for Discord chat (Discord limit is 2000 chars)
        self.assertLessEqual(len(result), 2000, f"Response should be under 2000 characters but was {len(result)}: {result}")


if __name__ == '__main__':
    unittest.main()