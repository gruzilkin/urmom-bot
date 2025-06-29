import unittest
from unittest.mock import AsyncMock, Mock
from general_query_generator import GeneralQueryGenerator
from schemas import GeneralParams
from conversation_graph import ConversationMessage
from tests.null_telemetry import NullTelemetry


class TestGeneralQueryGenerator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create four mock AI clients
        self.mock_gemini_flash = Mock()
        self.mock_gemini_flash.generate_content = AsyncMock()
        
        self.mock_grok = Mock()
        self.mock_grok.generate_content = AsyncMock()
        
        self.mock_claude = Mock()
        self.mock_claude.generate_content = AsyncMock()
        
        self.mock_gemma = Mock()
        self.mock_gemma.generate_content = AsyncMock()
        
        # Mock response summarizer that returns what it receives (passthrough)
        self.mock_response_summarizer = Mock()
        self.mock_response_summarizer.process_response = AsyncMock(side_effect=lambda x: x)
        
        # Mock store and user_resolver
        self.mock_store = Mock()
        self.mock_store.get_user_facts = Mock(return_value=None)
        
        self.mock_user_resolver = Mock()
        self.mock_user_resolver.get_display_name = AsyncMock(return_value="TestUser")
        self.mock_user_resolver.replace_user_mentions_with_names = AsyncMock(side_effect=lambda text, guild_id: text)
        
        self.telemetry = NullTelemetry()
        self.generator = GeneralQueryGenerator(
            self.mock_gemini_flash, 
            self.mock_grok, 
            self.mock_claude,
            self.mock_gemma,
            self.mock_response_summarizer,
            self.telemetry,
            self.mock_store,
            self.mock_user_resolver
        )


    async def test_handle_request_with_gemini_flash(self):
        """Test handling request with gemini_flash backend"""
        self.mock_gemini_flash.generate_content.return_value = "Quick answer!"
        
        params = GeneralParams(
            ai_backend="gemini_flash",
            temperature=0.3,
            cleaned_query="Simple question?"
        )
        
        # Mock conversation fetcher
        async def mock_conversation_fetcher():
            return []
        
        result = await self.generator.handle_request(params, mock_conversation_fetcher, guild_id=12345)
        
        self.assertEqual(result, "Quick answer!")
        self.mock_gemini_flash.generate_content.assert_called_once()
        self.mock_response_summarizer.process_response.assert_called_once()

    async def test_handle_request_with_grok(self):
        """Test handling request with grok backend"""
        self.mock_grok.generate_content.return_value = "Creative response!"
        
        params = GeneralParams(
            ai_backend="grok",
            temperature=0.8,
            cleaned_query="Be creative!"
        )
        
        # Mock conversation fetcher
        async def mock_conversation_fetcher():
            return [ConversationMessage(
                author_id=1000,
                content="Let's get creative",
                timestamp="2024-01-01 12:00:00",
                mentioned_user_ids=[]
            )]
        
        result = await self.generator.handle_request(params, mock_conversation_fetcher, guild_id=12345)
        
        self.assertEqual(result, "Creative response!")
        self.mock_grok.generate_content.assert_called_once()
        self.mock_response_summarizer.process_response.assert_called_once()

    async def test_handle_request_error_handling(self):
        """Test error handling in handle_request - exceptions should propagate"""
        self.mock_gemini_flash.generate_content.side_effect = Exception("API error")
        
        params = GeneralParams(
            ai_backend="gemini_flash",
            temperature=0.5,
            cleaned_query="What's the weather today?"
        )
        
        # Mock conversation fetcher
        async def mock_conversation_fetcher():
            return []
        
        # Exception should propagate instead of being caught
        with self.assertRaises(Exception) as context:
            await self.generator.handle_request(params, mock_conversation_fetcher, guild_id=12345)
        
        self.assertEqual(str(context.exception), "API error")
        # Response summarizer should not be called due to exception
        self.mock_response_summarizer.process_response.assert_not_called()

    def test_get_ai_client_selection(self):
        """Test that _get_ai_client selects the correct client"""
        self.assertEqual(self.generator._get_ai_client("gemini_flash"), self.mock_gemini_flash)
        self.assertEqual(self.generator._get_ai_client("grok"), self.mock_grok)
        self.assertEqual(self.generator._get_ai_client("claude"), self.mock_claude)
        self.assertEqual(self.generator._get_ai_client("gemma"), self.mock_gemma)
        
        with self.assertRaises(ValueError):
            self.generator._get_ai_client("invalid_backend")


if __name__ == '__main__':
    unittest.main()
