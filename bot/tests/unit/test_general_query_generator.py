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
        
        self.telemetry = NullTelemetry()
        self.generator = GeneralQueryGenerator(
            self.mock_gemini_flash, 
            self.mock_grok, 
            self.mock_claude,
            self.mock_gemma,
            self.telemetry
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
        
        result = await self.generator.handle_request(params, mock_conversation_fetcher)
        
        self.assertEqual(result, "Quick answer!")
        self.mock_gemini_flash.generate_content.assert_called_once()

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
                author_name="user1",
                content="Let's get creative",
                timestamp="2024-01-01 12:00:00"
            )]
        
        result = await self.generator.handle_request(params, mock_conversation_fetcher)
        
        self.assertEqual(result, "Creative response!")
        self.mock_grok.generate_content.assert_called_once()

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
            await self.generator.handle_request(params, mock_conversation_fetcher)
        
        self.assertEqual(str(context.exception), "API error")

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
