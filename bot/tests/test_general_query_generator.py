import unittest
from unittest.mock import AsyncMock, Mock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from general_query_generator import GeneralQueryGenerator
from tests.null_telemetry import NullTelemetry


class TestGeneralQueryGenerator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_ai_client = Mock()
        self.mock_ai_client.generate_content = AsyncMock()
        self.telemetry = NullTelemetry()
        self.generator = GeneralQueryGenerator(self.mock_ai_client, self.telemetry)

    async def test_is_general_query_valid_question(self):
        """Test that valid questions are recognized as general queries"""
        self.mock_ai_client.generate_content.return_value = "Yes"
        
        result = await self.generator.is_general_query("What's the weather today?")
        
        self.assertTrue(result)
        self.mock_ai_client.generate_content.assert_called_once()

    async def test_is_general_query_invalid_response(self):
        """Test that simple responses are not recognized as general queries"""
        self.mock_ai_client.generate_content.return_value = "No"
        
        result = await self.generator.is_general_query("lol")
        
        self.assertFalse(result)
        self.mock_ai_client.generate_content.assert_called_once()

    async def test_generate_general_response(self):
        """Test generating a general response"""
        self.mock_ai_client.generate_content.return_value = "The weather is sunny today!"
        
        # Mock conversation fetcher
        async def mock_conversation_fetcher():
            return [("user1", "What's the weather like?"), ("user2", "I hope it's nice")]
        
        result = await self.generator.generate_general_response(
            "What's the weather today?", 
            mock_conversation_fetcher
        )
        
        self.assertEqual(result, "The weather is sunny today!")
        self.mock_ai_client.generate_content.assert_called_once()

    async def test_generate_general_response_error_handling(self):
        """Test error handling in generate_general_response - exceptions should propagate"""
        self.mock_ai_client.generate_content.side_effect = Exception("API error")
        
        # Mock conversation fetcher
        async def mock_conversation_fetcher():
            return []
        
        # Exception should propagate instead of being caught
        with self.assertRaises(Exception) as context:
            await self.generator.generate_general_response(
                "What's the weather today?", 
                mock_conversation_fetcher
            )
        
        self.assertEqual(str(context.exception), "API error")


if __name__ == '__main__':
    unittest.main()
