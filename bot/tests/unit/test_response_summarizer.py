import unittest
from unittest.mock import AsyncMock, Mock
from discord_formatting import DISCORD_FORMATTING_INSTRUCTIONS
from response_summarizer import DISCORD_MESSAGE_LIMIT, ResponseSummarizer, is_unusable_summary
from null_telemetry import NullTelemetry


class TestIsUnusableSummary(unittest.TestCase):
    def test_normal_summary_is_usable(self):
        self.assertFalse(is_unusable_summary("A concise summary."))

    def test_summary_at_limit_is_usable(self):
        self.assertFalse(is_unusable_summary("x" * DISCORD_MESSAGE_LIMIT))

    def test_summary_over_limit_is_unusable(self):
        self.assertTrue(is_unusable_summary("x" * (DISCORD_MESSAGE_LIMIT + 1)))

    def test_empty_summary_is_unusable(self):
        self.assertTrue(is_unusable_summary(""))

    def test_whitespace_only_summary_is_unusable(self):
        self.assertTrue(is_unusable_summary("   \n\t  "))


class TestResponseSummarizer(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create mock AI client
        self.mock_ai_client = Mock()
        self.mock_ai_client.generate_content = AsyncMock()

        self.telemetry = NullTelemetry()
        self.summarizer = ResponseSummarizer(self.mock_ai_client, self.telemetry)

    async def test_short_response_passthrough(self):
        """Test that responses within limit are returned unchanged"""
        short_response = "This is a short response"

        result = await self.summarizer.process_response(short_response, max_length=2000)

        self.assertEqual(result, short_response)
        # AI client should not be called for short responses
        self.mock_ai_client.generate_content.assert_not_called()

    async def test_successful_summarization(self):
        """Test successful summarization of long response"""
        long_response = "x" * 2500  # Response longer than 2000 chars
        summarized_response = "x" * 1500  # Summarized version within limit

        self.mock_ai_client.generate_content.return_value = summarized_response

        result = await self.summarizer.process_response(long_response, max_length=2000)

        self.assertEqual(result, summarized_response)
        self.mock_ai_client.generate_content.assert_called_once()
        prompt = self.mock_ai_client.generate_content.await_args.kwargs["prompt"]
        self.assertIn(DISCORD_FORMATTING_INSTRUCTIONS, prompt)

    async def test_summarization_still_too_long_fallback(self):
        """Test fallback to truncation when summarization is still too long"""
        long_response = "x" * 2500  # Response longer than 2000 chars
        still_long_summary = "y" * 2100  # Summary still too long

        self.mock_ai_client.generate_content.return_value = still_long_summary

        result = await self.summarizer.process_response(long_response, max_length=2000)

        # Should fallback to truncation of original response
        expected_truncated = long_response[:1997] + "..."
        self.assertEqual(result, expected_truncated)
        self.mock_ai_client.generate_content.assert_called_once()

    async def test_summarization_exception_fallback(self):
        """Test fallback to truncation when summarization raises exception"""
        long_response = "x" * 2500  # Response longer than 2000 chars

        self.mock_ai_client.generate_content.side_effect = Exception("API error")

        result = await self.summarizer.process_response(long_response, max_length=2000)

        # Should fallback to truncation of original response
        expected_truncated = long_response[:1997] + "..."
        self.assertEqual(result, expected_truncated)
        self.mock_ai_client.generate_content.assert_called_once()


if __name__ == "__main__":
    unittest.main()
