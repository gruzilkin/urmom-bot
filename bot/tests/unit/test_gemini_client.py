"""Unit tests for GeminiClient."""

import unittest
from unittest.mock import AsyncMock, Mock

from google.genai.types import (
    Candidate,
    Content,
    FinishReason,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    Part,
)
from gemini_client import GeminiClient
from null_telemetry import NullTelemetry


class TestGeminiClientMultipleParts(unittest.IsolatedAsyncioTestCase):
    """Test GeminiClient handling of multiple parts in response."""

    async def test_multiple_parts_returns_first_part_only(self) -> None:
        """Test that when Gemini returns multiple parts, only the first part's text is returned."""
        # Create response with multiple parts using actual Gemini types
        response = GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(
                        parts=[
                            Part(text="This is the first response part."),
                            Part(text="This is the second response part with slight variation."),
                        ],
                        role="model",
                    ),
                    finish_reason=FinishReason.STOP,
                    index=0,
                )
            ],
            usage_metadata=GenerateContentResponseUsageMetadata(
                prompt_token_count=100,
                candidates_token_count=50,
                total_token_count=150,
            ),
        )

        # Create mock client
        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=response)

        # Create GeminiClient with mock client
        client = GeminiClient(
            api_key="",
            model_name="test-model",
            telemetry=NullTelemetry(),
            temperature=0.1,
            client=mock_client,
        )

        # Call generate_content
        result = await client.generate_content(message="Test message", prompt="Test prompt")

        # Verify only the first part's text is returned
        self.assertEqual(result, "This is the first response part.")
        self.assertNotIn("second response part", result)


if __name__ == "__main__":
    unittest.main()
