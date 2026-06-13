"""
Integration tests for DeepSeekClient structured output functionality.

Tests the ability to generate structured responses using Pydantic models via
DeepSeek's OpenAI-compatible API. Uses unittest.IsolatedAsyncioTestCase for async
testing as per project standards.
"""

import os
import unittest
from dotenv import load_dotenv
from deepseek_client import DeepSeekClient
from schemas import YesNo
from null_telemetry import NullTelemetry

load_dotenv()


class TestDeepSeekStructuredOutput(unittest.IsolatedAsyncioTestCase):
    """Integration tests for DeepSeek structured output with Pydantic models."""

    def setUp(self):
        """Set up test dependencies."""
        self.telemetry = NullTelemetry()

        # Skip paid tests unless explicitly enabled
        if os.getenv("ENABLE_PAID_TESTS", "").lower() != "true":
            self.skipTest("Paid tests disabled (set ENABLE_PAID_TESTS=true to enable)")

        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

        if not self.api_key:
            self.skipTest("DEEPSEEK_API_KEY environment variable not set")

        self.client = DeepSeekClient(
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.1,  # Fixed temperature for test stability
            telemetry=self.telemetry,
        )

    async def test_yes_no_structured_output_yes(self):
        """Test YES/NO structured output returns YES for affirmative question."""
        message = "Is the sky blue?"

        result = await self.client.generate_content(message=message, response_schema=YesNo)

        self.assertIsInstance(result, YesNo)
        self.assertEqual(result.answer, "YES")

    async def test_yes_no_structured_output_no(self):
        """Test YES/NO structured output returns NO for negative question."""
        message = "Is 2025 a leap year?"

        result = await self.client.generate_content(message=message, response_schema=YesNo)

        self.assertIsInstance(result, YesNo)
        self.assertEqual(result.answer, "NO")

    async def test_string_output_without_schema(self):
        """Test that string output still works when no schema is provided."""
        message = "What color is the sky?"
        prompt = "Answer in one short sentence."

        result = await self.client.generate_content(message=message, prompt=prompt)

        self.assertIsInstance(result, str)
        self.assertIn("blue", result.lower())


if __name__ == "__main__":
    unittest.main()
