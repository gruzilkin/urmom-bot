"""
Integration tests for CodexClient structured output functionality.

Tests the ability to generate structured responses using Pydantic models.
Uses unittest.IsolatedAsyncioTestCase for async testing as per project standards.
"""

import base64
import unittest
from codex_client import CodexClient
from schemas import YesNo
from null_telemetry import NullTelemetry


class TestCodexStructuredOutput(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.telemetry = NullTelemetry()
        self.client = CodexClient(telemetry=self.telemetry, model_name="gpt-5.4")

    async def test_yes_no_structured_output_yes(self):
        message = "Is the sky blue?"

        result = await self.client.generate_content(message=message, response_schema=YesNo)

        self.assertIsInstance(result, YesNo)
        self.assertEqual(result.answer, "YES")

    async def test_yes_no_structured_output_no(self):
        message = "Is 2025 a leap year?"

        result = await self.client.generate_content(message=message, response_schema=YesNo)

        self.assertIsInstance(result, YesNo)
        self.assertEqual(result.answer, "NO")

    async def test_string_output_without_schema(self):
        message = "What color is the sky?"
        prompt = "Answer in one short sentence."

        result = await self.client.generate_content(message=message, prompt=prompt)

        self.assertIsInstance(result, str)
        self.assertIn("blue", result.lower())


class TestCodexMiniStructuredOutput(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.telemetry = NullTelemetry()
        self.client = CodexClient(telemetry=self.telemetry, model_name="gpt-5.4-mini")

    async def test_yes_no_structured_output_yes(self):
        message = "Is the sky blue?"

        result = await self.client.generate_content(message=message, response_schema=YesNo)

        self.assertIsInstance(result, YesNo)
        self.assertEqual(result.answer, "YES")

    async def test_yes_no_structured_output_no(self):
        message = "Is 2025 a leap year?"

        result = await self.client.generate_content(message=message, response_schema=YesNo)

        self.assertIsInstance(result, YesNo)
        self.assertEqual(result.answer, "NO")

    async def test_string_output_without_schema(self):
        message = "What color is the sky?"
        prompt = "Answer in one short sentence."

        result = await self.client.generate_content(message=message, prompt=prompt)

        self.assertIsInstance(result, str)
        self.assertIn("blue", result.lower())

    async def test_image_recognition(self):
        red_square_png = base64.b64decode(  # noqa: E501
            "iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAAA40lEQVR4nO3QsQEAIAyAsOr/P+sLZU9mJs4btu66xKzCrMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArNn7il4Bx2GaB88AAAAASUVORK5CYII="
        )

        result = await self.client.generate_content(
            message="What color is this image?",
            prompt="Answer in one word.",
            image_data=red_square_png,
            image_mime_type="image/png",
        )

        self.assertIsInstance(result, str)
        self.assertIn("red", result.lower())


if __name__ == "__main__":
    unittest.main()
