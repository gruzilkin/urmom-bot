"""
Integration tests for ClaudeClient structured output functionality.

Tests the ability to generate structured responses using Pydantic models.
Uses unittest.IsolatedAsyncioTestCase for async testing as per project standards.
"""

import unittest
import uuid
from claude_client import ClaudeClient
from schemas import YesNo
from null_telemetry import NullTelemetry


class TestClaudeOpusStructuredOutput(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.telemetry = NullTelemetry()
        self.client = ClaudeClient(telemetry=self.telemetry, model_name="claude-opus-5-5")

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

    async def test_system_prompt_is_applied(self):
        codeword = uuid.uuid4().hex[:8]
        prompt = f"The secret codeword is {codeword}. When asked for the codeword, reply with the codeword only."

        result = await self.client.generate_content(message="What is the codeword?", prompt=prompt)

        self.assertIn(codeword, result)

    async def test_system_prompt_is_applied_with_structured_output(self):
        prompt = "We are playing the opposite game: answer every yes/no question with the opposite of the truth."

        result = await self.client.generate_content(message="Is the sky blue?", prompt=prompt, response_schema=YesNo)

        self.assertIsInstance(result, YesNo)
        self.assertEqual(result.answer, "NO")


class TestClaudeHaikuStructuredOutput(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.telemetry = NullTelemetry()
        self.client = ClaudeClient(telemetry=self.telemetry, model_name="claude-haiku-4-5", enable_web_search=False)

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

    async def test_system_prompt_is_applied(self):
        codeword = uuid.uuid4().hex[:8]
        prompt = f"The secret codeword is {codeword}. When asked for the codeword, reply with the codeword only."

        result = await self.client.generate_content(message="What is the codeword?", prompt=prompt)

        self.assertIn(codeword, result)

    async def test_system_prompt_is_applied_with_structured_output(self):
        prompt = "We are playing the opposite game: answer every yes/no question with the opposite of the truth."

        result = await self.client.generate_content(message="Is the sky blue?", prompt=prompt, response_schema=YesNo)

        self.assertIsInstance(result, YesNo)
        self.assertEqual(result.answer, "NO")


if __name__ == "__main__":
    unittest.main()
