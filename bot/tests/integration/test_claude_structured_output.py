"""
Integration tests for ClaudeClient structured output functionality.

Tests the ability to generate structured responses using Pydantic models.
Uses unittest.IsolatedAsyncioTestCase for async testing as per project standards.
"""

import unittest
from claude_client import ClaudeClient
from schemas import YesNo
from tests.null_telemetry import NullTelemetry


class TestClaudeStructuredOutput(unittest.IsolatedAsyncioTestCase):
    """Integration tests for Claude structured output with Pydantic models."""
    
    def setUp(self):
        """Set up test dependencies."""
        self.telemetry = NullTelemetry()
        
        self.client = ClaudeClient(
            telemetry=self.telemetry
        )
    
    async def test_yes_no_structured_output_yes(self):
        """Test YES/NO structured output returns YES for affirmative question."""
        message = "Is the sky blue?"
        
        result = await self.client.generate_content(
            message=message,
            response_schema=YesNo
        )
        
        # Verify we get a YesNo object back
        self.assertIsInstance(result, YesNo)
        self.assertEqual(result.answer, "YES")
    
    async def test_yes_no_structured_output_no(self):
        """Test YES/NO structured output returns NO for negative question."""
        message = "Is 2025 a leap year?"
        
        result = await self.client.generate_content(
            message=message,
            response_schema=YesNo
        )
        
        # Verify we get a YesNo object back
        self.assertIsInstance(result, YesNo)
        self.assertEqual(result.answer, "NO")
    
    async def test_string_output_without_schema(self):
        """Test that string output still works when no schema is provided."""
        message = "What color is the sky?"
        prompt = "Answer in one short sentence."
        
        result = await self.client.generate_content(
            message=message,
            prompt=prompt
        )
        
        # Verify we get a string back
        self.assertIsInstance(result, str)
        self.assertIn("blue", result.lower())
    
    async def test_structured_output_with_grounding_warning(self):
        """Test that using structured output with grounding logs warning but continues."""
        message = "Is Python a programming language?"
        
        # This should work but log a warning since grounding is not supported by Claude CLI
        result = await self.client.generate_content(
            message=message,
            enable_grounding=True,
            response_schema=YesNo
        )
        
        # Verify we still get a structured response despite grounding not being supported
        self.assertIsInstance(result, YesNo)
        self.assertEqual(result.answer, "YES")


if __name__ == '__main__':
    unittest.main()