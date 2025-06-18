"""
Integration tests for GeminiClient structured output functionality.

Tests the ability to generate structured responses using Pydantic models.
Uses unittest.IsolatedAsyncioTestCase for async testing as per project standards.
"""

import os
import unittest
from unittest.mock import Mock
from gemini_client import GeminiClient
from schemas import YesNo
from tests.null_telemetry import NullTelemetry


class TestGeminiStructuredOutput(unittest.IsolatedAsyncioTestCase):
    """Integration tests for Gemini structured output with Pydantic models."""
    
    def setUp(self):
        """Set up test dependencies."""
        self.telemetry = NullTelemetry()
        
        # Check for API key
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            self.skipTest("GEMINI_API_KEY environment variable not set")
            
        self.client = GeminiClient(
            api_key=self.api_key,
            model_name="gemini-2.5-flash",
            temperature=0.1,  # Low temperature for consistent results
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
    
    async def test_structured_output_with_grounding(self):
        """Test that using structured output with grounding raises appropriate error."""
        message = "Is Python a programming language?"
        
        # This should raise an error since structured output and grounding can't be used together
        with self.assertRaises(Exception) as context:
            await self.client.generate_content(
                message=message,
                enable_grounding=True,
                response_schema=YesNo
            )
        
        # Verify that the error mentions the conflict
        error_msg = str(context.exception)
        self.assertIn("Tool use with a response mime type", error_msg)


if __name__ == '__main__':
    unittest.main()
