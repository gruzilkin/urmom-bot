"""
Integration tests for GrokClient structured output functionality.

Tests the ability to generate structured responses using Pydantic models.
Uses unittest.IsolatedAsyncioTestCase for async testing as per project standards.
"""

import os
import unittest
from unittest.mock import Mock
from dotenv import load_dotenv
from grok_client import GrokClient
from schemas import YesNo
from tests.null_telemetry import NullTelemetry

load_dotenv()


class TestGrokStructuredOutput(unittest.IsolatedAsyncioTestCase):
    """Integration tests for Grok structured output with Pydantic models."""
    
    def setUp(self):
        """Set up test dependencies."""
        self.telemetry = NullTelemetry()
        
        # Check for API key and model name
        self.api_key = os.getenv('GROK_API_KEY')
        self.model_name = os.getenv('GROK_MODEL')
        
        if not self.api_key:
            self.skipTest("GROK_API_KEY environment variable not set")
        if not self.model_name:
            self.skipTest("GROK_MODEL environment variable not set")
        
        self.client = GrokClient(
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.1,  # Fixed temperature for test stability
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
        """Test structured output works with grounding enabled."""
        message = "Is Python a programming language?"
        
        result = await self.client.generate_content(
            message=message,
            enable_grounding=True,
            response_schema=YesNo
        )
        
        # Verify we get a YesNo object back
        self.assertIsInstance(result, YesNo)
        self.assertEqual(result.answer, "YES")


if __name__ == '__main__':
    unittest.main()
