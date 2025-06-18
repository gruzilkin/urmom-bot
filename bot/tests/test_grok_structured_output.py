"""
Integration tests for GrokClient structured output functionality.

Tests the ability to generate structured responses using Pydantic models.
Uses unittest.IsolatedAsyncioTestCase for async testing as per project standards.
"""

import os
import unittest
from unittest.mock import Mock
from grok_client import GrokClient
from schemas import YesNo
from tests.null_telemetry import NullTelemetry


class TestGrokStructuredOutput(unittest.IsolatedAsyncioTestCase):
    """Integration tests for Grok structured output with Pydantic models."""
    
    def setUp(self):
        """Set up test dependencies."""
        self.telemetry = NullTelemetry()
        
        # Check for API key
        self.api_key = os.getenv('GROK_API_KEY')
        if not self.api_key:
            self.skipTest("GROK_API_KEY environment variable not set")
        
        # Get model from environment or use default
        self.model_name = os.getenv('GROK_MODEL', 'grok-2-latest')
            
        self.client = GrokClient(
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.1,  # Low temperature for consistent results
            telemetry=self.telemetry
        )
    
    async def test_yes_no_structured_output_yes(self):
        """Test YES/NO structured output returns YES for affirmative question."""
        message = "Is the sky blue?"
        prompt = "Answer with YES or NO only. Be direct and concise."
        
        result = await self.client.generate_content(
            message=message,
            prompt=prompt,
            response_schema=YesNo
        )
        
        # Verify we get a YesNo object back
        self.assertIsInstance(result, YesNo)
        self.assertEqual(result.answer, "YES")
    
    async def test_yes_no_structured_output_no(self):
        """Test YES/NO structured output returns NO for negative question."""
        message = "Is 2025 a leap year?"
        prompt = "Answer with YES or NO only. Be direct and concise."
        
        result = await self.client.generate_content(
            message=message,
            prompt=prompt,
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
        prompt = "Answer with YES or NO only."
        
        result = await self.client.generate_content(
            message=message,
            prompt=prompt,
            enable_grounding=True,
            response_schema=YesNo
        )
        
        # Verify we get a YesNo object back
        self.assertIsInstance(result, YesNo)
        self.assertEqual(result.answer, "YES")


if __name__ == '__main__':
    unittest.main()
