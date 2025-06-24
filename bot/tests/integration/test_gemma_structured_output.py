"""
Integration tests for GemmaClient structured output functionality.

Tests the ability to generate structured responses using Pydantic models with manual JSON parsing.
Uses unittest.IsolatedAsyncioTestCase for async testing as per project standards.
"""

import os
import unittest
from dotenv import load_dotenv
from gemma_client import GemmaClient
from schemas import YesNo
from tests.null_telemetry import NullTelemetry

load_dotenv()


class TestGemmaStructuredOutput(unittest.IsolatedAsyncioTestCase):
    """Integration tests for Gemma structured output with manual JSON parsing."""
    
    def setUp(self):
        """Set up test dependencies."""
        self.telemetry = NullTelemetry()
        
        # Check for API key and model name
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.model_name = os.getenv('GEMINI_GEMMA_MODEL')
        
        if not self.api_key:
            self.skipTest("GEMINI_API_KEY environment variable not set")
        if not self.model_name:
            self.skipTest("GEMINI_GEMMA_MODEL environment variable not set")
            
        self.client = GemmaClient(
            api_key=self.api_key,
            model_name=self.model_name,
            telemetry=self.telemetry,
            temperature=0.1  # Fixed temperature for test stability
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
        
        # This should work but log a warning since grounding is not supported by Gemma
        result = await self.client.generate_content(
            message=message,
            enable_grounding=True,
            response_schema=YesNo
        )
        
        # Verify we still get a structured response despite grounding not being supported
        self.assertIsInstance(result, YesNo)
        self.assertEqual(result.answer, "YES")
    
    async def test_samples_warning(self):
        """Test that using samples logs warning but continues."""
        message = "Is Python a programming language?"
        samples = [("Test question", "Test answer")]
        
        # This should work but log a warning since samples are not supported in simplified mode
        result = await self.client.generate_content(
            message=message,
            samples=samples,
            response_schema=YesNo
        )
        
        # Verify we still get a structured response despite samples not being supported
        self.assertIsInstance(result, YesNo)
        self.assertEqual(result.answer, "YES")


if __name__ == '__main__':
    unittest.main()