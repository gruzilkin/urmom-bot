"""
Integration tests for Gemini Pro and Flash model availability.

Tests that both Gemini Pro and Flash models are accessible and can handle basic requests.
This is useful for detecting when Google changes model names or availability on different tiers.
"""

import os
import unittest
from dotenv import load_dotenv
from gemini_client import GeminiClient
from tests.null_telemetry import NullTelemetry

load_dotenv()


class TestGeminiModelAvailability(unittest.IsolatedAsyncioTestCase):
    """Integration tests to verify Gemini Pro and Flash models are available."""
    
    def setUp(self):
        """Set up test dependencies."""
        self.telemetry = NullTelemetry()
        
        # Check for API key
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            self.skipTest("GEMINI_API_KEY environment variable not set")
    
    async def test_gemini_pro_model_availability(self):
        """Test that Gemini Pro model is available and can handle a basic request."""
        model_name = os.getenv('GEMINI_PRO_MODEL')
        if not model_name:
            self.skipTest("GEMINI_PRO_MODEL environment variable not set")
        
        client = GeminiClient(
            api_key=self.api_key,
            model_name=model_name,
            temperature=0.1,
            telemetry=self.telemetry
        )
        
        # Simple test request - successful API call is enough
        response = await client.generate_content(
            message="Hello",
            prompt="Respond briefly."
        )
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    async def test_gemini_flash_model_availability(self):
        """Test that Gemini Flash model is available and can handle a basic request."""
        model_name = os.getenv('GEMINI_FLASH_MODEL')
        if not model_name:
            self.skipTest("GEMINI_FLASH_MODEL environment variable not set")
        
        client = GeminiClient(
            api_key=self.api_key,
            model_name=model_name,
            temperature=0.1,
            telemetry=self.telemetry
        )
        
        # Simple test request - successful API call is enough
        response = await client.generate_content(
            message="Hello",
            prompt="Respond briefly."
        )
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)


if __name__ == '__main__':
    unittest.main()