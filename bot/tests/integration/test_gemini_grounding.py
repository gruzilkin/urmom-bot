#!/usr/bin/env python3
"""
Test Gemini grounding capabilities (always enabled)
"""

import unittest
import os
from dotenv import load_dotenv
from gemini_client import GeminiClient
from tests.null_telemetry import NullTelemetry

load_dotenv()

class TestGeminiGrounding(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Check for API key and model name
        api_key = os.getenv('GEMINI_API_KEY')
        model_name = os.getenv('GEMINI_MODEL')
        
        if not api_key:
            self.skipTest("GEMINI_API_KEY environment variable not set")
        if not model_name:
            self.skipTest("GEMINI_MODEL environment variable not set")
        
        self.client = GeminiClient(
            api_key=api_key,
            model_name=model_name,
            temperature=0.1,  # Fixed temperature for test stability
            telemetry=NullTelemetry()
        )

    async def test_current_events_grounding(self):
        """Test that grounding provides current information for recent events"""
        response = await self.client.generate_content(
            message="What are the latest developments in AI in 2025?",
            prompt="You are a helpful AI assistant. Use web search to provide accurate and up-to-date information.",
            enable_grounding=True
        )
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 50)  # Should be a substantial response
        # Could add more specific assertions about content if needed

    async def test_current_weather_grounding(self):
        """Test grounding with current weather information"""
        response = await self.client.generate_content(
            message="What's the current weather in San Francisco?",
            prompt="Provide current weather information using web search."
        )
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 20)

    async def test_current_news_grounding(self):
        """Test grounding with current news"""
        response = await self.client.generate_content(
            message="What are the top news stories today?",
            prompt="Use web search to provide current news information."
        )
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 50)

    async def test_cryptocurrency_price_grounding(self):
        """Test grounding with current cryptocurrency prices"""
        response = await self.client.generate_content(
            message="What's the current price of Bitcoin?",
            prompt="Use web search to provide current cryptocurrency price information."
        )
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 20)

if __name__ == "__main__":
    unittest.main()
