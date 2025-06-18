"""
Integration tests for GeneralQueryGenerator with GeminiClient.

Tests the is_general_query method using structured output and generate_general_response method.
Uses unittest.IsolatedAsyncioTestCase for async testing as per project standards.
"""

import os
import unittest
from unittest.mock import AsyncMock
from dotenv import load_dotenv
from general_query_generator import GeneralQueryGenerator
from gemini_client import GeminiClient
from tests.null_telemetry import NullTelemetry

load_dotenv()


class TestGeneralQueryGeneratorIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for GeneralQueryGenerator with GeminiClient."""
    
    def setUp(self):
        """Set up test dependencies."""
        
        self.telemetry = NullTelemetry()
        
        # Check for API key and model name
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.model_name = os.getenv('GEMINI_MODEL')
        
        if not self.api_key:
            self.skipTest("GEMINI_API_KEY environment variable not set")
        if not self.model_name:
            self.skipTest("GEMINI_MODEL environment variable not set")
            
        self.gemini_client = GeminiClient(
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.1,  # Low temperature for consistent results
            telemetry=self.telemetry
        )
        
        self.generator = GeneralQueryGenerator(
            ai_client=self.gemini_client,
            telemetry=self.telemetry
        )
    
    async def test_is_general_query_valid_questions(self):
        """Test is_general_query returns True for valid general queries."""
        valid_queries = [
            "How does machine learning work?",
            "What are the benefits of exercise?",
            "Can you help me understand calculus?",
            "Where is the best place to learn programming?",
            "When did World War II end?"
        ]
        
        for query in valid_queries:
            with self.subTest(query=query):
                result = await self.generator.is_general_query(query)
                self.assertTrue(result, f"Expected True for query: {query}")
    
    async def test_is_general_query_invalid_reactions(self):
        """Test is_general_query returns False for reactions and acknowledgments."""
        invalid_queries = [
            "lol",
            "nice",
            "haha that's funny",
            "ok",
            "thanks"
        ]
        
        for query in invalid_queries:
            with self.subTest(query=query):
                result = await self.generator.is_general_query(query)
                self.assertFalse(result, f"Expected False for query: {query}")
    
    async def test_is_general_query_context_dependent(self):
        """Test is_general_query with context-dependent questions."""
        # Context-dependent questions that should be considered valid
        context_dependent_queries = [
            ("When was it?", True),  # Context-dependent but valid question
            ("What about this?", True),  # Context-dependent but valid
            ("How does that work?", True),  # Context-dependent but valid
            ("I need help", True),  # Clear request for assistance
            ("Explain this", True),  # Request for explanation
        ]
        
        for query, expected in context_dependent_queries:
            with self.subTest(query=query):
                result = await self.generator.is_general_query(query)
                self.assertEqual(result, expected, f"Expected {expected} for query: {query}")
    
    async def test_generate_general_response(self):
        """Test generate_general_response produces appropriate responses."""
        # Mock conversation fetcher
        async def mock_conversation_fetcher():
            return [
                ("user1", "Hey everyone!"),
                ("user2", "Hello there"),
                ("user1", "I'm learning to code")
            ]
        
        extracted_message = "What's your favorite programming language?"
        
        response = await self.generator.generate_general_response(
            extracted_message=extracted_message,
            conversation_fetcher=mock_conversation_fetcher
        )
        
        # Verify we get a string response
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        # Response should be under 1000 characters as specified in prompt
        self.assertLessEqual(len(response), 1000)
    
    async def test_generate_general_response_with_context(self):
        """Test generate_general_response uses conversation context appropriately."""
        # Mock conversation that mentions a specific topic
        async def mock_conversation_fetcher():
            return [
                ("alice", "I've been learning Python lately"),
                ("bob", "Python is great for data science"),
                ("alice", "I'm curious about its benefits")
            ]
        
        extracted_message = "Tell me more about Python's advantages"
        
        response = await self.generator.generate_general_response(
            extracted_message=extracted_message,
            conversation_fetcher=mock_conversation_fetcher
        )
        
        # Verify response references the conversation context
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        # Should mention Python since it was discussed in context
        self.assertIn("Python", response)


if __name__ == '__main__':
    unittest.main()