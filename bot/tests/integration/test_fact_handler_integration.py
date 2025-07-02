"""
Integration tests for FactHandler with real Gemma client.

Tests the FactHandler's memory operations using real AI structured output generation
with deterministic keyword-based assertions to verify memory content and language preservation.
"""

import os
import unittest
from unittest.mock import Mock
from dotenv import load_dotenv

from fact_handler import FactHandler
from gemma_client import GemmaClient
from tests.null_telemetry import NullTelemetry
from tests.test_store import TestStore
from tests.test_user_resolver import TestUserResolver

load_dotenv()


class TestFactHandlerIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for FactHandler with real Gemma AI client."""

    def setUp(self):
        """Set up test dependencies with real Gemma client."""
        self.telemetry = NullTelemetry()
        
        # Check for required environment variables
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        gemma_model = os.getenv('GEMINI_GEMMA_MODEL')
        
        if not all([gemini_api_key, gemma_model]):
            self.skipTest("Missing GEMINI_API_KEY or GEMINI_GEMMA_MODEL environment variables.")
        
        # Real Gemma client for structured output generation
        self.gemma_client = GemmaClient(
            api_key=gemini_api_key,
            model_name=gemma_model,
            telemetry=self.telemetry,
            temperature=0.0  # Deterministic for memory operations
        )
        
        # Test doubles for data persistence and user resolution
        self.test_store = TestStore()
        self.test_user_resolver = TestUserResolver()
        
        # Component under test with real AI client
        self.fact_handler = FactHandler(
            ai_client=self.gemma_client,
            store=self.test_store,
            telemetry=self.telemetry,
            user_resolver=self.test_user_resolver
        )
        
        # Test data - use physics guild and Einstein for consistency
        self.guild_id = self.test_user_resolver.physics_guild_id
        self.user_id = self.test_user_resolver.physicist_ids["Einstein"]

    async def test_remember_fact_for_new_user(self):
        """Test remembering a fact for a user with no prior memory using real Gemma."""
        # Arrange
        fact_content = "they like chocolate cake"
        
        # Einstein already exists in TestUserResolver, no need to add
        
        # Act - Use real Gemma to generate MemoryUpdate
        response = await self.fact_handler._remember_fact(self.guild_id, self.user_id, fact_content)
        
        # Assert - Check memory was saved with expected content
        saved_memory = await self.test_store.get_user_facts(self.guild_id, self.user_id)
        self.assertIsNotNone(saved_memory, "Memory should be saved to store")
        
        # Verify memory contains key facts
        saved_memory_lower = saved_memory.lower()
        self.assertIn("chocolate", saved_memory_lower, "Memory should contain 'chocolate'")
        self.assertIn("cake", saved_memory_lower, "Memory should contain 'cake'")
        
        # Verify confirmation message contains specific fact content
        self.assertIsNotNone(response, "Should return confirmation message")
        self.assertTrue(len(response) > 10, "Confirmation should be substantial")
        response_lower = response.lower()
        # Confirmation should mention the specific fact, not be generic
        fact_terms_in_confirmation = ["chocolate", "cake", "remember"]
        terms_found = sum(1 for term in fact_terms_in_confirmation if term in response_lower)
        self.assertGreaterEqual(terms_found, 2, f"Confirmation should mention specific fact content. Got: {response}")

    async def test_remember_additional_fact_merges_memory(self):
        """Test adding a second fact merges with existing memory using real Gemma."""
        # Arrange - Set up existing memory for Einstein  
        existing_memory = "they like chocolate cake"
        self.test_store.set_user_facts(self.guild_id, self.user_id, existing_memory)
        
        new_fact = "they work at Google"
        
        # Act - Add new fact with real Gemma merge
        response = await self.fact_handler._remember_fact(self.guild_id, self.user_id, new_fact)
        
        # Assert - Memory should contain both facts
        updated_memory = await self.test_store.get_user_facts(self.guild_id, self.user_id)
        updated_memory_lower = updated_memory.lower()
        
        # Should contain original fact
        self.assertIn("chocolate", updated_memory_lower, "Should retain original chocolate fact")
        self.assertIn("cake", updated_memory_lower, "Should retain original cake fact")
        
        # Should contain new fact
        self.assertIn("google", updated_memory_lower, "Should contain new Google fact")
        self.assertIn("work", updated_memory_lower, "Should contain work information")
        
        # Confirmation should mention the new fact specifically
        response_lower = response.lower()
        self.assertIn("google", response_lower, f"Confirmation should mention Google. Got: {response}")

    async def test_forget_existing_fact(self):
        """Test forgetting an existing fact using real Gemma fact detection."""
        # Arrange - Set up memory with multiple facts for Einstein
        existing_memory = "they like chocolate cake and they work at Google"
        self.test_store.set_user_facts(self.guild_id, self.user_id, existing_memory)
        
        fact_to_forget = "they work at Google"
        
        # Act - Use real Gemma to detect and remove fact
        response = await self.fact_handler._forget_fact(self.guild_id, self.user_id, fact_to_forget)
        
        # Assert - Memory should no longer contain forgotten fact
        updated_memory = await self.test_store.get_user_facts(self.guild_id, self.user_id)
        updated_memory_lower = updated_memory.lower()
        
        # Should NOT contain forgotten fact
        self.assertNotIn("google", updated_memory_lower, "Should have removed Google fact")
        self.assertNotIn("work", updated_memory_lower, "Should have removed work information")
        
        # Should still contain other facts
        self.assertIn("chocolate", updated_memory_lower, "Should retain chocolate fact")
        self.assertIn("cake", updated_memory_lower, "Should retain cake fact")
        
        # Confirmation should mention what was forgotten
        response_lower = response.lower()
        forgotten_terms = ["forgotten", "google", "work"]
        terms_found = sum(1 for term in forgotten_terms if term in response_lower)
        self.assertGreaterEqual(terms_found, 2, f"Confirmation should mention what was forgotten. Got: {response}")

    async def test_forget_nonexistent_fact(self):
        """Test forgetting a fact that doesn't exist using real Gemma detection."""
        # Arrange - Set up memory without the fact to forget for Einstein
        existing_memory = "they like chocolate cake"
        self.test_store.set_user_facts(self.guild_id, self.user_id, existing_memory)
        
        nonexistent_fact = "they dislike pizza"
        
        # Act - Try to forget non-existent fact
        response = await self.fact_handler._forget_fact(self.guild_id, self.user_id, nonexistent_fact)
        
        # Assert - Memory should be unchanged
        unchanged_memory = await self.test_store.get_user_facts(self.guild_id, self.user_id)
        self.assertEqual(unchanged_memory, existing_memory, "Memory should remain unchanged")
        
        # Confirmation should indicate fact was not found
        response_lower = response.lower()
        not_found_terms = ["not found", "couldn't find", "don't have"]
        found_not_found_term = any(term in response_lower for term in not_found_terms)
        self.assertTrue(found_not_found_term, f"Confirmation should indicate fact not found. Got: {response}")

    async def test_forget_fact_no_existing_memory(self):
        """Test forgetting from a user with no existing memory."""
        # Arrange - No existing memory for Einstein (test store starts empty)
        
        fact_to_forget = "they like pizza"
        
        # Act - Try to forget from empty memory
        response = await self.fact_handler._forget_fact(self.guild_id, self.user_id, fact_to_forget)
        
        # Assert - No memory should be saved (nothing to update)
        memory = await self.test_store.get_user_facts(self.guild_id, self.user_id)
        self.assertIsNone(memory, "No memory should exist for user")
        
        # Confirmation should indicate no memory exists
        response_lower = response.lower()
        no_memory_terms = ["no memory", "don't have", "no information"]
        found_no_memory_term = any(term in response_lower for term in no_memory_terms)
        self.assertTrue(found_no_memory_term, f"Confirmation should indicate no memory exists. Got: {response}")

    async def test_language_preservation_english_fact(self):
        """Test that English facts generate English confirmations."""
        # Arrange - Test with Einstein
        english_fact = "they are a software engineer"
        
        # Act
        response = await self.fact_handler._remember_fact(self.guild_id, self.user_id, english_fact)
        
        # Assert - Memory and confirmation should be in English
        saved_memory = await self.test_store.get_user_facts(self.guild_id, self.user_id)
        self.assertIn("engineer", saved_memory.lower(), "Memory should contain engineering fact")
        
        # Confirmation should be in English and contain specific fact
        self.assertIn("engineer", response.lower(), "Confirmation should mention engineer")
        # Should not contain non-English text patterns
        non_english_patterns = ["я ", "что ", "они "]  # Basic Russian patterns
        has_non_english = any(pattern in response.lower() for pattern in non_english_patterns)
        self.assertFalse(has_non_english, f"English fact should generate English confirmation. Got: {response}")

    async def test_confirmation_message_quality(self):
        """Test that confirmation messages are specific and informative."""
        # Arrange - Test with Einstein
        specific_fact = "they graduated from MIT in 2020"
        
        # Act
        response = await self.fact_handler._remember_fact(self.guild_id, self.user_id, specific_fact)
        
        # Assert - Confirmation should be specific, not generic
        response_lower = response.lower()
        
        # Should contain specific fact details
        specific_terms = ["mit", "graduated", "2020"]
        terms_found = sum(1 for term in specific_terms if term in response_lower)
        self.assertGreaterEqual(terms_found, 2, f"Confirmation should mention specific details. Got: {response}")
        
        # Should not be overly generic
        generic_phrases = ["i'll remember that about", "information stored", "data saved"]
        is_generic = any(phrase in response_lower for phrase in generic_phrases)
        self.assertFalse(is_generic, f"Confirmation should be specific, not generic. Got: {response}")
        
        # Should contain action word
        action_words = ["remember", "recall", "note"]
        has_action = any(word in response_lower for word in action_words)
        self.assertTrue(has_action, f"Confirmation should indicate the action taken. Got: {response}")


if __name__ == '__main__':
    unittest.main()