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
from ai_router import AiRouter
from gemma_client import GemmaClient
from language_detector import LanguageDetector
from null_telemetry import NullTelemetry
from test_store import TestStore
from test_user_resolver import TestUserResolver

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
        
        # Create real instances for route descriptions but mock parameter extraction to prevent execution
        from famous_person_generator import FamousPersonGenerator
        from general_query_generator import GeneralQueryGenerator
        
        # Real famous generator instance (with mocked dependencies since we only need route description)
        self.famous_generator = FamousPersonGenerator(
            ai_client=Mock(),
            response_summarizer=Mock(), 
            telemetry=self.telemetry,
            user_resolver=Mock()
        )
        # Mock parameter extraction to fail fast if wrong route taken
        self.famous_generator.get_parameter_schema = Mock(side_effect=Exception("Should not extract FAMOUS params in FACT tests"))
        self.famous_generator.get_parameter_extraction_prompt = Mock(side_effect=Exception("Should not extract FAMOUS params in FACT tests"))
        
        # Real general generator instance (with mocked dependencies since we only need route description)
        self.general_generator = GeneralQueryGenerator(
            gemini_flash=Mock(),
            grok=Mock(),
            claude=Mock(),
            gemma=Mock(),
            response_summarizer=Mock(),
            telemetry=self.telemetry,
            store=Mock(),
            user_resolver=Mock(),
            memory_manager=Mock()
        )
        # Mock parameter extraction to fail fast if wrong route taken
        self.general_generator.get_parameter_schema = Mock(side_effect=Exception("Should not extract GENERAL params in FACT tests"))
        self.general_generator.get_parameter_extraction_prompt = Mock(side_effect=Exception("Should not extract GENERAL params in FACT tests"))
        
        # AiRouter for end-to-end testing
        self.language_detector = LanguageDetector(ai_client=self.gemma_client, telemetry=self.telemetry)
        self.ai_router = AiRouter(
            ai_client=self.gemma_client,
            telemetry=self.telemetry,
            language_detector=self.language_detector,
            famous_generator=self.famous_generator,
            general_generator=self.general_generator,
            fact_handler=self.fact_handler
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
        response = await self.fact_handler._remember_fact(self.guild_id, self.user_id, fact_content, "English")
        
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
        response = await self.fact_handler._remember_fact(self.guild_id, self.user_id, new_fact, "English")
        
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
        response = await self.fact_handler._forget_fact(self.guild_id, self.user_id, fact_to_forget, "English")
        
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
        response = await self.fact_handler._forget_fact(self.guild_id, self.user_id, nonexistent_fact, "English")
        
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
        response = await self.fact_handler._forget_fact(self.guild_id, self.user_id, fact_to_forget, "English")
        
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
        response = await self.fact_handler._remember_fact(self.guild_id, self.user_id, english_fact, "English")
        
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
        response = await self.fact_handler._remember_fact(self.guild_id, self.user_id, specific_fact, "English")
        
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

    async def test_end_to_end_russian_language_preservation(self):
        """Test end-to-end flow: AiRouter parameter extraction → FactHandler with Russian language preservation."""
        # Arrange - Russian fact message (similar to the case that failed in production)
        russian_message = "БОТ запомни что Rutherford так же известен как Крокодил"
        
        # Act - Use AiRouter to route and extract parameters (this is where the bug occurred)
        route, params = await self.ai_router.route_request(russian_message)
        
        # Assert - Route should be FACT
        self.assertEqual(route, "FACT", "Should route Russian memory command to FACT")
        self.assertIsNotNone(params, "Should extract parameters for FACT route")
        self.assertEqual(params.operation, "remember", "Should extract 'remember' operation")
        self.assertEqual(params.user_mention, "Rutherford", "Should extract user mention")
        
        # Critical: fact_content should preserve Russian language  
        fact_content_lower = params.fact_content.lower()
        
        # Should contain Russian words (не должно быть переведено на английский)
        russian_terms = ["известен", "крокодил"]  # Key Russian words that should be preserved
        russian_found = sum(1 for term in russian_terms if term in fact_content_lower)
        self.assertGreaterEqual(russian_found, 1, f"Fact content should preserve Russian words. Got: {params.fact_content}")
        
        # Should NOT contain English translations
        english_translations = ["known as", "also known"]  # English translations that indicate the bug
        english_found = any(term in fact_content_lower for term in english_translations)
        self.assertFalse(english_found, f"Fact content should NOT be translated to English. Got: {params.fact_content}")
        
        # Should be in third person perspective (он/они instead of original form)
        third_person_indicators = ["он", "они"]  # Russian third person pronouns
        has_third_person = any(pronoun in fact_content_lower for pronoun in third_person_indicators)
        self.assertTrue(has_third_person, f"Should convert to third person in Russian. Got: {params.fact_content}")
        
        # Act - Continue with fact handling using extracted parameters
        response = await self.fact_handler.handle_request(params, self.guild_id)
        
        # Assert - Memory should be saved with Russian content
        rutherford_id = self.test_user_resolver.physicist_ids["Rutherford"]
        saved_memory = await self.test_store.get_user_facts(self.guild_id, rutherford_id)
        self.assertIsNotNone(saved_memory, "Memory should be saved")
        
        saved_memory_lower = saved_memory.lower()
        self.assertIn("крокодил", saved_memory_lower, "Saved memory should contain Russian nickname 'Крокодил'")
        
        # Confirmation should also be in Russian (matching the fact content language)
        response_lower = response.lower()
        # Should contain Russian confirmation terms or at least the Russian nickname
        russian_confirmation_terms = ["крокодил", "запомн"]  # Russian nickname and remember-related words
        confirmation_russian_found = any(term in response_lower for term in russian_confirmation_terms)
        self.assertTrue(confirmation_russian_found, f"Confirmation should contain Russian elements. Got: {response}")

    async def test_end_to_end_english_still_works(self):
        """Test that English facts still work correctly through the end-to-end flow."""
        # Arrange - English fact message
        english_message = "BOT remember that Einstein likes chocolate"
        
        # Act - Use AiRouter for complete flow
        route, params = await self.ai_router.route_request(english_message)
        
        # Assert - Should work as before
        self.assertEqual(route, "FACT")
        self.assertEqual(params.operation, "remember")
        self.assertEqual(params.user_mention, "Einstein")
        
        # English fact content should be preserved
        self.assertIn("chocolate", params.fact_content.lower())
        self.assertIn("like", params.fact_content.lower())
        
        # Continue with fact handling
        await self.fact_handler.handle_request(params, self.guild_id)
        
        # Memory should be saved
        saved_memory = await self.test_store.get_user_facts(self.guild_id, self.user_id)
        self.assertIsNotNone(saved_memory)
        self.assertIn("chocolate", saved_memory.lower())


if __name__ == '__main__':
    unittest.main()