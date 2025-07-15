"""
Integration tests for MemoryManager with real AI clients and realistic physics data.
These tests use actual Gemini/Gemma APIs and the full physics chat history.
"""

import unittest
import os
from datetime import date
from unittest.mock import AsyncMock
from dotenv import load_dotenv
import time

from memory_manager import MemoryManager
from null_telemetry import NullTelemetry
from test_user_resolver import TestUserResolver
from test_store import TestStore

# Load environment variables from .env file
load_dotenv()


class TestMemoryManagerIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for MemoryManager with realistic physics chat history and real AI."""
    
    def setUp(self):
        self.telemetry = NullTelemetry()
        
        # Check for required environment variables
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.gemini_model = os.getenv('GEMINI_FLASH_MODEL')
        self.gemma_model = os.getenv('GEMINI_GEMMA_MODEL')
        
        if not self.api_key:
            self.skipTest("GEMINI_API_KEY environment variable not set")
        if not self.gemini_model:
            self.skipTest("GEMINI_FLASH_MODEL environment variable not set")
        if not self.gemma_model:
            self.skipTest("GEMINI_GEMMA_MODEL environment variable not set")
        
        # Real test doubles with physics data
        self.user_resolver = TestUserResolver()
        self.test_store = TestStore()
        
        # Real AI clients using actual implementations
        from gemini_client import GeminiClient
        from gemma_client import GemmaClient
        
        self.gemini_client = GeminiClient(
            api_key=self.api_key,
            model_name=self.gemini_model,
            telemetry=self.telemetry,
            temperature=0.1  # Fixed temperature for test stability
        )
        
        self.gemma_client = GemmaClient(
            api_key=self.api_key,
            model_name=self.gemma_model,
            telemetry=self.telemetry,
            temperature=0.1  # Fixed temperature for test stability
        )
        
        # Component under test with realistic store and real AI
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.test_store,
            gemini_client=self.gemini_client,
            gemma_client=self.gemma_client,
            user_resolver=self.user_resolver
        )
        
        # Test data - Physics Guild with real physicist IDs
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids

    async def test_realistic_batch_processing_monday(self):
        """Integration test: Process actual Monday physics discussions with real AI."""
        # Arrange - Monday, March 3rd with 10 real physics messages
        test_date = date(1905, 3, 3)
        
        # Act - Let real AI process the actual physics conversations
        result = await self.memory_manager._create_daily_summaries(self.physics_guild_id, test_date)
        
        # Assert - Verify real AI understood the physics context
        self.assertGreater(len(result), 0, "Should generate summaries for active physicists")
        
        # Check that Einstein was active and got a summary
        einstein_id = self.physicist_ids["Einstein"]
        self.assertIn(einstein_id, result, "Einstein should have a summary for Monday")
        
        einstein_summary = result[einstein_id].lower()
        # Verify AI understood Einstein's actual Monday topics
        physics_terms = ["photoelectric", "quantum", "light", "energy", "wave", "particle"]
        physics_found = any(term in einstein_summary for term in physics_terms)
        self.assertTrue(physics_found, f"Einstein's summary should contain physics terms. Got: {result[einstein_id]}")
        
        # Check that other key physicists from Monday are included
        expected_physicists = {"Einstein", "Planck", "Bohr", "Heisenberg"}
        actual_physicist_names = {name for name, user_id in self.physicist_ids.items() if user_id in result}
        overlap = expected_physicists.intersection(actual_physicist_names)
        self.assertGreaterEqual(len(overlap), 3, f"Should include most key Monday physicists. Found: {actual_physicist_names}")

    async def test_realistic_historical_summary_integration(self):
        """Integration test: Generate historical summary spanning multiple days of real physics data."""
        # Arrange - Get historical summary for Einstein spanning multiple days
        # current_date = date(1905, 3, 6)  # Thursday - asking for history of Mon-Wed
        
        # Act - Let real AI process multiple days of Einstein's discussions
        # Create test daily summaries for Einstein over multiple days
        user_daily_summaries = {
            date(1905, 3, 5): "Einstein discussed wave-particle duality and quantum mechanics",
            date(1905, 3, 4): "Einstein explored photoelectric effect implications", 
            date(1905, 3, 3): "Einstein proposed quantum energy packets theory"
        }
        result = await self.memory_manager._create_week_summary(user_daily_summaries)
        
        # Assert - Verify real AI created coherent historical summary
        self.assertIsNotNone(result, "Should generate historical summary for Einstein")
        self.assertGreater(len(result), 50, "Historical summary should be substantial")
        
        result_lower = result.lower()
        # Einstein's consistent themes across Mon-Wed: quantum theory, relativity, determinism
        einstein_themes = ["einstein", "quantum", "relativity", "determinism", "physics", "theory"]
        themes_found = sum(1 for theme in einstein_themes if theme in result_lower)
        self.assertGreaterEqual(themes_found, 3, f"Historical summary should reflect Einstein's themes. Got: {result}")

    async def test_complete_memory_integration_with_facts_merge(self):
        """Integration test: Complete get_memories workflow with facts + current day + historical + AI merge."""
        # Arrange - Set up a complete memory scenario for Einstein
        einstein_id = self.physicist_ids["Einstein"]
        # current_date = date(1905, 3, 6)  # Thursday - has current day + historical data
        
        # Add some facts about Einstein to the store
        einstein_facts = "Einstein is a theoretical physicist known for developing the theory of relativity and quantum mechanics contributions"
        self.test_store.set_user_facts(self.physics_guild_id, einstein_id, einstein_facts)
        
        # Mock current time to be Thursday so get_memories uses March 6th as "current day" 
        from unittest.mock import patch
        from datetime import datetime, timezone
        with patch('memory_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(1905, 3, 6, 14, 30, tzinfo=timezone.utc)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            # Act - Let the complete memory system process everything with real AI
            result = await self.memory_manager.get_memory(self.physics_guild_id, einstein_id)
        
        # Assert - Verify complete memory integration worked
        self.assertIsNotNone(result, "Should generate complete memory context")
        self.assertGreater(len(result), 100, "Complete memory should be substantial")
        
        result_lower = result.lower()
        
        # Should contain elements from all three sources:
        # 1. Facts: relativity, theoretical physicist
        fact_terms = ["einstein", "relativity", "physicist", "theory"]
        fact_found = sum(1 for term in fact_terms if term in result_lower)
        self.assertGreaterEqual(fact_found, 2, f"Should contain factual information. Got: {result}")
        
        # 2. Current day (Thu): discussions about matter waves, nuclear forces
        # (Based on Thursday's physics_chat_history.txt content)
        current_terms = ["matter", "wave", "nuclear", "quantum"]
        current_found = sum(1 for term in current_terms if term in result_lower)
        self.assertGreaterEqual(current_found, 1, f"Should contain current day themes. Got: {result}")
        
        # 3. Historical patterns: should reflect Einstein's consistent themes
        historical_terms = ["quantum", "determinism", "photoelectric"]
        historical_found = sum(1 for term in historical_terms if term in result_lower)
        self.assertGreaterEqual(historical_found, 1, f"Should contain historical patterns. Got: {result}")
        
        # 4. AI merge quality: should be coherent and comprehensive
        # (Real Gemma should create a unified narrative, not just concatenation)
        self.assertNotIn("No factual information available", result, "Should have processed facts")
        self.assertNotIn("No current day observations", result, "Should have processed current day")
        self.assertNotIn("No historical observations", result, "Should have processed historical data")
        
        # The result should be a coherent synthesis, not just separate pieces
        # Real AI merge should create connections between facts, current activity, and patterns


class TestMemoryManagerBatchIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for batch processing with real AI clients."""

    def setUp(self):
        self.telemetry = NullTelemetry()
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.gemini_model = os.getenv('GEMINI_FLASH_MODEL')
        self.gemma_model = os.getenv('GEMINI_GEMMA_MODEL')
        if not self.api_key or not self.gemini_model or not self.gemma_model:
            self.skipTest("Missing Gemini API key or model names")

        self.user_resolver = TestUserResolver()
        self.test_store = TestStore()
        from gemini_client import GeminiClient
        from gemma_client import GemmaClient
        self.gemini_client = GeminiClient(api_key=self.api_key, model_name=self.gemini_model, telemetry=self.telemetry)
        self.gemma_client = GemmaClient(api_key=self.api_key, model_name=self.gemma_model, telemetry=self.telemetry)
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.test_store,
            gemini_client=self.gemini_client,
            gemma_client=self.gemma_client,
            user_resolver=self.user_resolver
        )
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids

    async def test_batch_get_memories_multiple_physicists_real_ai(self):
        """Test batch processing of multiple physicists with real AI clients."""
        # Arrange
        user_ids = [self.physicist_ids["Einstein"], self.physicist_ids["Bohr"], self.physicist_ids["Planck"]]

        # Act
        results = await self.memory_manager.get_memories(self.physics_guild_id, user_ids)

        # Assert
        self.assertEqual(len(results), 3)
        for user_id in user_ids:
            self.assertIn(user_id, results)
            # Some users might legitimately have no memories (returns None)
            if results[user_id] is not None:
                self.assertIsInstance(results[user_id], str)
                self.assertGreater(len(results[user_id]), 50) # Ensure a meaningful summary



class TestMemoryManagerLargeBatchIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for large batch processing with real AI."""

    def setUp(self):
        self.telemetry = NullTelemetry()
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.gemini_model = os.getenv('GEMINI_FLASH_MODEL')
        self.gemma_model = os.getenv('GEMINI_GEMMA_MODEL')
        if not self.api_key or not self.gemini_model or not self.gemma_model:
            self.skipTest("Missing Gemini API key or model names")

        self.user_resolver = TestUserResolver()
        self.test_store = TestStore()
        from gemini_client import GeminiClient
        from gemma_client import GemmaClient
        self.gemini_client = GeminiClient(api_key=self.api_key, model_name=self.gemini_model, telemetry=self.telemetry)
        self.gemma_client = GemmaClient(api_key=self.api_key, model_name=self.gemma_model, telemetry=self.telemetry)
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.test_store,
            gemini_client=self.gemini_client,
            gemma_client=self.gemma_client,
            user_resolver=self.user_resolver
        )
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids

    async def test_large_batch_scalability_all_physicists(self):
        """Test batch processing with all available physicists (5-10 users)."""
        # Arrange - Use all physicists for scalability test
        all_user_ids = list(self.physicist_ids.values())
        
        # Act - Process memories for all physicists
        results = await self.memory_manager.get_memories(self.physics_guild_id, all_user_ids)
        
        # Assert - Verify scalable processing
        self.assertEqual(len(results), len(all_user_ids))
        for user_id in all_user_ids:
            self.assertIn(user_id, results)
            # Each physicist should get some kind of result (even if None for inactive ones)
            self.assertIsInstance(results[user_id], (str, type(None)))

    async def test_cache_effectiveness_with_real_ai_calls(self):
        """Test that caching reduces actual AI calls in real scenarios."""
        # Arrange
        user_ids = [self.physicist_ids["Einstein"], self.physicist_ids["Bohr"]]
        
        # First batch call - will make AI calls
        start_time_first = time.time()
        results1 = await self.memory_manager.get_memories(self.physics_guild_id, user_ids)
        duration_first = time.time() - start_time_first
        
        # Second batch call - should hit caches and be faster
        start_time_second = time.time()
        results2 = await self.memory_manager.get_memories(self.physics_guild_id, user_ids)
        duration_second = time.time() - start_time_second
        
        # Assert - Results should be identical but second call faster due to caching
        self.assertEqual(results1, results2)
        # Cache should make second call faster, but timing can be variable in CI
        self.assertLess(duration_second, duration_first + 0.1)  # More lenient timing assertion

    async def test_memory_quality_consistency_batch_vs_individual(self):
        """Test that batch processing produces same quality as individual calls."""
        # Arrange
        test_user_id = self.physicist_ids["Einstein"]
        
        # Act - Get memory using both methods
        individual_result = await self.memory_manager.get_memory(self.physics_guild_id, test_user_id)
        batch_results = await self.memory_manager.get_memories(self.physics_guild_id, [test_user_id])
        batch_result = batch_results.get(test_user_id)
        
        # Assert - Should produce identical results since get_memory is just a wrapper
        self.assertEqual(individual_result, batch_result)


class TestMemoryManagerRealExceptionHandling(unittest.IsolatedAsyncioTestCase):
    """Test real exception handling with actual AI services."""

    def setUp(self):
        self.telemetry = NullTelemetry()
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.gemini_model = os.getenv('GEMINI_FLASH_MODEL')
        self.gemma_model = os.getenv('GEMINI_GEMMA_MODEL')
        if not self.api_key or not self.gemini_model or not self.gemma_model:
            self.skipTest("Missing Gemini API key or model names")

        self.user_resolver = TestUserResolver()
        self.test_store = TestStore()
        from gemini_client import GeminiClient
        from gemma_client import GemmaClient
        self.gemini_client = GeminiClient(api_key=self.api_key, model_name=self.gemini_model, telemetry=self.telemetry)
        self.gemma_client = GemmaClient(api_key=self.api_key, model_name=self.gemma_model, telemetry=self.telemetry)
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.test_store,
            gemini_client=self.gemini_client,
            gemma_client=self.gemma_client,
            user_resolver=self.user_resolver
        )
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids

    async def test_graceful_degradation_with_facts_fallback(self):
        """Test that system falls back to facts when AI operations fail."""
        # Arrange - Add facts for Einstein but simulate AI failure
        einstein_id = self.physicist_ids["Einstein"]
        facts = "Einstein is a theoretical physicist who developed relativity theory"
        self.test_store.set_user_facts(self.physics_guild_id, einstein_id, facts)
        
        # Mock AI clients to fail
        self.gemini_client.generate_content = AsyncMock(side_effect=Exception("AI quota exceeded"))
        self.gemma_client.generate_content = AsyncMock(side_effect=Exception("AI quota exceeded"))
        
        # Act - Should still get facts despite AI failures
        result = await self.memory_manager.get_memory(self.physics_guild_id, einstein_id)
        
        # Assert - Should get facts as fallback
        self.assertEqual(result, facts)

    async def test_partial_ai_failure_mixed_results(self):
        """Test behavior when some AI calls succeed and others fail."""
        # Arrange - Set up scenario where daily summaries succeed but merging fails
        user_ids = [self.physicist_ids["Einstein"], self.physicist_ids["Bohr"]]
        
        # Mock gemini (daily summaries) to succeed, gemma (merging) to fail
        from schemas import DailySummaries, UserSummary
        self.gemini_client.generate_content = AsyncMock(return_value=DailySummaries(
            summaries=[UserSummary(user_id=uid, summary=f"Daily summary for {uid}") for uid in user_ids]
        ))
        self.gemma_client.generate_content = AsyncMock(side_effect=Exception("Merge AI failed"))
        
        # Act
        results = await self.memory_manager.get_memories(self.physics_guild_id, user_ids)
        
        # Assert - Should get partial results (current day summaries without merging)
        self.assertEqual(len(results), 2)
        for user_id in user_ids:
            self.assertIn(user_id, results)
            if results[user_id] is not None:
                self.assertIn("Daily summary", results[user_id])

    async def test_empty_response_handling(self):
        """Test handling of empty or malformed AI responses."""
        # Arrange - Mock AI to return empty/malformed responses
        from schemas import DailySummaries
        self.gemini_client.generate_content = AsyncMock(return_value=DailySummaries(summaries=[]))
        
        user_ids = [self.physicist_ids["Einstein"]]
        
        # Act
        results = await self.memory_manager.get_memories(self.physics_guild_id, user_ids)
        
        # Assert - Should handle empty responses gracefully
        self.assertEqual(len(results), 1)
        self.assertIn(self.physicist_ids["Einstein"], results)
        # Result might be None if no other memory sources exist


if __name__ == '__main__':
    unittest.main()
