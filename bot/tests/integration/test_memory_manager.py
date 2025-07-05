"""
Integration tests for MemoryManager with real AI clients and realistic physics data.
These tests use actual Gemini/Gemma APIs and the full physics chat history.
"""

import unittest
import os
from datetime import date
from unittest.mock import Mock
from dotenv import load_dotenv

from memory_manager import MemoryManager
from tests.null_telemetry import NullTelemetry
from tests.test_user_resolver import TestUserResolver
from tests.test_store import TestStore

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
        einstein_id = self.physicist_ids["Einstein"]
        current_date = date(1905, 3, 6)  # Thursday - asking for history of Mon-Wed
        
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
        current_date = date(1905, 3, 6)  # Thursday - has current day + historical data
        
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


if __name__ == '__main__':
    unittest.main()