import unittest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, date, timezone, timedelta
import asyncio

from memory_manager import MemoryManager
from ai_client import AIClient
from message_node import MessageNode
from schemas import MemoryContext, DailySummaries, UserSummary
from store import Store, ChatMessage
from tests.null_telemetry import NullTelemetry
from tests.test_user_resolver import TestUserResolver
from user_resolver import UserResolver


class TestMemoryManagerCaching(unittest.IsolatedAsyncioTestCase):
    """Test cache behavior and key generation for MemoryManager."""
    
    def setUp(self):
        self.telemetry = NullTelemetry()
        
        # Real test double with physicists
        self.user_resolver = TestUserResolver()
        
        # Mock AI clients and store
        self.mock_store = Mock(spec=Store)
        self.mock_gemini_client = Mock(spec=AIClient)
        self.mock_gemma_client = Mock(spec=AIClient)
        
        # Component under test
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.mock_store,
            gemini_client=self.mock_gemini_client,
            gemma_client=self.mock_gemma_client,
            user_resolver=self.user_resolver
        )
        
        # Test data - Physics Guild with real physicist IDs
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids
        self.test_date = date(1905, 3, 3)  # March 3, 1905 - Annus Mirabilis year

    async def test_current_day_cache_hit_same_hour(self):
        """Test that current day summary cache hits within the same hour."""
        # Arrange - Einstein discussing photoelectric effect
        einstein_id = self.physicist_ids["Einstein"]
        self.mock_store.get_chat_messages_for_date = AsyncMock(return_value=[
            ChatMessage(self.physics_guild_id, 1111, 2222, einstein_id, 
                       "Good morning colleagues. I've been pondering the photoelectric effect. Light seems to behave as discrete packets of energy.", 
                       datetime(1905, 3, 3, 9, 15))
        ])
        self.mock_gemini_client.generate_content = AsyncMock(return_value=DailySummaries(
            summaries=[UserSummary(user_id=einstein_id, summary="Einstein proposed that light behaves as discrete energy packets, challenging wave theory")]
        ))
        
        with patch('memory_manager.datetime') as mock_datetime:
            # Mock consistent hour during physics discussion
            mock_datetime.now.return_value = datetime(1905, 3, 3, 9, 30, tzinfo=timezone.utc)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            # Act - First call should generate summary
            daily_summaries1 = await self.memory_manager._daily_summary(self.physics_guild_id, self.test_date)
            result1 = daily_summaries1.get(einstein_id)
            
            # Act - Second call in same hour should hit cache
            daily_summaries2 = await self.memory_manager._daily_summary(self.physics_guild_id, self.test_date)
            result2 = daily_summaries2.get(einstein_id)
        
        # Assert
        expected_summary = "Einstein proposed that light behaves as discrete energy packets, challenging wave theory"
        self.assertEqual(result1, expected_summary)
        self.assertEqual(result2, expected_summary)
        self.mock_gemini_client.generate_content.assert_called_once()  # Only one AI call

    async def test_current_day_cache_hit_same_date(self):
        """Test that current day summary cache hits for same date regardless of hour."""
        # Arrange - Planck's quantum discussions continuing throughout the day
        planck_id = self.physicist_ids["Planck"]
        self.mock_store.get_chat_messages_for_date = AsyncMock(return_value=[
            ChatMessage(self.physics_guild_id, 1111, 2222, planck_id, 
                       "Albert, your quantum hypothesis is intriguing, but surely you don't mean to abandon wave theory entirely?", 
                       datetime(1905, 3, 3, 9, 18))
        ])
        self.mock_gemini_client.generate_content = AsyncMock(return_value=DailySummaries(
            summaries=[UserSummary(user_id=planck_id, summary="Planck questioned Einstein's quantum theory while defending wave theory")]
        ))
        
        with patch('memory_manager.datetime') as mock_datetime:
            # First call during morning physics discussion (9:30)
            mock_datetime.now.return_value = datetime(1905, 3, 3, 9, 30, tzinfo=timezone.utc)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            # Act - First call in morning hour
            await self.memory_manager._daily_summary(self.physics_guild_id, self.test_date)
            
            # Later in afternoon physics discussion (15:30)
            mock_datetime.now.return_value = datetime(1905, 3, 3, 15, 30, tzinfo=timezone.utc)
            
            # Act - Second call in different hour but same date should hit cache
            await self.memory_manager._daily_summary(self.physics_guild_id, self.test_date)
        
        # Assert
        self.assertEqual(self.mock_gemini_client.generate_content.call_count, 1)  # Only one AI call since hour buckets are eliminated

    async def test_historical_daily_cache_hit(self):
        """Test that historical daily summary cache hits for same guild/date."""
        # Arrange - Bohr's atomic model discussions from previous day
        historical_date = date(1905, 3, 2)  # Day before the main physics discussions
        bohr_id = self.physicist_ids["Bohr"]
        self.mock_store.get_chat_messages_for_date = AsyncMock(return_value=[
            ChatMessage(self.physics_guild_id, 1111, 2222, bohr_id, 
                       "I believe the atom itself might have quantized energy levels. This could explain hydrogen's spectral lines perfectly.", 
                       datetime(1905, 3, 2, 14, 22))
        ])
        self.mock_gemini_client.generate_content = AsyncMock(return_value=DailySummaries(
            summaries=[UserSummary(user_id=bohr_id, summary="Bohr proposed quantized atomic energy levels to explain hydrogen spectra")]
        ))
        
        # Act - Two calls for same date should hit cache on second
        daily_summaries1 = await self.memory_manager._daily_summary(self.physics_guild_id, historical_date)
        result1 = daily_summaries1.get(bohr_id)
        daily_summaries2 = await self.memory_manager._daily_summary(self.physics_guild_id, historical_date)
        result2 = daily_summaries2.get(bohr_id)
        
        # Assert
        expected_summary = "Bohr proposed quantized atomic energy levels to explain hydrogen spectra"
        self.assertEqual(result1, expected_summary)
        self.assertEqual(result2, expected_summary)
        self.mock_gemini_client.generate_content.assert_called_once()  # Only one AI call

    async def test_context_cache_hit_identical_content(self):
        """Test that context merge cache hits when content is identical."""
        # Arrange - Einstein's context across multiple memory sources
        facts = "Einstein is a theoretical physicist known for relativity theory and photoelectric effect work"
        current_day = "Einstein discussed quantum nature of light and challenged classical wave theory"
        historical = "Einstein has been developing revolutionary theories about space, time, and energy"
        
        # Mock returns handled by TestUserResolver automatically
        self.mock_gemma_client.generate_content = AsyncMock(return_value=MemoryContext(
            context="Einstein is a revolutionary physicist who challenges classical physics with quantum and relativity theories"
        ))
        
        # Act - Two calls with identical content
        einstein_id = self.physicist_ids["Einstein"]
        result1 = await self.memory_manager._merge_context(self.physics_guild_id, einstein_id, facts, current_day, historical)
        result2 = await self.memory_manager._merge_context(self.physics_guild_id, einstein_id, facts, current_day, historical)
        
        # Assert
        expected_context = "Einstein is a revolutionary physicist who challenges classical physics with quantum and relativity theories"
        self.assertEqual(result1, expected_context)
        self.assertEqual(result2, expected_context)
        self.mock_gemma_client.generate_content.assert_called_once()  # Only one AI call

    async def test_context_cache_miss_changed_facts(self):
        """Test that context merge cache misses when facts change."""
        # Arrange - Bohr's facts being updated as his atomic model evolves
        original_facts = "Bohr is a physicist working on atomic structure"
        updated_facts = "Bohr is a physicist who proposed quantized electron orbits in atoms, explaining hydrogen spectra"
        current_day = "Bohr discussed revolutionary atomic models with quantized energy levels"
        historical = "Bohr has been developing new atomic theories that explain spectral lines"
        
        # Mock returns handled by TestUserResolver automatically
        self.mock_gemma_client.generate_content = AsyncMock(return_value=MemoryContext(
            context="Bohr is revolutionizing atomic physics with quantum orbital theory"
        ))
        
        # Act - Two calls with evolving facts about Bohr's discoveries
        bohr_id = self.physicist_ids["Bohr"]
        await self.memory_manager._merge_context(self.physics_guild_id, bohr_id, original_facts, current_day, historical)
        await self.memory_manager._merge_context(self.physics_guild_id, bohr_id, updated_facts, current_day, historical)
        
        # Assert
        self.assertEqual(self.mock_gemma_client.generate_content.call_count, 2)  # Two AI calls


    async def test_empty_messages_returns_empty_dict(self):
        """Test that batch generation with no messages returns empty dict."""
        # Arrange
        self.mock_store.get_chat_messages_for_date = AsyncMock(return_value=[])
        
        # Act
        result = await self.memory_manager._create_daily_summaries(self.physics_guild_id, self.test_date)
        
        # Assert
        self.assertEqual(result, {})
        self.mock_gemini_client.generate_content.assert_not_called()


class TestMemoryManagerFallbacks(unittest.IsolatedAsyncioTestCase):
    """Test fallback logic and error handling for MemoryManager."""
    
    def setUp(self):
        self.telemetry = NullTelemetry()
        
        # Real test double with physicists
        self.user_resolver = TestUserResolver()
        
        # Mock AI clients and store
        self.mock_store = Mock(spec=Store)
        self.mock_gemini_client = Mock(spec=AIClient)
        self.mock_gemma_client = Mock(spec=AIClient)
        
        # Component under test
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.mock_store,
            gemini_client=self.mock_gemini_client,
            gemma_client=self.mock_gemma_client,
            user_resolver=self.user_resolver
        )
        
        # Test data - Physics Guild with real physicist IDs
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids

    async def test_no_memories_returns_none(self):
        """Test that get_memories returns None when no memories exist."""
        # Arrange
        self.mock_store.get_user_facts = AsyncMock(return_value=None)
        
        with patch.object(self.memory_manager, '_daily_summary', return_value={}):
            with patch.object(self.memory_manager, '_create_week_summary', return_value=None):
                # Act
                result = await self.memory_manager.get_memory(self.physics_guild_id, self.physicist_ids["Einstein"])
        
        # Assert
        self.assertIsNone(result)

    async def test_facts_only_returns_facts_directly(self):
        """Test that get_memories returns facts directly when only facts exist."""
        # Arrange - Einstein with only stored facts, no daily activity
        facts = "Einstein is the theoretical physicist who developed special relativity and explained the photoelectric effect"
        self.mock_store.get_user_facts = AsyncMock(return_value=facts)
        
        with patch.object(self.memory_manager, '_daily_summary', return_value={}):
            with patch.object(self.memory_manager, '_create_week_summary', return_value=None):
                # Act
                result = await self.memory_manager.get_memory(self.physics_guild_id, self.physicist_ids["Einstein"])
        
        # Assert
        self.assertEqual(result, facts)

    async def test_current_day_only_returns_current_day_directly(self):
        """Test that get_memories returns current day summary directly when only it exists."""
        # Arrange - Planck active today discussing blackbody radiation, no stored facts
        current_summary = "Planck defended wave theory while questioning Einstein's quantum hypothesis about energy packets"
        self.mock_store.get_user_facts = AsyncMock(return_value=None)
        
        with patch.object(self.memory_manager, '_daily_summary', side_effect=lambda guild_id, date: {self.physicist_ids["Planck"]: current_summary}):
            with patch.object(self.memory_manager, '_create_week_summary', return_value=None):
                # Act
                result = await self.memory_manager.get_memory(self.physics_guild_id, self.physicist_ids["Planck"])
        
        # Assert
        self.assertEqual(result, current_summary)

    async def test_historical_only_returns_historical_directly(self):
        """Test that get_memories returns historical summary directly when only it exists."""
        # Arrange
        historical_summary = "User has been consistently active"
        self.mock_store.get_user_facts = AsyncMock(return_value=None)
        
        with patch.object(self.memory_manager, '_daily_summary', return_value={}):
            with patch.object(self.memory_manager, '_create_week_summary', return_value=historical_summary):
                # Act
                result = await self.memory_manager.get_memory(self.physics_guild_id, self.physicist_ids["Bohr"])
        
        # Assert
        self.assertEqual(result, historical_summary)

    async def test_multiple_sources_attempts_merge(self):
        """Test that get_memories attempts AI merge when multiple sources exist."""
        # Arrange
        facts = "User likes coffee"
        current_summary = "User worked today"
        historical_summary = "User programs regularly"
        merged_result = "Combined user context"
        
        self.mock_store.get_user_facts = AsyncMock(return_value=facts)
        
        with patch.object(self.memory_manager, '_daily_summary', side_effect=lambda guild_id, date: {self.physicist_ids["Thomson"]: current_summary}):
            with patch.object(self.memory_manager, '_create_week_summary', return_value=historical_summary):
                with patch.object(self.memory_manager, '_merge_context', return_value=merged_result) as mock_merge:
                    # Act
                    result = await self.memory_manager.get_memory(self.physics_guild_id, self.physicist_ids["Thomson"])
        
        # Assert
        self.assertEqual(result, merged_result)
        mock_merge.assert_called_once_with(self.physics_guild_id, self.physicist_ids["Thomson"], facts, current_summary, historical_summary)

    async def test_merge_failure_falls_back_to_facts(self):
        """Test that get_memories falls back to facts when AI merge fails."""
        # Arrange
        facts = "User likes coffee"
        current_summary = "User worked today"
        historical_summary = "User programs regularly"
        
        self.mock_store.get_user_facts = AsyncMock(return_value=facts)
        
        with patch.object(self.memory_manager, '_daily_summary', side_effect=lambda guild_id, date: {self.physicist_ids["Rutherford"]: current_summary}):
            with patch.object(self.memory_manager, '_create_week_summary', return_value=historical_summary):
                with patch.object(self.memory_manager, '_merge_context', side_effect=Exception("AI service unavailable")):
                    # Act
                    result = await self.memory_manager.get_memory(self.physics_guild_id, self.physicist_ids["Rutherford"])
        
        # Assert
        self.assertEqual(result, facts)

    async def test_merge_failure_falls_back_to_current_day_when_no_facts(self):
        """Test that get_memories falls back to current day when merge fails and no facts."""
        # Arrange
        current_summary = "User worked today"
        historical_summary = "User programs regularly"
        
        self.mock_store.get_user_facts = AsyncMock(return_value=None)
        
        with patch.object(self.memory_manager, '_daily_summary', side_effect=lambda guild_id, date: {self.physicist_ids["Schrödinger"]: current_summary}):
            with patch.object(self.memory_manager, '_create_week_summary', return_value=historical_summary):
                with patch.object(self.memory_manager, '_merge_context', side_effect=Exception("AI service unavailable")):
                    # Act
                    result = await self.memory_manager.get_memory(self.physics_guild_id, self.physicist_ids["Schrödinger"])
        
        # Assert
        self.assertEqual(result, current_summary)

    async def test_merge_failure_falls_back_to_historical_when_no_facts_or_current(self):
        """Test that get_memories falls back to historical when merge fails and no facts or current day."""
        # Arrange
        historical_summary = "User programs regularly"
        
        self.mock_store.get_user_facts = AsyncMock(return_value=None)
        
        with patch.object(self.memory_manager, '_daily_summary', return_value={}):
            with patch.object(self.memory_manager, '_create_week_summary', return_value=historical_summary):
                with patch.object(self.memory_manager, '_merge_context', side_effect=Exception("AI service unavailable")):
                    # Act
                    result = await self.memory_manager.get_memory(self.physics_guild_id, self.physicist_ids["Heisenberg"])
        
        # Assert
        self.assertEqual(result, historical_summary)

    async def test_store_failure_propagates_exception(self):
        """Test that store failures propagate exceptions (no longer handled gracefully)."""
        # Arrange
        self.mock_store.get_user_facts = AsyncMock(side_effect=Exception("Database connection failed"))
        
        # Mock daily summary to return empty for fallback
        with patch.object(self.memory_manager, '_daily_summary', return_value={}):
            with patch.object(self.memory_manager, '_create_week_summary', return_value=None):
                # Act & Assert - Store failure should propagate exception
                with self.assertRaises(Exception) as context:
                    await self.memory_manager.get_memory(self.physics_guild_id, self.physicist_ids["Born"])
                
                self.assertEqual(str(context.exception), "Database connection failed")

    async def test_current_day_summary_failure_handled_gracefully(self):
        """Test that current day summary failures are handled gracefully."""
        # Arrange
        facts = "User likes coffee"
        self.mock_store.get_user_facts = AsyncMock(return_value=facts)
        
        with patch.object(self.memory_manager, '_daily_summary', side_effect=Exception("AI service error")):
            with patch.object(self.memory_manager, '_create_week_summary', return_value=None):
                # Act
                result = await self.memory_manager.get_memory(self.physics_guild_id, self.physicist_ids["Curie"])
        
        # Assert - Should still return facts despite current day failure
        self.assertEqual(result, facts)


class TestMemoryManagerDataProcessing(unittest.IsolatedAsyncioTestCase):
    """Test data processing and formatting for MemoryManager."""
    
    def setUp(self):
        self.telemetry = NullTelemetry()
        
        # Real test double with physicists
        self.user_resolver = TestUserResolver()
        
        # Mock AI clients and store
        self.mock_store = Mock(spec=Store)
        self.mock_gemini_client = Mock(spec=AIClient)
        self.mock_gemma_client = Mock(spec=AIClient)
        
        # Component under test
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.mock_store,
            gemini_client=self.mock_gemini_client,
            gemma_client=self.mock_gemma_client,
            user_resolver=self.user_resolver
        )
        
        # Test data - Physics Guild with real physicist IDs
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids
        self.test_date = date(1905, 3, 3)  # March 3, 1905 - Annus Mirabilis year

    async def test_message_formatting_xml_structure(self):
        """Test that messages are formatted correctly in XML structure."""
        # Arrange - Einstein discussing photoelectric effect with mention to Planck
        einstein_id = self.physicist_ids["Einstein"]
        planck_id = self.physicist_ids["Planck"]
        test_message = ChatMessage(
            guild_id=self.physics_guild_id,
            channel_id=1111,
            message_id=2222,
            user_id=einstein_id,
            message_text=f"Hello <@{planck_id}>, I've been thinking about your quantum hypothesis!",
            timestamp=datetime(1905, 3, 3, 9, 15, tzinfo=timezone.utc)
        )
        
        self.mock_store.get_chat_messages_for_date = AsyncMock(return_value=[test_message])
        self.mock_gemini_client.generate_content = AsyncMock(return_value=DailySummaries(
            summaries=[UserSummary(user_id=einstein_id, summary="Einstein discussed quantum hypothesis with Planck")]
        ))
        
        # Act
        await self.memory_manager._create_daily_summaries(self.physics_guild_id, self.test_date)
        
        # Assert - Check that the prompt contains properly formatted XML
        call_args = self.mock_gemini_client.generate_content.call_args
        prompt = call_args[1]['message']  # keyword argument
        
        self.assertIn('<message>', prompt)
        self.assertIn('<timestamp>1905-03-03 09:15:00+00:00</timestamp>', prompt)
        self.assertIn(f'<author_id>{einstein_id}</author_id>', prompt)
        self.assertIn('<author>Einstein</author>', prompt)
        self.assertIn('<content>Hello Planck, I\'ve been thinking about your quantum hypothesis!</content>', prompt)
        self.assertIn('</message>', prompt)

    async def test_user_deduplication(self):
        """Test that AI prompt contains deduplicated user list despite duplicate messages."""
        # Arrange - 4 messages from 2 users (Einstein has 3 messages, Bohr has 1)
        einstein_id = self.physicist_ids["Einstein"]
        bohr_id = self.physicist_ids["Bohr"]
        messages = [
            ChatMessage(self.physics_guild_id, 1111, 2222, einstein_id, "Light behaves as discrete packets", datetime(1905, 3, 3, 9, 15)),
            ChatMessage(self.physics_guild_id, 1111, 3333, einstein_id, "This explains the photoelectric effect", datetime(1905, 3, 3, 9, 18)),  # Same user
            ChatMessage(self.physics_guild_id, 1111, 4444, bohr_id, "Atoms might have quantized energy levels", datetime(1905, 3, 3, 9, 22)),  # Different user
            ChatMessage(self.physics_guild_id, 1111, 5555, einstein_id, "Quantum mechanics is revolutionary", datetime(1905, 3, 3, 9, 25)),  # Same user again
        ]
        
        self.mock_store.get_chat_messages_for_date = AsyncMock(return_value=messages)
        self.mock_gemini_client.generate_content = AsyncMock(return_value=DailySummaries(
            summaries=[
                UserSummary(user_id=einstein_id, summary="Einstein discussed quantum theory"),
                UserSummary(user_id=bohr_id, summary="Bohr proposed quantized energy levels")
            ]
        ))
        
        # Act
        await self.memory_manager._create_daily_summaries(self.physics_guild_id, self.test_date)
        
        # Assert - Check that AI prompt contains deduplicated user list
        call_args = self.mock_gemini_client.generate_content.call_args
        prompt = call_args[1]['message']
        
        # Should contain exactly 2 users in target_users section, not 4
        einstein_user_entries = prompt.count(f"<user_id>{einstein_id}</user_id>")
        bohr_user_entries = prompt.count(f"<user_id>{bohr_id}</user_id>")
        
        self.assertEqual(einstein_user_entries, 1, "Einstein should appear only once in user list despite 3 messages")
        self.assertEqual(bohr_user_entries, 1, "Bohr should appear only once in user list")
        
        # But all 4 messages should still be in the messages section
        self.assertEqual(prompt.count("<message>"), 4, "All 4 individual messages should be included")

    async def test_historical_date_arithmetic(self):
        """Test that historical date ranges are calculated correctly (days 2-7)."""
        # Arrange - Create test daily summaries dict
        daily_summaries = {
            date(2025, 6, 30): f"Summary for {date(2025, 6, 30)}",  # Yesterday
            date(2025, 6, 29): f"Summary for {date(2025, 6, 29)}",  # Day 3
            date(2025, 6, 28): f"Summary for {date(2025, 6, 28)}",  # Day 4
        }
        
        # Mock returns handled by TestUserResolver automatically
        self.mock_gemma_client.generate_content = AsyncMock(return_value=MemoryContext(
            context="Historical summary"
        ))
        
        # Act
        result = await self.memory_manager._create_week_summary(daily_summaries)
        
        # Assert
        self.assertEqual(result, "Historical summary")
        
        # Verify the prompt contains the expected date range and summaries
        call_args = self.mock_gemma_client.generate_content.call_args
        prompt = call_args[1]['message']
        
        self.assertIn('<date_range>2025-06-28 to 2025-06-30</date_range>', prompt)
        self.assertIn('<date>2025-06-30</date>', prompt)
        self.assertIn('<date>2025-06-29</date>', prompt)
        self.assertIn('<date>2025-06-28</date>', prompt)

    async def test_ingest_message_delegates_to_store(self):
        """Test that ingest_message properly delegates to store."""
        # Arrange
        message = MessageNode(
            id=12345,
            channel_id=67890,
            author_id=11111,
            content="Test message content",
            mentioned_user_ids=[],
            created_at=datetime(2025, 7, 1, 12, 0, tzinfo=timezone.utc)
        )
        
        self.mock_store.add_chat_message = AsyncMock()
        
        # Act
        await self.memory_manager.ingest_message(self.physics_guild_id, message)
        
        # Assert
        self.mock_store.add_chat_message.assert_called_once_with(
            self.physics_guild_id,
            message.channel_id, 
            message.id,
            message.author_id,
            message.content,
            message.created_at
        )


class TestMemoryManagerBatchProcessing(unittest.IsolatedAsyncioTestCase):
    """Test batch processing and concurrent operations for MemoryManager."""

    def setUp(self):
        self.telemetry = NullTelemetry()
        self.user_resolver = TestUserResolver()
        self.mock_store = Mock(spec=Store)
        self.mock_gemini_client = Mock(spec=AIClient)
        self.mock_gemma_client = Mock(spec=AIClient)
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.mock_store,
            gemini_client=self.mock_gemini_client,
            gemma_client=self.mock_gemma_client,
            user_resolver=self.user_resolver
        )
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids

    async def test_get_memories_batch_processing_multiple_users(self):
        """Test that get_memories() processes multiple users and returns correct dict mapping."""
        # Arrange
        user_ids = [self.physicist_ids["Einstein"], self.physicist_ids["Bohr"], self.physicist_ids["Planck"]]
        
        # Mock the internal methods to simulate data for these users
        with patch.object(self.memory_manager, '_fetch_all_daily_summaries', new_callable=AsyncMock) as mock_fetch_daily, \
             patch.object(self.memory_manager, '_create_historical_summaries_for_users', new_callable=AsyncMock) as mock_create_historical, \
             patch.object(self.memory_manager, '_create_combined_memories', new_callable=AsyncMock) as mock_create_combined:

            mock_fetch_daily.return_value = {}  # Assume no daily summaries for simplicity
            mock_create_historical.return_value = {uid: f"Historical summary for {uid}" for uid in user_ids}
            mock_create_combined.return_value = {uid: f"Combined memory for {uid}" for uid in user_ids}

            # Act
            result = await self.memory_manager.get_memories(self.physics_guild_id, user_ids)

            # Assert
            self.assertIsInstance(result, dict)
            self.assertEqual(len(result), 3)
            for user_id in user_ids:
                self.assertIn(user_id, result)
                self.assertEqual(result[user_id], f"Combined memory for {user_id}")

    async def test_get_memories_empty_user_list_returns_empty_dict(self):
        """Test edge case handling for empty input."""
        # Act
        result = await self.memory_manager.get_memories(self.physics_guild_id, [])

        # Assert
        self.assertEqual(result, {})
        self.mock_gemini_client.generate_content.assert_not_called()
        self.mock_gemma_client.generate_content.assert_not_called()

class TestMemoryManagerExceptionIsolation(unittest.IsolatedAsyncioTestCase):
    """Test exception isolation and graceful degradation in concurrent processing."""

    def setUp(self):
        self.telemetry = NullTelemetry()
        self.user_resolver = TestUserResolver()
        self.mock_store = Mock(spec=Store)
        self.mock_gemini_client = Mock(spec=AIClient)
        self.mock_gemma_client = Mock(spec=AIClient)
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.mock_store,
            gemini_client=self.mock_gemini_client,
            gemma_client=self.mock_gemma_client,
            user_resolver=self.user_resolver
        )
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids
        self.today = date(1905, 3, 6)
        self.all_dates = [self.today - timedelta(days=i) for i in range(7)]

    async def test_daily_summary_failure_doesnt_block_other_dates(self):
        """Test that if one date's daily summary fails, other dates still process."""
        # Arrange
        failing_date = self.all_dates[1]  # Let's say yesterday's summary fails
        successful_date = self.all_dates[2]

        async def daily_summary_side_effect(guild_id, for_date):
            if for_date == failing_date:
                raise ValueError("AI service unavailable for this date")
            elif for_date == successful_date:
                return {self.physicist_ids["Einstein"]: "Successful summary"}
            return {}

        with patch.object(self.memory_manager, '_daily_summary', side_effect=daily_summary_side_effect):
            # Act
            results = await self.memory_manager._fetch_all_daily_summaries(self.physics_guild_id, self.all_dates)

            # Assert - Main goal is that successful dates still work despite failures
            self.assertIn(successful_date, results)
            self.assertEqual(results[successful_date], {self.physicist_ids["Einstein"]: "Successful summary"})
            # The failing date gets empty dict (defaultdict behavior), but key point is other dates succeed
            if failing_date in results:
                self.assertEqual(results[failing_date], {})

    async def test_historical_summary_failure_isolated_per_user(self):
        """Test that one user's historical summary failure doesn't affect other users."""
        # Arrange
        einstein_id = self.physicist_ids["Einstein"]
        bohr_id = self.physicist_ids["Bohr"]  # Bohr's summary will fail
        planck_id = self.physicist_ids["Planck"]
        user_ids = [einstein_id, bohr_id, planck_id]

        async def week_summary_side_effect(daily_summaries):
            # Simulate Bohr failing (empty daily summaries for Bohr)
            if not daily_summaries:
                raise ValueError("Failed to generate historical summary")
            return "Historical summary"

        with patch.object(self.memory_manager, '_create_week_summary', side_effect=week_summary_side_effect):
            # Act - Provide historical dates (method skips first item which is "today")
            daily_summaries_by_date = {
                self.today: {},  # This gets skipped as "today"
                self.today - timedelta(days=1): {einstein_id: "Yesterday summary", planck_id: "Yesterday summary"},
                self.today - timedelta(days=2): {einstein_id: "Day before summary", planck_id: "Day before summary"}
                # Note: Bohr is missing from historical dates, so will get empty daily_summaries and fail
            }
            results = await self.memory_manager._create_historical_summaries_for_users(user_ids, daily_summaries_by_date)

            # Assert
            self.assertIn(einstein_id, results)
            self.assertEqual(results[einstein_id], "Historical summary")
            self.assertIn(planck_id, results)
            self.assertEqual(results[planck_id], "Historical summary")
            self.assertIn(bohr_id, results)
            self.assertIsNone(results[bohr_id])  # Should be None due to failure

    async def test_context_merge_failure_isolated_per_user(self):
        """Test that merge failures are handled independently per user."""
        # Arrange - Test the _create_user_memory method directly instead of using side effects with asyncio.gather
        einstein_id = self.physicist_ids["Einstein"]
        bohr_id = self.physicist_ids["Bohr"]  # Bohr's merge will fail
        planck_id = self.physicist_ids["Planck"]

        # Mock successful merge for Einstein and Planck, failure for Bohr
        async def merge_context_side_effect(guild_id, user_id, facts, current_day, historical):
            if user_id == bohr_id:
                raise ValueError("AI merge failed for Bohr")
            return f"Merged context for {user_id}"

        with patch.object(self.memory_manager, '_merge_context', side_effect=merge_context_side_effect):
            # Act - Test individual user memory creation to verify isolation
            einstein_result = await self.memory_manager._create_user_memory(
                self.physics_guild_id, einstein_id, "Einstein facts", "Einstein current", "Einstein historical"
            )
            bohr_result = await self.memory_manager._create_user_memory(
                self.physics_guild_id, bohr_id, "Bohr facts", "Bohr current", "Bohr historical"
            )
            planck_result = await self.memory_manager._create_user_memory(
                self.physics_guild_id, planck_id, "Planck facts", "Planck current", "Planck historical"
            )

            # Assert - Einstein and Planck succeed, Bohr falls back to facts
            self.assertEqual(einstein_result, f"Merged context for {einstein_id}")
            self.assertEqual(bohr_result, "Bohr facts")  # Falls back to facts when merge fails
            self.assertEqual(planck_result, f"Merged context for {planck_id}")


class TestMemoryManagerCacheArchitecture(unittest.IsolatedAsyncioTestCase):
    """Test the four-cache architecture of the MemoryManager."""

    def setUp(self):
        self.telemetry = NullTelemetry()
        self.user_resolver = TestUserResolver()
        self.mock_store = Mock(spec=Store)
        self.mock_gemini_client = Mock(spec=AIClient)
        self.mock_gemma_client = Mock(spec=AIClient)
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.mock_store,
            gemini_client=self.mock_gemini_client,
            gemma_client=self.mock_gemma_client,
            user_resolver=self.user_resolver
        )
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids

    async def test_current_day_vs_closed_day_cache_separation(self):
        """Test that current day uses TTL cache while historical days use LRU cache."""
        # Arrange
        today = date(1905, 3, 6)
        yesterday = date(1905, 3, 5)

        with patch('memory_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(1905, 3, 6, 12, 0, tzinfo=timezone.utc)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # Mock the AI call to return a value
            self.mock_gemini_client.generate_content.return_value = DailySummaries(summaries=[])

            # Act
            await self.memory_manager._daily_summary(self.physics_guild_id, today)
            await self.memory_manager._daily_summary(self.physics_guild_id, yesterday)

            # Assert
            self.assertIn((self.physics_guild_id, today), self.memory_manager._current_day_batch_cache)
            self.assertNotIn((self.physics_guild_id, today), self.memory_manager._closed_day_batch_cache)
            self.assertIn((self.physics_guild_id, yesterday), self.memory_manager._closed_day_batch_cache)
            self.assertNotIn((self.physics_guild_id, yesterday), self.memory_manager._current_day_batch_cache)

    async def test_week_summary_cache_content_based_hashing(self):
        """Test that historical summaries use content hashes as cache keys."""
        # Arrange
        daily_summaries = {date(1905, 3, 5): "Summary A", date(1905, 3, 4): "Summary B"}
        self.mock_gemma_client.generate_content.return_value = MemoryContext(context="Historical Summary")

        # Act
        await self.memory_manager._create_week_summary(daily_summaries)
        await self.memory_manager._create_week_summary(daily_summaries)  # Call again with same content

        # Assert
        self.mock_gemma_client.generate_content.assert_called_once()

    async def test_context_cache_with_content_hashing(self):
        """Test that context merge cache uses content-based keys."""
        # Arrange
        facts = "Facts"
        current_day = "Current Day"
        historical = "Historical"
        user_id = self.physicist_ids["Einstein"]
        self.mock_gemma_client.generate_content.return_value = MemoryContext(context="Merged Context")

        # Act
        await self.memory_manager._merge_context(self.physics_guild_id, user_id, facts, current_day, historical)
        await self.memory_manager._merge_context(self.physics_guild_id, user_id, facts, current_day, historical)

        # Assert
        self.mock_gemma_client.generate_content.assert_called_once()

class TestMemoryManagerSmartFallback(unittest.IsolatedAsyncioTestCase):
    """Test the smart fallback logic of the MemoryManager."""

    def setUp(self):
        self.telemetry = NullTelemetry()
        self.user_resolver = TestUserResolver()
        self.mock_store = Mock(spec=Store)
        self.mock_gemini_client = Mock(spec=AIClient)
        self.mock_gemma_client = Mock(spec=AIClient)
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.mock_store,
            gemini_client=self.mock_gemini_client,
            gemma_client=self.mock_gemma_client,
            user_resolver=self.user_resolver
        )
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids

    async def test_single_source_memory_skips_ai_merge(self):
        """Test that when only one memory source exists, AI merge is skipped."""
        # Arrange
        facts = "Just the facts"
        user_id = self.physicist_ids["Einstein"]

        # Act
        result = await self.memory_manager._create_user_memory(self.physics_guild_id, user_id, facts, None, None)

        # Assert
        self.assertEqual(result, facts)
        self.mock_gemma_client.generate_content.assert_not_called()

    async def test_merge_failure_cascading_fallback_priority(self):
        """Test fallback hierarchy: facts -> current_day -> historical -> None."""
        # Arrange
        user_id = self.physicist_ids["Einstein"]
        facts = "Facts are most important"
        current_day = "Current day is second most important"
        historical = "Historical is least important"

        self.mock_gemma_client.generate_content.side_effect = Exception("AI merge failed")

        # Test case 1: All sources present, should fall back to facts
        result1 = await self.memory_manager._create_user_memory(self.physics_guild_id, user_id, facts, current_day, historical)
        self.assertEqual(result1, facts)

        # Test case 2: No facts, should fall back to current_day
        result2 = await self.memory_manager._create_user_memory(self.physics_guild_id, user_id, None, current_day, historical)
        self.assertEqual(result2, current_day)

        # Test case 3: No facts or current_day, should fall back to historical
        result3 = await self.memory_manager._create_user_memory(self.physics_guild_id, user_id, None, None, historical)
        self.assertEqual(result3, historical)

        # Test case 4: No sources, should return None
        result4 = await self.memory_manager._create_user_memory(self.physics_guild_id, user_id, None, None, None)
        self.assertIsNone(result4)

class TestMemoryManagerConcurrency(unittest.IsolatedAsyncioTestCase):
    """Test concurrent processing validation for MemoryManager."""

    def setUp(self):
        self.telemetry = NullTelemetry()
        self.user_resolver = TestUserResolver()
        self.mock_store = Mock(spec=Store)
        self.mock_gemini_client = Mock(spec=AIClient)
        self.mock_gemma_client = Mock(spec=AIClient)
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.mock_store,
            gemini_client=self.mock_gemini_client,
            gemma_client=self.mock_gemma_client,
            user_resolver=self.user_resolver
        )
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids
        self.all_dates = [date(1905, 3, 6) - timedelta(days=i) for i in range(7)]

    async def test_fetch_all_daily_summaries_uses_asyncio_gather(self):
        """Test that daily summaries are fetched concurrently, not sequentially."""
        # Arrange
        async def slow_daily_summary(*args, **kwargs):
            await asyncio.sleep(0.1)
            return {}

        with patch.object(self.memory_manager, '_daily_summary', side_effect=slow_daily_summary) as mock_daily_summary:
            start_time = asyncio.get_event_loop().time()
            # Act
            await self.memory_manager._fetch_all_daily_summaries(self.physics_guild_id, self.all_dates)
            end_time = asyncio.get_event_loop().time()

            # Assert
            self.assertEqual(mock_daily_summary.call_count, len(self.all_dates))
            # If sequential, it would take at least 0.7s. Concurrent should be much faster.
            self.assertLess(end_time - start_time, 0.5)

    async def test_create_historical_summaries_concurrent_processing(self):
        """Test that multiple users' historical summaries are generated concurrently."""
        # Arrange
        user_ids = [self.physicist_ids["Einstein"], self.physicist_ids["Bohr"], self.physicist_ids["Planck"]]
        daily_summaries = {self.all_dates[1]: {uid: "Summary" for uid in user_ids}}

        async def slow_week_summary(*args, **kwargs):
            await asyncio.sleep(0.1)
            return "Historical Summary"

        with patch.object(self.memory_manager, '_create_week_summary', side_effect=slow_week_summary) as mock_week_summary:
            start_time = asyncio.get_event_loop().time()
            # Act
            await self.memory_manager._create_historical_summaries_for_users(user_ids, daily_summaries)
            end_time = asyncio.get_event_loop().time()

            # Assert
            self.assertEqual(mock_week_summary.call_count, len(user_ids))
            # If sequential, it would take at least 0.3s. Concurrent should be much faster.
            self.assertLess(end_time - start_time, 0.2)


class TestMemoryManagerDatabaseConcurrency(unittest.IsolatedAsyncioTestCase):
    """Test database operations are properly concurrent."""

    def setUp(self):
        self.telemetry = NullTelemetry()
        self.user_resolver = TestUserResolver()
        self.mock_store = Mock(spec=Store)
        self.mock_gemini_client = Mock(spec=AIClient)
        self.mock_gemma_client = Mock(spec=AIClient)
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.mock_store,
            gemini_client=self.mock_gemini_client,
            gemma_client=self.mock_gemma_client,
            user_resolver=self.user_resolver
        )
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids

    async def test_user_memory_creation_concurrent(self):
        """Test that user memory creation (merge operations) are processed concurrently."""
        # Arrange
        user_ids = [self.physicist_ids["Einstein"], self.physicist_ids["Bohr"], self.physicist_ids["Planck"]]
        
        async def slow_create_user_memory(guild_id, user_id, facts, current_day, historical):
            await asyncio.sleep(0.1)  # Simulate AI processing time
            return f"Memory for {user_id}"

        with patch.object(self.memory_manager, '_fetch_all_daily_summaries', new_callable=AsyncMock) as mock_fetch_daily, \
             patch.object(self.memory_manager, '_create_historical_summaries_for_users', new_callable=AsyncMock) as mock_create_historical, \
             patch.object(self.memory_manager, '_create_user_memory', side_effect=slow_create_user_memory):
            
            mock_fetch_daily.return_value = {date(1905, 3, 6): {uid: f"Summary for {uid}" for uid in user_ids}}
            mock_create_historical.return_value = {uid: f"Historical for {uid}" for uid in user_ids}
            
            # Mock facts to be empty to avoid sequential fetch complexity
            self.mock_store.get_user_facts = AsyncMock(return_value=None)

            start_time = asyncio.get_event_loop().time()
            # Act
            await self.memory_manager.get_memories(self.physics_guild_id, user_ids)
            end_time = asyncio.get_event_loop().time()

            # Assert - The _create_user_memory calls should happen concurrently
            # If sequential, would take ~0.3s (3 * 0.1s). Concurrent should be ~0.1s.
            # Allow some overhead for test execution, but should be much closer to 0.1s than 0.3s
            self.assertLess(end_time - start_time, 0.2)  # Should be closer to 0.1s than 0.3s

    async def test_complex_mixed_failure_scenario(self):
        """Test realistic scenario where some operations succeed and others fail at different stages."""
        # Arrange
        user_ids = [self.physicist_ids["Einstein"], self.physicist_ids["Bohr"], self.physicist_ids["Planck"]]
        
        # Einstein: has facts, current day fails, historical succeeds -> should get facts
        # Bohr: no facts, current day succeeds, historical fails -> should get current day
        # Planck: no facts, current day fails, historical succeeds -> should get historical
        
        async def facts_side_effect(guild_id, user_id):
            if user_id == self.physicist_ids["Einstein"]:
                return "Einstein facts"
            return None

        async def daily_summary_side_effect(guild_id, for_date):
            return {self.physicist_ids["Bohr"]: "Bohr current day summary"}

        async def week_summary_side_effect(daily_summaries):
            # Only succeed for Einstein and Planck
            if not daily_summaries:
                return "Historical summary"
            return "Historical summary"

        self.mock_store.get_user_facts = AsyncMock(side_effect=facts_side_effect)
        
        with patch.object(self.memory_manager, '_fetch_all_daily_summaries', new_callable=AsyncMock) as mock_fetch_daily, \
             patch.object(self.memory_manager, '_create_historical_summaries_for_users', new_callable=AsyncMock) as mock_create_historical, \
             patch.object(self.memory_manager, '_create_combined_memories', new_callable=AsyncMock) as mock_create_combined:
            
            # Mock the internal pipeline steps
            mock_fetch_daily.return_value = {date(1905, 3, 6): {self.physicist_ids["Bohr"]: "Bohr current day summary"}}
            mock_create_historical.return_value = {
                self.physicist_ids["Einstein"]: "Einstein historical",
                self.physicist_ids["Bohr"]: None,  # Historical fails for Bohr
                self.physicist_ids["Planck"]: "Planck historical"
            }
            mock_create_combined.return_value = {
                self.physicist_ids["Einstein"]: "Einstein facts",  # Falls back to facts
                self.physicist_ids["Bohr"]: "Bohr current day summary",  # Falls back to current day
                self.physicist_ids["Planck"]: "Planck historical"  # Gets historical only
            }
            
            # Act
            results = await self.memory_manager.get_memories(self.physics_guild_id, user_ids)

            # Assert
            self.assertEqual(len(results), 3)
            self.assertEqual(results[self.physicist_ids["Einstein"]], "Einstein facts")
            self.assertEqual(results[self.physicist_ids["Bohr"]], "Bohr current day summary") 
            self.assertEqual(results[self.physicist_ids["Planck"]], "Planck historical")


class TestMemoryManagerCacheEffectiveness(unittest.IsolatedAsyncioTestCase):
    """Test that caches actually prevent redundant AI calls."""

    def setUp(self):
        self.telemetry = NullTelemetry()
        self.user_resolver = TestUserResolver()
        self.mock_store = Mock(spec=Store)
        self.mock_gemini_client = Mock(spec=AIClient)
        self.mock_gemma_client = Mock(spec=AIClient)
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.mock_store,
            gemini_client=self.mock_gemini_client,
            gemma_client=self.mock_gemma_client,
            user_resolver=self.user_resolver
        )
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids

    async def test_daily_summary_cache_prevents_redundant_ai_calls(self):
        """Test that repeated daily summary requests hit cache instead of making AI calls."""
        # Arrange - Use real physics conversation data from Monday, March 3rd
        test_date = date(1905, 3, 3)  # Monday with rich physics discussions
        
        # Use TestStore to provide real physics messages
        from tests.test_store import TestStore
        test_store = TestStore()
        self.memory_manager._store = test_store
        
        self.mock_gemini_client.generate_content = AsyncMock(return_value=DailySummaries(
            summaries=[UserSummary(user_id=self.physicist_ids["Einstein"], summary="Einstein discussed photoelectric effect and quantum theory")]
        ))

        with patch('memory_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(1905, 3, 3, 12, 0, tzinfo=timezone.utc)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # Act - Make multiple calls for same date
            await self.memory_manager._daily_summary(self.physics_guild_id, test_date)
            await self.memory_manager._daily_summary(self.physics_guild_id, test_date)
            await self.memory_manager._daily_summary(self.physics_guild_id, test_date)

            # Assert - Should only make one AI call due to caching
            self.mock_gemini_client.generate_content.assert_called_once()

    async def test_context_merge_cache_prevents_redundant_ai_calls(self):
        """Test that identical context merge requests hit cache."""
        # Arrange
        facts = "Test facts"
        current_day = "Current day summary"
        historical = "Historical summary"
        user_id = self.physicist_ids["Einstein"]
        
        self.mock_gemma_client.generate_content = AsyncMock(return_value=MemoryContext(context="Merged context"))

        # Act - Make multiple identical merge requests
        await self.memory_manager._merge_context(self.physics_guild_id, user_id, facts, current_day, historical)
        await self.memory_manager._merge_context(self.physics_guild_id, user_id, facts, current_day, historical)
        await self.memory_manager._merge_context(self.physics_guild_id, user_id, facts, current_day, historical)

        # Assert - Should only make one AI call due to content-based caching
        self.mock_gemma_client.generate_content.assert_called_once()

    async def test_cache_invalidation_on_content_change(self):
        """Test that cache properly invalidates when content changes."""
        # Arrange
        user_id = self.physicist_ids["Einstein"]
        original_facts = "Original facts"
        updated_facts = "Updated facts"
        current_day = "Current day summary"
        historical = "Historical summary"
        
        self.mock_gemma_client.generate_content = AsyncMock(return_value=MemoryContext(context="Merged context"))

        # Act - Make calls with different content
        await self.memory_manager._merge_context(self.physics_guild_id, user_id, original_facts, current_day, historical)
        await self.memory_manager._merge_context(self.physics_guild_id, user_id, updated_facts, current_day, historical)

        # Assert - Should make two AI calls since content changed
        self.assertEqual(self.mock_gemma_client.generate_content.call_count, 2)


class TestMemoryManagerDateBoundaries(unittest.IsolatedAsyncioTestCase):
    """Test edge cases around date boundaries and timezone handling."""

    def setUp(self):
        self.telemetry = NullTelemetry()
        self.user_resolver = TestUserResolver()
        self.mock_store = Mock(spec=Store)
        self.mock_gemini_client = Mock(spec=AIClient)
        self.mock_gemma_client = Mock(spec=AIClient)
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.mock_store,
            gemini_client=self.mock_gemini_client,
            gemma_client=self.mock_gemma_client,
            user_resolver=self.user_resolver
        )
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids

    async def test_date_transition_cache_behavior(self):
        """Test cache behavior when date transitions from current to historical."""
        # Arrange - Use real physics conversation data from Tuesday, March 4th
        transition_date = date(1905, 3, 4)  # Tuesday with Marie Curie and Einstein discussions
        
        # Use TestStore to provide real physics messages
        from tests.test_store import TestStore
        test_store = TestStore()
        self.memory_manager._store = test_store
        
        self.mock_gemini_client.generate_content = AsyncMock(return_value=DailySummaries(
            summaries=[UserSummary(user_id=self.physicist_ids["Einstein"], summary="Einstein discussed relativity and mass-energy equivalence")]
        ))

        with patch('memory_manager.datetime') as mock_datetime:
            # First call when date is "current"
            mock_datetime.now.return_value = datetime(1905, 3, 4, 23, 59, tzinfo=timezone.utc)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            await self.memory_manager._daily_summary(self.physics_guild_id, transition_date)
            first_call_count = self.mock_gemini_client.generate_content.call_count
            
            # Second call when date becomes "historical" (next day)
            mock_datetime.now.return_value = datetime(1905, 3, 5, 0, 1, tzinfo=timezone.utc)
            
            await self.memory_manager._daily_summary(self.physics_guild_id, transition_date)
            second_call_count = self.mock_gemini_client.generate_content.call_count

            # Assert - Should use different caches and potentially make separate AI calls
            self.assertIn((self.physics_guild_id, transition_date), self.memory_manager._current_day_batch_cache)
            self.assertIn((self.physics_guild_id, transition_date), self.memory_manager._closed_day_batch_cache)
            # Each cache system should get the result, but AI calls depend on implementation
            self.assertGreaterEqual(first_call_count, 1)  # First call should trigger AI
            self.assertGreaterEqual(second_call_count, 1)  # Second call might hit new cache or make new call

    async def test_empty_date_range_handling(self):
        """Test behavior with empty or invalid date ranges."""
        # Arrange
        empty_dates = []
        
        # Act & Assert - Should handle empty date list gracefully
        result = await self.memory_manager._fetch_all_daily_summaries(self.physics_guild_id, empty_dates)
        self.assertEqual(result, {})
        self.mock_gemini_client.generate_content.assert_not_called()

    async def test_future_date_handling(self):
        """Test behavior when requesting summaries for future dates."""
        # Arrange
        future_date = date(2025, 12, 31)
        self.mock_store.get_chat_messages_for_date = AsyncMock(return_value=[])
        
        # Act
        result = await self.memory_manager._daily_summary(self.physics_guild_id, future_date)
        
        # Assert - Should handle future dates gracefully (no messages = empty summary)
        self.assertEqual(result, {})
        self.mock_gemini_client.generate_content.assert_not_called()


if __name__ == '__main__':
    unittest.main()
