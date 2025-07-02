import unittest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, date, timezone, timedelta

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
            result1 = await self.memory_manager._get_current_day_summary(self.physics_guild_id, einstein_id, self.test_date)
            
            # Act - Second call in same hour should hit cache
            result2 = await self.memory_manager._get_current_day_summary(self.physics_guild_id, einstein_id, self.test_date)
        
        # Assert
        expected_summary = "Einstein proposed that light behaves as discrete energy packets, challenging wave theory"
        self.assertEqual(result1, expected_summary)
        self.assertEqual(result2, expected_summary)
        self.mock_gemini_client.generate_content.assert_called_once()  # Only one AI call

    async def test_current_day_cache_miss_different_hour(self):
        """Test that current day summary cache misses in different hours."""
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
            await self.memory_manager._get_current_day_summary(self.physics_guild_id, planck_id, self.test_date)
            
            # Later in afternoon physics discussion (15:30)
            mock_datetime.now.return_value = datetime(1905, 3, 3, 15, 30, tzinfo=timezone.utc)
            
            # Act - Second call in different hour should miss cache
            await self.memory_manager._get_current_day_summary(self.physics_guild_id, planck_id, self.test_date)
        
        # Assert
        self.assertEqual(self.mock_gemini_client.generate_content.call_count, 2)  # Two AI calls for different hours

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
        result1 = await self.memory_manager._get_historical_daily_summary(self.physics_guild_id, bohr_id, historical_date)
        result2 = await self.memory_manager._get_historical_daily_summary(self.physics_guild_id, bohr_id, historical_date)
        
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
        result = await self.memory_manager._generate_daily_summaries_batch(self.physics_guild_id, self.test_date)
        
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
        
        with patch.object(self.memory_manager, '_get_current_day_summary', return_value=None):
            with patch.object(self.memory_manager, '_get_historical_summary', return_value=None):
                # Act
                result = await self.memory_manager.get_memories(self.physics_guild_id, self.physicist_ids["Einstein"])
        
        # Assert
        self.assertIsNone(result)

    async def test_facts_only_returns_facts_directly(self):
        """Test that get_memories returns facts directly when only facts exist."""
        # Arrange - Einstein with only stored facts, no daily activity
        facts = "Einstein is the theoretical physicist who developed special relativity and explained the photoelectric effect"
        self.mock_store.get_user_facts = AsyncMock(return_value=facts)
        
        with patch.object(self.memory_manager, '_get_current_day_summary', return_value=None):
            with patch.object(self.memory_manager, '_get_historical_summary', return_value=None):
                # Act
                result = await self.memory_manager.get_memories(self.physics_guild_id, self.physicist_ids["Einstein"])
        
        # Assert
        self.assertEqual(result, facts)

    async def test_current_day_only_returns_current_day_directly(self):
        """Test that get_memories returns current day summary directly when only it exists."""
        # Arrange - Planck active today discussing blackbody radiation, no stored facts
        current_summary = "Planck defended wave theory while questioning Einstein's quantum hypothesis about energy packets"
        self.mock_store.get_user_facts = AsyncMock(return_value=None)
        
        with patch.object(self.memory_manager, '_get_current_day_summary', return_value=current_summary):
            with patch.object(self.memory_manager, '_get_historical_summary', return_value=None):
                # Act
                result = await self.memory_manager.get_memories(self.physics_guild_id, self.physicist_ids["Planck"])
        
        # Assert
        self.assertEqual(result, current_summary)

    async def test_historical_only_returns_historical_directly(self):
        """Test that get_memories returns historical summary directly when only it exists."""
        # Arrange
        historical_summary = "User has been consistently active"
        self.mock_store.get_user_facts = AsyncMock(return_value=None)
        
        with patch.object(self.memory_manager, '_get_current_day_summary', return_value=None):
            with patch.object(self.memory_manager, '_get_historical_summary', return_value=historical_summary):
                # Act
                result = await self.memory_manager.get_memories(self.physics_guild_id, self.physicist_ids["Bohr"])
        
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
        
        with patch.object(self.memory_manager, '_get_current_day_summary', return_value=current_summary):
            with patch.object(self.memory_manager, '_get_historical_summary', return_value=historical_summary):
                with patch.object(self.memory_manager, '_merge_context', return_value=merged_result) as mock_merge:
                    # Act
                    result = await self.memory_manager.get_memories(self.physics_guild_id, self.physicist_ids["Thomson"])
        
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
        
        with patch.object(self.memory_manager, '_get_current_day_summary', return_value=current_summary):
            with patch.object(self.memory_manager, '_get_historical_summary', return_value=historical_summary):
                with patch.object(self.memory_manager, '_merge_context', side_effect=Exception("AI service unavailable")):
                    # Act
                    result = await self.memory_manager.get_memories(self.physics_guild_id, self.physicist_ids["Rutherford"])
        
        # Assert
        self.assertEqual(result, facts)

    async def test_merge_failure_falls_back_to_current_day_when_no_facts(self):
        """Test that get_memories falls back to current day when merge fails and no facts."""
        # Arrange
        current_summary = "User worked today"
        historical_summary = "User programs regularly"
        
        self.mock_store.get_user_facts = AsyncMock(return_value=None)
        
        with patch.object(self.memory_manager, '_get_current_day_summary', return_value=current_summary):
            with patch.object(self.memory_manager, '_get_historical_summary', return_value=historical_summary):
                with patch.object(self.memory_manager, '_merge_context', side_effect=Exception("AI service unavailable")):
                    # Act
                    result = await self.memory_manager.get_memories(self.physics_guild_id, self.physicist_ids["Schrödinger"])
        
        # Assert
        self.assertEqual(result, current_summary)

    async def test_merge_failure_falls_back_to_historical_when_no_facts_or_current(self):
        """Test that get_memories falls back to historical when merge fails and no facts or current day."""
        # Arrange
        historical_summary = "User programs regularly"
        
        self.mock_store.get_user_facts = AsyncMock(return_value=None)
        
        with patch.object(self.memory_manager, '_get_current_day_summary', return_value=None):
            with patch.object(self.memory_manager, '_get_historical_summary', return_value=historical_summary):
                with patch.object(self.memory_manager, '_merge_context', side_effect=Exception("AI service unavailable")):
                    # Act
                    result = await self.memory_manager.get_memories(self.physics_guild_id, self.physicist_ids["Heisenberg"])
        
        # Assert
        self.assertEqual(result, historical_summary)

    async def test_store_failure_propagates_exception(self):
        """Test that store failures propagate as exceptions since facts are critical."""
        # Arrange
        self.mock_store.get_user_facts = AsyncMock(side_effect=Exception("Database connection failed"))
        
        # Act & Assert - Store failure should propagate since facts are critical
        with self.assertRaises(Exception) as context:
            await self.memory_manager.get_memories(self.physics_guild_id, self.physicist_ids["Born"])
        
        self.assertIn("Database connection failed", str(context.exception))

    async def test_current_day_summary_failure_handled_gracefully(self):
        """Test that current day summary failures are handled gracefully."""
        # Arrange
        facts = "User likes coffee"
        self.mock_store.get_user_facts = AsyncMock(return_value=facts)
        
        with patch.object(self.memory_manager, '_get_current_day_summary', side_effect=Exception("AI service error")):
            with patch.object(self.memory_manager, '_get_historical_summary', return_value=None):
                # Act
                result = await self.memory_manager.get_memories(self.physics_guild_id, self.physicist_ids["Curie"])
        
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
        await self.memory_manager._generate_daily_summaries_batch(self.physics_guild_id, self.test_date)
        
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
        await self.memory_manager._generate_daily_summaries_batch(self.physics_guild_id, self.test_date)
        
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
        # Arrange
        current_date = date(2025, 7, 1)
        expected_dates = [
            date(2025, 6, 30),  # Yesterday (day 2)
            date(2025, 6, 29),  # Day 3
            date(2025, 6, 28),  # Day 4
            date(2025, 6, 27),  # Day 5
            date(2025, 6, 26),  # Day 6
            date(2025, 6, 25),  # Day 7
        ]
        
        # Mock to return summaries for first 3 dates only
        async def mock_get_historical_daily_summary(guild_id, user_id, for_date):
            if for_date in expected_dates[:3]:
                return f"Summary for {for_date}"
            return None
        
        # Mock returns handled by TestUserResolver automatically
        self.mock_gemma_client.generate_content = AsyncMock(return_value=MemoryContext(
            context="Historical summary"
        ))
        
        with patch.object(self.memory_manager, '_get_historical_daily_summary', side_effect=mock_get_historical_daily_summary):
            # Act
            result = await self.memory_manager._get_historical_summary(self.physics_guild_id, self.physicist_ids["Einstein"], current_date)
        
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



if __name__ == '__main__':
    unittest.main()