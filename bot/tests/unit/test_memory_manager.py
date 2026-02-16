import unittest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, date, timezone, timedelta
import asyncio

from memory_manager import MemoryManager
from ai_client import AIClient, BlockedException
from message_node import MessageNode
from schemas import MemoryContext, DailySummaries, UserSummary, UserAliases
from store import Store
from null_redis_cache import NullRedisCache
from null_telemetry import NullTelemetry
from test_user_resolver import TestUserResolver
from test_store import TestStore  # Moved import to top


class TestMemoryManagerCaching(unittest.IsolatedAsyncioTestCase):
    """Test cache behavior and key generation for MemoryManager."""

    def setUp(self):
        self.telemetry = NullTelemetry()

        # Use TestStore for realistic data and interactions
        self.test_store = TestStore()
        self.user_resolver = self.test_store.user_resolver
        self.redis_cache = NullRedisCache()

        # Mock AI clients
        self.mock_gemini_client = Mock(spec=AIClient)
        self.mock_gemma_client = Mock(spec=AIClient)

        # Component under test
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.test_store,
            gemini_client=self.mock_gemini_client,
            gemma_client=self.mock_gemma_client,
            user_resolver=self.user_resolver,
            redis_cache=self.redis_cache,
        )

        # Test data - Physics Guild with real physicist IDs
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids
        self.test_date = date(1905, 3, 3)  # March 3, 1905 - Annus Mirabilis year

    async def test_historical_daily_cache_hit(self):
        """Test that historical daily summary cache hits for same guild/date, avoiding AI calls."""
        # Arrange - Use TestStore for realistic database behavior
        from test_store import TestStore

        test_store = TestStore()
        self.memory_manager._store = test_store  # Replace mock store with real test double

        # Use a date that has messages in TestStore to make the test realistic
        historical_date = date(1905, 3, 4)
        bohr_id = self.physicist_ids["Bohr"]
        expected_summary = "Bohr proposed quantized atomic energy levels to explain hydrogen spectra"
        expected_summaries = {bohr_id: expected_summary}

        # Pre-populate the "database" with the summary to simulate a cache hit
        await test_store.save_daily_summaries(self.physics_guild_id, historical_date, expected_summaries)

        # Ensure the AI client mock is clean before the test action
        self.mock_gemini_client.generate_content.reset_mock()

        with patch("memory_manager.datetime") as mock_datetime:
            # Set current time to be after the historical date
            mock_datetime.now.return_value = datetime(1905, 3, 5, 12, 0, tzinfo=timezone.utc)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # Act - A single call should be sufficient to test the database cache
            daily_summaries = await self.memory_manager._daily_summary(self.physics_guild_id, historical_date)
            result = daily_summaries.get(bohr_id)

        # Assert
        self.assertEqual(result, expected_summary)
        # The key assertion: The database cache was hit, so no AI call was made
        self.mock_gemini_client.generate_content.assert_not_called()

    async def test_context_cache_hit_identical_content(self):
        """Test that context merge cache hits when content is identical."""
        # Arrange - Einstein's context across multiple memory sources
        facts = "Einstein is a theoretical physicist known for relativity theory and photoelectric effect work"
        current_day = "Einstein discussed quantum nature of light and challenged classical wave theory"
        historical = "Einstein has been developing revolutionary theories about space, time, and energy"

        # Mock returns handled by TestUserResolver automatically
        self.mock_gemma_client.generate_content = AsyncMock(
            return_value=MemoryContext(
                context=(
                    "Einstein is a revolutionary physicist who challenges classical physics"
                    " with quantum and relativity theories"
                )
            )
        )

        # Act - Two calls with identical content
        einstein_id = self.physicist_ids["Einstein"]
        daily_summaries = {date(2025, 1, 1): current_day, date(2025, 1, 2): historical}
        result1 = await self.memory_manager._merge_context(self.physics_guild_id, einstein_id, facts, daily_summaries)
        result2 = await self.memory_manager._merge_context(self.physics_guild_id, einstein_id, facts, daily_summaries)

        # Assert
        expected_context = (
            "Einstein is a revolutionary physicist who challenges classical physics"
            " with quantum and relativity theories"
        )
        self.assertEqual(result1, expected_context)
        self.assertEqual(result2, expected_context)
        self.mock_gemma_client.generate_content.assert_called_once()  # Only one AI call

    async def test_context_cache_miss_changed_facts(self):
        """Test that context merge cache misses when facts change."""
        # Arrange - Bohr's facts being updated as his atomic model evolves
        original_facts = "Bohr is a physicist working on atomic structure"
        updated_facts = (
            "Bohr is a physicist who proposed quantized electron orbits in atoms, explaining hydrogen spectra"
        )
        current_day = "Bohr discussed revolutionary atomic models with quantized energy levels"
        historical = "Bohr has been developing new atomic theories that explain spectral lines"

        # Mock returns handled by TestUserResolver automatically
        self.mock_gemma_client.generate_content = AsyncMock(
            return_value=MemoryContext(context="Bohr is revolutionizing atomic physics with quantum orbital theory")
        )

        # Act - Two calls with evolving facts about Bohr's discoveries
        bohr_id = self.physicist_ids["Bohr"]
        daily_summaries_orig = {date(2025, 1, 1): current_day, date(2025, 1, 2): historical}
        daily_summaries_upd = {date(2025, 1, 1): current_day, date(2025, 1, 2): historical}
        await self.memory_manager._merge_context(self.physics_guild_id, bohr_id, original_facts, daily_summaries_orig)
        await self.memory_manager._merge_context(self.physics_guild_id, bohr_id, updated_facts, daily_summaries_upd)

        # Assert
        self.assertEqual(self.mock_gemma_client.generate_content.call_count, 2)  # Two AI calls

    async def test_empty_messages_returns_empty_dict(self):
        """Test that batch generation with no messages returns empty dict."""
        # Arrange - Use a date with no messages in TestStore
        no_message_date = date(1905, 3, 2)

        # Act
        result = await self.memory_manager._create_daily_summaries(self.physics_guild_id, no_message_date)

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
            user_resolver=self.user_resolver,
            redis_cache=NullRedisCache(),
        )

        # Test data - Physics Guild with real physicist IDs
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids

    async def test_no_memories_returns_none(self):
        """Test that get_memories returns None when no memories exist."""
        # Arrange
        self.mock_store.get_user_facts = AsyncMock(return_value=None)

        with patch.object(self.memory_manager, "_daily_summary", return_value={}):
            # Act
            result = await self.memory_manager.get_memory(self.physics_guild_id, self.physicist_ids["Einstein"])

        # Assert
        self.assertIsNone(result)

    async def test_facts_only_returns_facts_directly(self):
        """Test that get_memories returns facts directly when only facts exist."""
        # Arrange - Einstein with only stored facts, no daily activity
        facts = (
            "Einstein is the theoretical physicist who developed special relativity"
            " and explained the photoelectric effect"
        )
        self.mock_store.get_user_facts = AsyncMock(return_value=facts)

        with patch.object(self.memory_manager, "_daily_summary", return_value={}):
            # Act
            result = await self.memory_manager.get_memory(self.physics_guild_id, self.physicist_ids["Einstein"])

        # Assert
        self.assertEqual(result, facts)

    async def test_current_day_only_returns_current_day_directly(self):
        """Test that get_memories returns current day summary directly when only it exists."""
        # Arrange - Planck active today discussing blackbody radiation, no stored facts
        current_summary = (
            "Planck defended wave theory while questioning Einstein's quantum hypothesis about energy packets"
        )
        self.mock_store.get_user_facts = AsyncMock(return_value=None)

        with patch.object(
            self.memory_manager,
            "_daily_summary",
            side_effect=lambda guild_id, date: {self.physicist_ids["Planck"]: current_summary}
            if date == datetime.now(timezone.utc).date()
            else {},
        ):
            # Act
            result = await self.memory_manager.get_memory(self.physics_guild_id, self.physicist_ids["Planck"])

        # Assert
        self.assertEqual(result, current_summary)

    async def test_historical_only_returns_historical_directly(self):
        """Test that get_memories returns historical summary directly when only it exists."""
        # Arrange
        historical_summary = "User has been consistently active"
        self.mock_store.get_user_facts = AsyncMock(return_value=None)

        # Mock to return exactly one daily summary (for yesterday only)
        def daily_summary_side_effect(guild_id, date):
            if date == datetime.now(timezone.utc).date() - timedelta(days=1):
                return {self.physicist_ids["Bohr"]: historical_summary}
            return {}

        with patch.object(self.memory_manager, "_daily_summary", side_effect=daily_summary_side_effect):
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

        with patch.object(
            self.memory_manager,
            "_daily_summary",
            side_effect=lambda guild_id, date: {self.physicist_ids["Thomson"]: current_summary}
            if date == datetime.now(timezone.utc).date()
            else {self.physicist_ids["Thomson"]: historical_summary},
        ):
            with patch.object(self.memory_manager, "_merge_context", return_value=merged_result) as mock_merge:
                # Act
                result = await self.memory_manager.get_memory(self.physics_guild_id, self.physicist_ids["Thomson"])

        # Assert
        self.assertEqual(result, merged_result)
        # Verify merge was called with facts and a dict of daily summaries
        self.assertEqual(len(mock_merge.call_args_list), 1)
        call_args = mock_merge.call_args_list[0]
        self.assertEqual(call_args[0][0], self.physics_guild_id)  # guild_id
        self.assertEqual(call_args[0][1], self.physicist_ids["Thomson"])  # user_id
        self.assertEqual(call_args[0][2], facts)  # facts
        daily_summaries = call_args[0][3]  # daily_summaries dict
        self.assertIsInstance(daily_summaries, dict)
        self.assertIn(datetime.now(timezone.utc).date(), daily_summaries)
        self.assertEqual(daily_summaries[datetime.now(timezone.utc).date()], current_summary)

    async def test_merge_failure_falls_back_to_facts(self):
        """Test that get_memories falls back to facts when AI merge fails."""
        # Arrange
        facts = "User likes coffee"
        current_summary = "User worked today"
        historical_summary = "User programs regularly"

        self.mock_store.get_user_facts = AsyncMock(return_value=facts)

        with patch.object(
            self.memory_manager,
            "_daily_summary",
            side_effect=lambda guild_id, date: {self.physicist_ids["Rutherford"]: current_summary}
            if date == datetime.now(timezone.utc).date()
            else {self.physicist_ids["Rutherford"]: historical_summary},
        ):
            with patch.object(self.memory_manager, "_merge_context", side_effect=Exception("AI service unavailable")):
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

        with patch.object(
            self.memory_manager,
            "_daily_summary",
            side_effect=lambda guild_id, date: {self.physicist_ids["Schrödinger"]: current_summary}
            if date == datetime.now(timezone.utc).date()
            else {self.physicist_ids["Schrödinger"]: historical_summary},
        ):
            with patch.object(self.memory_manager, "_merge_context", side_effect=Exception("AI service unavailable")):
                # Act
                result = await self.memory_manager.get_memory(self.physics_guild_id, self.physicist_ids["Schrödinger"])

        # Assert
        # When merge fails, fallback goes to most recent daily summary (which should be current day)
        self.assertEqual(result, current_summary)

    async def test_merge_failure_falls_back_to_historical_when_no_facts_or_current(self):
        """Test that get_memories falls back to historical when merge fails and no facts or current day."""
        # Arrange
        historical_summary = "User programs regularly"

        self.mock_store.get_user_facts = AsyncMock(return_value=None)

        with patch.object(
            self.memory_manager,
            "_daily_summary",
            side_effect=lambda guild_id, date: {}
            if date == datetime.now(timezone.utc).date()
            else {self.physicist_ids["Heisenberg"]: historical_summary},
        ):
            with patch.object(self.memory_manager, "_merge_context", side_effect=Exception("AI service unavailable")):
                # Act
                result = await self.memory_manager.get_memory(self.physics_guild_id, self.physicist_ids["Heisenberg"])

        # Assert
        self.assertEqual(result, historical_summary)

    async def test_store_failure_propagates_exception(self):
        """Test that store failures propagate exceptions (no longer handled gracefully)."""
        # Arrange
        self.mock_store.get_user_facts = AsyncMock(side_effect=Exception("Database connection failed"))

        # Mock daily summary to return empty for fallback
        with patch.object(self.memory_manager, "_daily_summary", return_value={}):
            # Act & Assert - Store failure should propagate exception
            with self.assertRaises(Exception) as context:
                await self.memory_manager.get_memory(self.physics_guild_id, self.physicist_ids["Born"])

            self.assertEqual(str(context.exception), "Database connection failed")

    async def test_current_day_summary_failure_handled_gracefully(self):
        """Test that current day summary failures are handled gracefully."""
        # Arrange
        facts = "User likes coffee"
        self.mock_store.get_user_facts = AsyncMock(return_value=facts)

        with patch.object(self.memory_manager, "_daily_summary", side_effect=Exception("AI service error")):
            # Act
            result = await self.memory_manager.get_memory(self.physics_guild_id, self.physicist_ids["Curie"])

        # Assert - Should still return facts despite current day failure
        self.assertEqual(result, facts)


class TestMemoryManagerDataProcessing(unittest.IsolatedAsyncioTestCase):
    """Test data processing and formatting for MemoryManager."""

    def setUp(self):
        self.telemetry = NullTelemetry()

        # Use TestStore for realistic data and interactions
        self.test_store = TestStore()
        self.user_resolver = self.test_store.user_resolver

        # Mock AI clients
        self.mock_gemini_client = Mock(spec=AIClient)
        self.mock_gemma_client = Mock(spec=AIClient)

        # Component under test
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.test_store,
            gemini_client=self.mock_gemini_client,
            gemma_client=self.mock_gemma_client,
            user_resolver=self.user_resolver,
            redis_cache=NullRedisCache(),
        )

        # Test data - Physics Guild with real physicist IDs
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids
        self.test_date = date(1905, 3, 3)  # March 3, 1905 - Annus Mirabilis year

    async def test_message_formatting_xml_structure(self):
        """Test that messages are formatted correctly in XML structure."""
        # Arrange
        einstein_id = self.physicist_ids["Einstein"]

        self.mock_gemini_client.generate_content = AsyncMock(
            return_value=DailySummaries(
                summaries=[
                    UserSummary(user_id=einstein_id, summary="Einstein discussed quantum hypothesis with Planck")
                ]
            )
        )

        # Act
        await self.memory_manager._create_daily_summaries(self.physics_guild_id, self.test_date)

        # Assert - Check that the prompt contains properly formatted XML
        call_args = self.mock_gemini_client.generate_content.call_args
        prompt = call_args[1]["message"]  # keyword argument

        self.assertIn("<message>", prompt)
        self.assertIn("<timestamp>1905-03-03 09:15:00+00:00</timestamp>", prompt)
        self.assertIn(f"<author_id>{einstein_id}</author_id>", prompt)
        self.assertIn("<author>Einstein</author>", prompt)

        # This content comes directly from TestStore's data for March 3rd, 1905
        expected_content = (
            "Good morning colleagues. I've been pondering the photoelectric effect."
            " Light seems to behave as discrete packets of energy,"
            " not continuous waves as we've long assumed."
        )
        self.assertIn(f"<content>{expected_content}</content>", prompt)

        self.assertIn("</message>", prompt)

    async def test_user_deduplication(self):
        """Test that AI prompt contains deduplicated user list despite duplicate messages."""
        # Arrange - TestStore for March 3rd contains 10 messages from multiple users, with duplicates
        einstein_id = self.physicist_ids["Einstein"]
        bohr_id = self.physicist_ids["Bohr"]

        self.mock_gemini_client.generate_content = AsyncMock(
            return_value=DailySummaries(
                summaries=[
                    UserSummary(user_id=einstein_id, summary="Einstein discussed quantum theory"),
                    UserSummary(user_id=bohr_id, summary="Bohr proposed quantized energy levels"),
                ]
            )
        )

        # Act
        await self.memory_manager._create_daily_summaries(self.physics_guild_id, self.test_date)

        # Assert - Check that AI prompt contains deduplicated user list
        call_args = self.mock_gemini_client.generate_content.call_args
        prompt = call_args[1]["message"]

        # Should contain each user only once in the target_users section
        einstein_user_entries = prompt.count(f"<user_id>{einstein_id}</user_id>")
        bohr_user_entries = prompt.count(f"<user_id>{bohr_id}</user_id>")

        self.assertEqual(
            einstein_user_entries, 1, "Einstein should appear only once in user list despite multiple messages"
        )
        self.assertEqual(bohr_user_entries, 1, "Bohr should appear only once in user list")

        # But all 10 messages from TestStore for that date should be in the prompt
        self.assertEqual(prompt.count("<message>"), 10, "All 10 individual messages should be included")

    async def test_ingest_message_adds_to_store(self):
        """Test that ingest_message correctly adds a message to the store."""
        # Arrange
        message_date = date(2025, 7, 1)
        message = MessageNode(
            id=12345,
            channel_id=67890,
            author_id=11111,
            content="Test message content for state-based test",
            mentioned_user_ids=[],
            created_at=datetime(2025, 7, 1, 12, 0, tzinfo=timezone.utc),
        )

        # Get initial state
        initial_messages = await self.test_store.get_chat_messages_for_date(self.physics_guild_id, message_date)
        initial_message_count = len(initial_messages)

        # Act
        await self.memory_manager.ingest_message(self.physics_guild_id, message)

        # Assert
        # Check that the number of messages has increased by one for that date
        final_messages = await self.test_store.get_chat_messages_for_date(self.physics_guild_id, message_date)
        self.assertEqual(len(final_messages), initial_message_count + 1)

        # Find the newly added message (it should be the last one)
        added_message = final_messages[-1]

        # Check that the added message has the correct data
        self.assertEqual(added_message.guild_id, self.physics_guild_id)
        self.assertEqual(added_message.channel_id, message.channel_id)
        self.assertEqual(added_message.user_id, message.author_id)
        self.assertEqual(added_message.message_text, message.content)
        self.assertEqual(added_message.timestamp, message.created_at)


class TestMemoryManagerBatchProcessing(unittest.IsolatedAsyncioTestCase):
    """Test batch processing and concurrent operations for MemoryManager."""

    def setUp(self):
        self.telemetry = NullTelemetry()

        # Use TestStore for realistic data and interactions
        self.test_store = TestStore()
        self.user_resolver = self.test_store.user_resolver

        self.mock_gemini_client = Mock(spec=AIClient)
        self.mock_gemma_client = Mock(spec=AIClient)
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.test_store,
            gemini_client=self.mock_gemini_client,
            gemma_client=self.mock_gemma_client,
            user_resolver=self.user_resolver,
            redis_cache=NullRedisCache(),
        )
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids

    async def test_get_memories_batch_processing_multiple_users(self):
        """Test that get_memories() processes multiple users and returns correct dict mapping."""
        # Arrange
        user_ids = [self.physicist_ids["Einstein"], self.physicist_ids["Bohr"], self.physicist_ids["Planck"]]

        # Mock the internal methods to simulate data for these users
        with (
            patch.object(self.memory_manager, "_fetch_all_daily_summaries", new_callable=AsyncMock) as mock_fetch_daily,
            patch.object(
                self.memory_manager, "_create_combined_memories", new_callable=AsyncMock
            ) as mock_create_combined,
        ):
            mock_fetch_daily.return_value = {}  # Assume no daily summaries for simplicity
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

    async def test_daily_summary_blocked_returns_empty(self):
        """Daily summary should return empty dict when AI blocks the request."""
        historical_date = date(1905, 3, 3)
        blocked_reason = "PROHIBITED_CONTENT"

        # Gemini client refuses to generate content
        self.mock_gemini_client.generate_content = AsyncMock(side_effect=BlockedException(reason=blocked_reason))

        result = await self.memory_manager._daily_summary(self.physics_guild_id, historical_date)

        self.assertEqual(result, {}, "Blocked summaries should return empty dict")
        self.mock_gemini_client.generate_content.assert_awaited_once()
        # Store should persist the empty result to avoid retries
        stored = await self.test_store.get_daily_summaries(self.physics_guild_id, historical_date)
        self.assertEqual(stored, {})


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
            user_resolver=self.user_resolver,
            redis_cache=NullRedisCache(),
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

        with patch.object(self.memory_manager, "_daily_summary", side_effect=daily_summary_side_effect):
            # Act
            results = await self.memory_manager._fetch_all_daily_summaries(self.physics_guild_id, self.all_dates)

            # Assert - Main goal is that successful dates still work despite failures
            self.assertIn(successful_date, results)
            self.assertEqual(results[successful_date], {self.physicist_ids["Einstein"]: "Successful summary"})
            # The failing date gets empty dict (defaultdict behavior), but key point is other dates succeed
            if failing_date in results:
                self.assertEqual(results[failing_date], {})

    async def test_merge_failure_isolated_per_user(self):
        """Test that one user's merge failure doesn't affect other users."""
        # Arrange
        einstein_id = self.physicist_ids["Einstein"]
        bohr_id = self.physicist_ids["Bohr"]  # Bohr's merge will fail
        planck_id = self.physicist_ids["Planck"]
        user_ids = [einstein_id, bohr_id, planck_id]

        # Mock facts
        self.mock_store.get_user_facts = AsyncMock(
            side_effect=lambda guild_id, user_id: f"Facts for {user_id}" if user_id != bohr_id else "Bohr facts"
        )

        # Mock merge context to fail for Bohr only
        async def merge_context_side_effect(guild_id, user_id, facts, daily_summaries):
            if user_id == bohr_id:
                raise ValueError("AI merge failed for Bohr")
            return f"Merged context for {user_id}"

        # Create daily summaries for all users
        daily_summaries_by_date = {
            self.today: {einstein_id: "Einstein today", planck_id: "Planck today"},
            self.today - timedelta(days=1): {
                einstein_id: "Einstein yesterday",
                bohr_id: "Bohr yesterday",
                planck_id: "Planck yesterday",
            },
        }

        with patch.object(self.memory_manager, "_merge_context", side_effect=merge_context_side_effect):
            # Act
            results = await self.memory_manager._create_combined_memories(
                self.physics_guild_id, user_ids, daily_summaries_by_date
            )

            # Assert
            self.assertIn(einstein_id, results)
            self.assertEqual(results[einstein_id], f"Merged context for {einstein_id}")
            self.assertIn(planck_id, results)
            self.assertEqual(results[planck_id], f"Merged context for {planck_id}")
            self.assertIn(bohr_id, results)
            self.assertEqual(results[bohr_id], "Bohr facts")  # Falls back to facts on merge failure

    async def test_context_merge_failure_isolated_per_user(self):
        """Test that merge failures are handled independently per user."""
        # Arrange - Test the _create_user_memory method directly instead of using side effects with asyncio.gather
        einstein_id = self.physicist_ids["Einstein"]
        bohr_id = self.physicist_ids["Bohr"]  # Bohr's merge will fail
        planck_id = self.physicist_ids["Planck"]

        # Mock successful merge for Einstein and Planck, failure for Bohr
        async def merge_context_side_effect(guild_id, user_id, facts, daily_summaries):
            if user_id == bohr_id:
                raise ValueError("AI merge failed for Bohr")
            return f"Merged context for {user_id}"

        with patch.object(self.memory_manager, "_merge_context", side_effect=merge_context_side_effect):
            # Act - Test individual user memory creation to verify isolation
            einstein_result = await self.memory_manager._create_user_memory(
                self.physics_guild_id, einstein_id, "Einstein facts", {self.today: "Einstein current"}
            )
            bohr_result = await self.memory_manager._create_user_memory(
                self.physics_guild_id, bohr_id, "Bohr facts", {self.today: "Bohr current"}
            )
            planck_result = await self.memory_manager._create_user_memory(
                self.physics_guild_id, planck_id, "Planck facts", {self.today: "Planck current"}
            )

            # Assert - Einstein and Planck succeed, Bohr falls back to facts
            self.assertEqual(einstein_result, f"Merged context for {einstein_id}")
            self.assertEqual(bohr_result, "Bohr facts")  # Falls back to facts when merge fails
            self.assertEqual(planck_result, f"Merged context for {planck_id}")


class TestMemoryManagerCacheArchitecture(unittest.IsolatedAsyncioTestCase):
    """Test the cache architecture of the MemoryManager."""

    def setUp(self):
        self.telemetry = NullTelemetry()

        # Use TestStore for realistic data and interactions
        self.test_store = TestStore()
        self.user_resolver = self.test_store.user_resolver
        self.redis_cache = NullRedisCache()

        self.mock_gemini_client = Mock(spec=AIClient)
        self.mock_gemma_client = Mock(spec=AIClient)
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.test_store,
            gemini_client=self.mock_gemini_client,
            gemma_client=self.mock_gemma_client,
            user_resolver=self.user_resolver,
            redis_cache=self.redis_cache,
        )
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids

    async def test_current_day_vs_historical_day_cache_separation(self):
        """Test that current day stores in Redis while historical days store in database."""
        today = date(1905, 3, 6)
        yesterday = date(1905, 3, 5)
        einstein_id = self.physicist_ids["Einstein"]

        self.mock_gemini_client.generate_content = AsyncMock(
            return_value=DailySummaries(
                summaries=[UserSummary(user_id=einstein_id, summary="Einstein discussed physics")]
            )
        )

        with patch("memory_manager.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(1905, 3, 6, 12, 0, tzinfo=timezone.utc)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # Act - Process both today and yesterday
            await self.memory_manager._daily_summary(self.physics_guild_id, today)
            yesterday_result = await self.memory_manager._daily_summary(self.physics_guild_id, yesterday)
            await asyncio.sleep(0.1)  # Wait for today's async rebuild

            # Assert - Today goes to Redis, NOT database
            redis_today = await self.redis_cache.get_daily_summary(self.physics_guild_id, today)
            self.assertIsNotNone(redis_today)
            stored_today = await self.test_store.get_daily_summaries(self.physics_guild_id, today)
            self.assertEqual(stored_today, {})

            # Assert - Yesterday goes to database, NOT Redis
            redis_yesterday = await self.redis_cache.get_daily_summary(self.physics_guild_id, yesterday)
            self.assertIsNone(redis_yesterday)
            stored_yesterday = await self.test_store.get_daily_summaries(self.physics_guild_id, yesterday)
            self.assertEqual(stored_yesterday, yesterday_result)

    async def test_context_merge_content_based_hashing(self):
        """Test that context merge uses content hashes as cache keys."""
        # Arrange
        daily_summaries = {date(1905, 3, 5): "Summary A", date(1905, 3, 4): "Summary B"}
        facts = "Test facts"
        user_id = self.physicist_ids["Einstein"]
        self.mock_gemma_client.generate_content.return_value = MemoryContext(context="Merged Summary")

        # Act
        await self.memory_manager._merge_context(self.physics_guild_id, user_id, facts, daily_summaries)
        await self.memory_manager._merge_context(self.physics_guild_id, user_id, facts, daily_summaries)  # Same content

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
        daily_summaries = {
            datetime.now(timezone.utc).date(): current_day,
            datetime.now(timezone.utc).date() - timedelta(days=1): historical,
        }
        await self.memory_manager._merge_context(self.physics_guild_id, user_id, facts, daily_summaries)
        await self.memory_manager._merge_context(self.physics_guild_id, user_id, facts, daily_summaries)

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
            user_resolver=self.user_resolver,
            redis_cache=NullRedisCache(),
        )
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids

    async def test_single_source_memory_skips_ai_merge(self):
        """Test that when only one memory source exists, AI merge is skipped."""
        # Arrange
        facts = "Just the facts"
        user_id = self.physicist_ids["Einstein"]

        # Act
        result = await self.memory_manager._create_user_memory(self.physics_guild_id, user_id, facts, {})

        # Assert
        self.assertEqual(result, facts)
        self.mock_gemma_client.generate_content.assert_not_called()

    async def test_merge_failure_cascading_fallback_priority(self):
        """Test fallback hierarchy: facts -> most recent daily summary -> None."""
        # Arrange
        user_id = self.physicist_ids["Einstein"]
        facts = "Facts are most important"
        recent_summary = "Recent daily summary"
        older_summary = "Older daily summary"

        self.mock_gemma_client.generate_content.side_effect = Exception("AI merge failed")

        # Test case 1: Facts present, should fall back to facts regardless of daily summaries
        daily_summaries = {date(2025, 1, 3): recent_summary, date(2025, 1, 1): older_summary}
        result1 = await self.memory_manager._create_user_memory(self.physics_guild_id, user_id, facts, daily_summaries)
        self.assertEqual(result1, facts)

        # Test case 2: No facts, should fall back to most recent daily summary
        result2 = await self.memory_manager._create_user_memory(self.physics_guild_id, user_id, None, daily_summaries)
        self.assertEqual(result2, recent_summary)  # Should pick the most recent date (2025-1-3)

        # Test case 3: No sources, should return None
        result3 = await self.memory_manager._create_user_memory(self.physics_guild_id, user_id, None, {})
        self.assertIsNone(result3)


class TestMemoryManagerConcurrency(unittest.IsolatedAsyncioTestCase):
    """Test concurrent processing validation for MemoryManager."""

    def setUp(self):
        self.telemetry = NullTelemetry()

        # Use TestStore for realistic data and interactions
        self.test_store = TestStore()
        self.user_resolver = self.test_store.user_resolver

        self.mock_gemini_client = Mock(spec=AIClient)
        self.mock_gemma_client = Mock(spec=AIClient)
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.test_store,
            gemini_client=self.mock_gemini_client,
            gemma_client=self.mock_gemma_client,
            user_resolver=self.user_resolver,
            redis_cache=NullRedisCache(),
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

        with patch.object(self.memory_manager, "_daily_summary", side_effect=slow_daily_summary) as mock_daily_summary:
            start_time = asyncio.get_event_loop().time()
            # Act
            await self.memory_manager._fetch_all_daily_summaries(self.physics_guild_id, self.all_dates)
            end_time = asyncio.get_event_loop().time()

            # Assert
            self.assertEqual(mock_daily_summary.call_count, len(self.all_dates))
            # If sequential, it would take at least 0.7s. Concurrent should be much faster.
            self.assertLess(end_time - start_time, 0.5)

    async def test_create_combined_memories_concurrent_processing(self):
        """Test that multiple users' memories are processed concurrently."""
        # Arrange
        user_ids = [self.physicist_ids["Einstein"], self.physicist_ids["Bohr"], self.physicist_ids["Planck"]]
        daily_summaries_by_date = {self.all_dates[1]: {uid: "Summary" for uid in user_ids}}

        # Mock facts for all users
        self.test_store.get_user_facts = AsyncMock(side_effect=lambda guild_id, user_id: f"Facts for {user_id}")

        async def slow_merge_context(*args, **kwargs):
            await asyncio.sleep(0.1)
            return "Merged context"

        with patch.object(self.memory_manager, "_merge_context", side_effect=slow_merge_context) as mock_merge:
            start_time = asyncio.get_event_loop().time()
            # Act
            await self.memory_manager._create_combined_memories(
                self.physics_guild_id, user_ids, daily_summaries_by_date
            )
            end_time = asyncio.get_event_loop().time()

            # Assert
            self.assertEqual(mock_merge.call_count, len(user_ids))
            # If sequential, it would take at least 0.3s. Concurrent should be much faster.
            self.assertLess(end_time - start_time, 0.2)


class TestMemoryManagerDatabaseConcurrency(unittest.IsolatedAsyncioTestCase):
    """Test database operations are properly concurrent."""

    def setUp(self):
        self.telemetry = NullTelemetry()

        # Use TestStore for realistic data and interactions
        self.test_store = TestStore()
        self.user_resolver = self.test_store.user_resolver

        self.mock_gemini_client = Mock(spec=AIClient)
        self.mock_gemma_client = Mock(spec=AIClient)
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.test_store,
            gemini_client=self.mock_gemini_client,
            gemma_client=self.mock_gemma_client,
            user_resolver=self.user_resolver,
            redis_cache=NullRedisCache(),
        )
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids

    async def test_user_memory_creation_concurrent(self):
        """Test that user memory creation (merge operations) are processed concurrently."""
        # Arrange
        user_ids = [self.physicist_ids["Einstein"], self.physicist_ids["Bohr"], self.physicist_ids["Planck"]]

        async def slow_create_user_memory(guild_id, user_id, facts, daily_summaries):
            await asyncio.sleep(0.1)  # Simulate AI processing time
            return f"Memory for {user_id}"

        with patch.object(self.memory_manager, "_create_user_memory", side_effect=slow_create_user_memory):
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

        async def facts_side_effect(guild_id, user_id):
            if user_id == self.physicist_ids["Einstein"]:
                return "Einstein facts"
            return None

        async def daily_summary_side_effect(guild_id, for_date):
            return {self.physicist_ids["Bohr"]: "Bohr current day summary"}

        self.test_store.get_user_facts = AsyncMock(side_effect=facts_side_effect)

        with (
            patch.object(self.memory_manager, "_fetch_all_daily_summaries", new_callable=AsyncMock) as mock_fetch_daily,
            patch.object(
                self.memory_manager, "_create_combined_memories", new_callable=AsyncMock
            ) as mock_create_combined,
        ):
            # Mock the internal pipeline steps
            mock_fetch_daily.return_value = {date(1905, 3, 6): {self.physicist_ids["Bohr"]: "Bohr current day summary"}}
            mock_create_combined.return_value = {
                self.physicist_ids["Einstein"]: "Einstein facts",  # Falls back to facts
                self.physicist_ids["Bohr"]: "Bohr current day summary",  # Falls back to current day
                self.physicist_ids["Planck"]: "Planck historical",  # Gets historical only
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

        # Use TestStore for realistic data and interactions
        self.test_store = TestStore()
        self.user_resolver = self.test_store.user_resolver
        self.redis_cache = NullRedisCache()

        self.mock_gemini_client = Mock(spec=AIClient)
        self.mock_gemma_client = Mock(spec=AIClient)
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.test_store,
            gemini_client=self.mock_gemini_client,
            gemma_client=self.mock_gemma_client,
            user_resolver=self.user_resolver,
            redis_cache=self.redis_cache,
        )
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids

    async def test_daily_summary_cache_prevents_redundant_ai_calls(self):
        """Test that repeated daily summary requests hit Redis cache instead of making AI calls."""
        test_date = date(1905, 3, 3)

        self.mock_gemini_client.generate_content = AsyncMock(
            return_value=DailySummaries(
                summaries=[
                    UserSummary(
                        user_id=self.physicist_ids["Einstein"],
                        summary="Einstein discussed photoelectric effect and quantum theory",
                    )
                ]
            )
        )

        with patch("memory_manager.datetime") as mock_datetime:
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # First call populates cache via async rebuild
            mock_datetime.now.return_value = datetime(1905, 3, 3, 11, 0, tzinfo=timezone.utc)
            await self.memory_manager._daily_summary(self.physics_guild_id, test_date)
            await asyncio.sleep(0.1)
            self.mock_gemini_client.generate_content.assert_called_once()

            # Reset mock to track subsequent calls
            self.mock_gemini_client.generate_content.reset_mock()

            # Act - Multiple calls within fresh window, all should hit cache
            mock_datetime.now.return_value = datetime(1905, 3, 3, 11, 30, tzinfo=timezone.utc)
            await self.memory_manager._daily_summary(self.physics_guild_id, test_date)
            await self.memory_manager._daily_summary(self.physics_guild_id, test_date)
            await self.memory_manager._daily_summary(self.physics_guild_id, test_date)

            # Assert - No AI calls after initial rebuild
            self.mock_gemini_client.generate_content.assert_not_called()

    async def test_context_merge_cache_prevents_redundant_ai_calls(self):
        """Test that identical context merge requests hit cache."""
        # Arrange
        facts = "Test facts"
        current_day = "Current day summary"
        historical = "Historical summary"
        user_id = self.physicist_ids["Einstein"]

        self.mock_gemma_client.generate_content = AsyncMock(return_value=MemoryContext(context="Merged context"))

        # Act - Make multiple identical merge requests
        daily_summaries = {
            datetime.now(timezone.utc).date(): current_day,
            datetime.now(timezone.utc).date() - timedelta(days=1): historical,
        }
        await self.memory_manager._merge_context(self.physics_guild_id, user_id, facts, daily_summaries)
        await self.memory_manager._merge_context(self.physics_guild_id, user_id, facts, daily_summaries)
        await self.memory_manager._merge_context(self.physics_guild_id, user_id, facts, daily_summaries)

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
        daily_summaries = {
            datetime.now(timezone.utc).date(): current_day,
            datetime.now(timezone.utc).date() - timedelta(days=1): historical,
        }
        await self.memory_manager._merge_context(self.physics_guild_id, user_id, original_facts, daily_summaries)
        await self.memory_manager._merge_context(self.physics_guild_id, user_id, updated_facts, daily_summaries)

        # Assert - Should make two AI calls since content changed
        self.assertEqual(self.mock_gemma_client.generate_content.call_count, 2)


class TestMemoryManagerDateBoundaries(unittest.IsolatedAsyncioTestCase):
    """Test edge cases around date boundaries and timezone handling."""

    def setUp(self):
        self.telemetry = NullTelemetry()

        # Use TestStore for realistic data and interactions
        self.test_store = TestStore()
        self.user_resolver = self.test_store.user_resolver
        self.redis_cache = NullRedisCache()

        self.mock_gemini_client = Mock(spec=AIClient)
        self.mock_gemma_client = Mock(spec=AIClient)
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.test_store,
            gemini_client=self.mock_gemini_client,
            gemma_client=self.mock_gemma_client,
            user_resolver=self.user_resolver,
            redis_cache=self.redis_cache,
        )
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids

    async def test_date_transition_cache_behavior(self):
        """Test cache behavior when date transitions from current to historical."""
        # Arrange - Use real physics conversation data from Tuesday, March 4th
        transition_date = date(1905, 3, 4)  # Tuesday with Marie Curie and Einstein discussions
        einstein_id = self.physicist_ids["Einstein"]
        expected_summaries = {einstein_id: "Einstein discussed relativity and mass-energy equivalence"}

        self.mock_gemini_client.generate_content = AsyncMock(
            return_value=DailySummaries(
                summaries=[
                    UserSummary(
                        user_id=einstein_id,
                        summary="Einstein discussed relativity and mass-energy equivalence",
                    )
                ]
            )
        )

        with patch("memory_manager.datetime") as mock_datetime:
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # First call when date is "current" — cold start returns {} and triggers async rebuild
            mock_datetime.now.return_value = datetime(1905, 3, 4, 23, 59, tzinfo=timezone.utc)
            result1 = await self.memory_manager._daily_summary(self.physics_guild_id, transition_date)
            self.assertEqual(result1, {})

            # Wait for async rebuild to populate Redis
            await asyncio.sleep(0.1)
            redis_data = await self.redis_cache.get_daily_summary(self.physics_guild_id, transition_date)
            self.assertIsNotNone(redis_data)
            self.assertEqual(redis_data[0], expected_summaries)

            self.mock_gemini_client.generate_content.reset_mock()

            # Second call after midnight — date is now "historical", goes through DB path
            mock_datetime.now.return_value = datetime(1905, 3, 5, 0, 1, tzinfo=timezone.utc)
            result2 = await self.memory_manager._daily_summary(self.physics_guild_id, transition_date)
            self.assertEqual(result2, expected_summaries)

            # Historical path makes exactly one AI call for this date
            self.mock_gemini_client.generate_content.assert_called_once()

            # Verify data persisted to DB via historical path
            db_data = await self.test_store.get_daily_summaries(self.physics_guild_id, transition_date)
            self.assertEqual(db_data, expected_summaries)

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

        with patch("memory_manager.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(1905, 3, 3, 12, 0, tzinfo=timezone.utc)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # Act
            result = await self.memory_manager._daily_summary(self.physics_guild_id, future_date)

            # Assert - Should handle future dates gracefully (no messages = empty summary)
            self.assertEqual(result, {})
            self.mock_gemini_client.generate_content.assert_not_called()


class TestMemoryManagerDatabaseBacked(unittest.IsolatedAsyncioTestCase):
    """Test database-backed daily summaries functionality."""

    def setUp(self):
        self.telemetry = NullTelemetry()

        # Use TestStore for realistic database simulation
        self.test_store = TestStore()
        self.user_resolver = self.test_store.user_resolver

        # Mock AI clients only
        self.mock_gemini_client = Mock(spec=AIClient)
        self.mock_gemma_client = Mock(spec=AIClient)

        # Component under test
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.test_store,
            gemini_client=self.mock_gemini_client,
            gemma_client=self.mock_gemma_client,
            user_resolver=self.user_resolver,
            redis_cache=NullRedisCache(),
        )

        # Test data
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids

    async def test_historical_date_database_hit(self):
        """Test that historical dates use database and avoid LLM calls when summaries exist."""
        # Arrange - Use a date with messages and pre-populate summaries
        physics_date = date(1905, 3, 4)  # March 4th has real physics messages in TestStore
        einstein_id = self.physicist_ids["Einstein"]
        expected_summaries = {einstein_id: "Einstein discussed photoelectric effect"}

        # Pre-save summaries to TestStore to simulate existing database data
        await self.test_store.save_daily_summaries(self.physics_guild_id, physics_date, expected_summaries)

        with patch("memory_manager.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(1905, 3, 5, 12, 0, tzinfo=timezone.utc)  # Next day
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # Act
            result = await self.memory_manager._daily_summary(self.physics_guild_id, physics_date)

            # Assert
            self.assertEqual(result, expected_summaries)
            self.mock_gemini_client.generate_content.assert_not_called()  # No LLM call needed

    async def test_historical_date_database_miss_generates_and_saves(self):
        """Test that historical dates generate summaries and save to database when missing."""
        # Arrange - Use date with real messages in TestStore but no pre-saved summaries
        physics_discussion_date = date(1905, 3, 3)  # March 3rd has real Einstein photoelectric effect discussion
        einstein_id = self.physicist_ids["Einstein"]
        generated_summaries = {einstein_id: "Einstein discussed photoelectric effect"}

        # Mock AI response for generating summaries
        self.mock_gemini_client.generate_content = AsyncMock(
            return_value=DailySummaries(
                summaries=[UserSummary(user_id=einstein_id, summary="Einstein discussed photoelectric effect")]
            )
        )

        with patch("memory_manager.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(1905, 3, 4, 12, 0, tzinfo=timezone.utc)  # Next day
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # Act
            result = await self.memory_manager._daily_summary(self.physics_guild_id, physics_discussion_date)

            # Assert
            self.assertEqual(result, generated_summaries)
            self.mock_gemini_client.generate_content.assert_called_once()  # LLM call made

            # Verify summaries were saved to TestStore and can be retrieved
            saved_summaries = await self.test_store.get_daily_summaries(self.physics_guild_id, physics_discussion_date)
            self.assertEqual(saved_summaries, generated_summaries)

    async def test_historical_date_no_messages_skips_database_and_llm(self):
        """Test that historical dates with no messages return empty without DB or LLM calls."""
        # Arrange - Use March 2nd which has no messages in TestStore
        historical_date = date(1905, 3, 2)  # Date with no physics discussions

        with patch("memory_manager.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(1905, 3, 3, 12, 0, tzinfo=timezone.utc)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # Act
            result = await self.memory_manager._daily_summary(self.physics_guild_id, historical_date)

            # Assert
            self.assertEqual(result, {})
            self.mock_gemini_client.generate_content.assert_not_called()  # Skip LLM

            # Verify TestStore correctly reports no messages for this date
            has_messages = await self.test_store.has_chat_messages_for_date(self.physics_guild_id, historical_date)
            self.assertFalse(has_messages)

    async def test_current_day_returns_empty_on_cache_miss(self):
        """Test that current day returns empty dict on cache miss and triggers async rebuild."""
        # Arrange - Use date with real messages in TestStore
        today = date(1905, 3, 3)  # March 3rd has real Einstein discussion
        einstein_id = self.physicist_ids["Einstein"]

        # Mock AI response for generating summaries
        self.mock_gemini_client.generate_content = AsyncMock(
            return_value=DailySummaries(
                summaries=[UserSummary(user_id=einstein_id, summary="Einstein discussed photoelectric effect")]
            )
        )

        with patch("memory_manager.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(1905, 3, 3, 12, 0, tzinfo=timezone.utc)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # Act - First call returns empty (cold start)
            result = await self.memory_manager._daily_summary(self.physics_guild_id, today)

            # Assert - Returns empty immediately
            self.assertEqual(result, {})

            # Wait for async rebuild to populate Redis
            await asyncio.sleep(0.1)
            self.mock_gemini_client.generate_content.assert_called_once()

    async def test_database_persistence_integration(self):
        """Test that summaries are properly saved and retrieved from database via Store."""
        # Arrange - Use date with real messages in TestStore
        physics_date = date(1905, 3, 3)  # March 3rd has real physics discussion
        einstein_id = self.physicist_ids["Einstein"]
        generated_summaries = {einstein_id: "Einstein discussed photoelectric effect"}

        # Mock AI response for generating summaries
        self.mock_gemini_client.generate_content = AsyncMock(
            return_value=DailySummaries(
                summaries=[UserSummary(user_id=einstein_id, summary="Einstein discussed photoelectric effect")]
            )
        )

        with patch("memory_manager.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(1905, 3, 4, 12, 0, tzinfo=timezone.utc)  # Next day
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # Act - First call should generate and save summaries
            result1 = await self.memory_manager._daily_summary(self.physics_guild_id, physics_date)

            # Second call should retrieve from database (no additional AI call)
            result2 = await self.memory_manager._daily_summary(self.physics_guild_id, physics_date)

            # Assert
            self.assertEqual(result1, generated_summaries)
            self.assertEqual(result2, generated_summaries)
            self.mock_gemini_client.generate_content.assert_called_once()  # Only one AI call

            # Verify persistence integration - summaries stored and retrievable
            stored_summaries = await self.test_store.get_daily_summaries(self.physics_guild_id, physics_date)
            self.assertEqual(stored_summaries, generated_summaries)


class TestMemoryManagerStalenessLogic(unittest.IsolatedAsyncioTestCase):
    """Test staleness-based caching with async rebuild functionality."""

    def setUp(self):
        self.telemetry = NullTelemetry()
        self.test_store = TestStore()
        self.user_resolver = self.test_store.user_resolver
        self.redis_cache = NullRedisCache()
        self.mock_gemini_client = Mock(spec=AIClient)
        self.mock_gemma_client = Mock(spec=AIClient)
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.test_store,
            gemini_client=self.mock_gemini_client,
            gemma_client=self.mock_gemma_client,
            user_resolver=self.user_resolver,
            redis_cache=self.redis_cache,
        )
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids
        self.today = date(1905, 3, 3)
        self.einstein_id = self.physicist_ids["Einstein"]

    async def _populate_cache_via_rebuild(self, mock_datetime, at_time, ai_response):
        """Populate Redis cache using the memory manager's own async rebuild mechanism."""
        mock_datetime.now.return_value = at_time
        self.mock_gemini_client.generate_content = AsyncMock(return_value=ai_response)
        await self.memory_manager._daily_summary(self.physics_guild_id, self.today)
        await asyncio.sleep(0.1)

    def _make_summaries_response(self, summaries: dict[int, str]) -> DailySummaries:
        return DailySummaries(
            summaries=[UserSummary(user_id=uid, summary=text) for uid, text in summaries.items()]
        )

    async def test_fresh_cache_returns_immediately_no_rebuild(self):
        """Test that cache less than 1 hour old returns immediately without triggering rebuild."""
        with patch("memory_manager.datetime") as mock_datetime:
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # Populate cache at 10:00 via async rebuild
            await self._populate_cache_via_rebuild(
                mock_datetime,
                at_time=datetime(1905, 3, 3, 10, 0, tzinfo=timezone.utc),
                ai_response=self._make_summaries_response(
                    {self.einstein_id: "Einstein discussed photoelectric effect"}
                ),
            )
            self.mock_gemini_client.generate_content.reset_mock()

            # Act - Call at 10:30 (30 min later, still fresh)
            mock_datetime.now.return_value = datetime(1905, 3, 3, 10, 30, tzinfo=timezone.utc)
            result = await self.memory_manager._daily_summary(self.physics_guild_id, self.today)

            # Assert - Returns cached data, no new AI call
            self.assertEqual(result, {self.einstein_id: "Einstein discussed photoelectric effect"})
            self.mock_gemini_client.generate_content.assert_not_called()

    async def test_stale_cache_returns_immediately_triggers_async_rebuild(self):
        """Test that cache >= 1 hour old returns stale data and triggers async rebuild."""
        with patch("memory_manager.datetime") as mock_datetime:
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # Populate cache at 9:00
            await self._populate_cache_via_rebuild(
                mock_datetime,
                at_time=datetime(1905, 3, 3, 9, 0, tzinfo=timezone.utc),
                ai_response=self._make_summaries_response(
                    {self.einstein_id: "Original summary"}
                ),
            )

            # Prepare fresh response for the rebuild
            self.mock_gemini_client.generate_content = AsyncMock(
                return_value=self._make_summaries_response(
                    {self.einstein_id: "Fresh summary"}
                )
            )

            # Act - Call at 10:30 (1.5h later, stale)
            mock_datetime.now.return_value = datetime(1905, 3, 3, 10, 30, tzinfo=timezone.utc)
            result = await self.memory_manager._daily_summary(self.physics_guild_id, self.today)

            # Assert - Returns stale data immediately
            self.assertEqual(result, {self.einstein_id: "Original summary"})

            # Wait for async rebuild, then verify fresh data is available
            await asyncio.sleep(0.1)
            self.mock_gemini_client.generate_content.assert_called_once()
            fresh = await self.memory_manager._daily_summary(self.physics_guild_id, self.today)
            self.assertEqual(fresh, {self.einstein_id: "Fresh summary"})

    async def test_very_stale_cache_still_returns_data_triggers_rebuild(self):
        """Test that even very stale cache returns data and triggers rebuild (no sync blocking)."""
        with patch("memory_manager.datetime") as mock_datetime:
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # Populate cache at 9:00
            await self._populate_cache_via_rebuild(
                mock_datetime,
                at_time=datetime(1905, 3, 3, 9, 0, tzinfo=timezone.utc),
                ai_response=self._make_summaries_response(
                    {self.einstein_id: "Very old summary"}
                ),
            )

            # Prepare fresh response for rebuild
            self.mock_gemini_client.generate_content = AsyncMock(
                return_value=self._make_summaries_response(
                    {self.einstein_id: "Fresh summary"}
                )
            )

            # Act - Call at 16:00 (7h later, very stale)
            mock_datetime.now.return_value = datetime(1905, 3, 3, 16, 0, tzinfo=timezone.utc)
            result = await self.memory_manager._daily_summary(self.physics_guild_id, self.today)

            # Assert - Returns stale data immediately (no sync blocking)
            self.assertEqual(result, {self.einstein_id: "Very old summary"})

            await asyncio.sleep(0.1)
            self.mock_gemini_client.generate_content.assert_called_once()

    async def test_duplicate_async_rebuild_prevention_via_redis_lock(self):
        """Test that multiple concurrent requests don't trigger duplicate async rebuilds."""
        with patch("memory_manager.datetime") as mock_datetime:
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # Populate cache at 9:00
            await self._populate_cache_via_rebuild(
                mock_datetime,
                at_time=datetime(1905, 3, 3, 9, 0, tzinfo=timezone.utc),
                ai_response=self._make_summaries_response(
                    {self.einstein_id: "Stale summary"}
                ),
            )

            # Slow AI response for rebuild to test concurrency
            async def slow_generate(*args, **kwargs):
                await asyncio.sleep(0.2)
                return self._make_summaries_response({self.einstein_id: "Fresh summary"})

            self.mock_gemini_client.generate_content = AsyncMock(side_effect=slow_generate)

            # Act - 3 concurrent calls at 10:30 (stale), each triggers rebuild attempt
            mock_datetime.now.return_value = datetime(1905, 3, 3, 10, 30, tzinfo=timezone.utc)
            results = await asyncio.gather(
                self.memory_manager._daily_summary(self.physics_guild_id, self.today),
                self.memory_manager._daily_summary(self.physics_guild_id, self.today),
                self.memory_manager._daily_summary(self.physics_guild_id, self.today),
            )

            # Assert - All return stale data immediately
            for result in results:
                self.assertEqual(result, {self.einstein_id: "Stale summary"})

            # Wait for rebuild, only one AI call due to lock deduplication
            await asyncio.sleep(0.3)
            self.assertEqual(self.mock_gemini_client.generate_content.call_count, 1)

    async def test_async_rebuild_releases_lock_on_completion(self):
        """Test that completed async rebuild releases the lock, allowing future rebuilds."""
        with patch("memory_manager.datetime") as mock_datetime:
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # Populate cache at 9:00
            await self._populate_cache_via_rebuild(
                mock_datetime,
                at_time=datetime(1905, 3, 3, 9, 0, tzinfo=timezone.utc),
                ai_response=self._make_summaries_response(
                    {self.einstein_id: "Original summary"}
                ),
            )

            # First stale rebuild at 10:30
            self.mock_gemini_client.generate_content = AsyncMock(
                return_value=self._make_summaries_response(
                    {self.einstein_id: "First refresh"}
                )
            )
            mock_datetime.now.return_value = datetime(1905, 3, 3, 10, 30, tzinfo=timezone.utc)
            await self.memory_manager._daily_summary(self.physics_guild_id, self.today)
            await asyncio.sleep(0.1)

            # Second stale rebuild at 12:00 — should succeed (lock was released)
            self.mock_gemini_client.generate_content = AsyncMock(
                return_value=self._make_summaries_response(
                    {self.einstein_id: "Second refresh"}
                )
            )
            mock_datetime.now.return_value = datetime(1905, 3, 3, 12, 0, tzinfo=timezone.utc)
            await self.memory_manager._daily_summary(self.physics_guild_id, self.today)
            await asyncio.sleep(0.1)

            # Assert - Second rebuild ran (lock was released after first)
            self.mock_gemini_client.generate_content.assert_called_once()

    async def test_async_rebuild_handles_blocked_exception(self):
        """Test that async rebuild handles BlockedException gracefully and releases lock."""
        with patch("memory_manager.datetime") as mock_datetime:
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # Populate cache at 9:00
            await self._populate_cache_via_rebuild(
                mock_datetime,
                at_time=datetime(1905, 3, 3, 9, 0, tzinfo=timezone.utc),
                ai_response=self._make_summaries_response(
                    {self.einstein_id: "Stale summary"}
                ),
            )

            # Rebuild will fail with BlockedException
            self.mock_gemini_client.generate_content = AsyncMock(
                side_effect=BlockedException(reason="PROHIBITED_CONTENT")
            )

            # Act - Stale call triggers rebuild that fails
            mock_datetime.now.return_value = datetime(1905, 3, 3, 10, 30, tzinfo=timezone.utc)
            result = await self.memory_manager._daily_summary(self.physics_guild_id, self.today)
            self.assertEqual(result, {self.einstein_id: "Stale summary"})
            await asyncio.sleep(0.1)

            # Assert - Lock released despite error: a new rebuild can proceed
            self.mock_gemini_client.generate_content = AsyncMock(
                return_value=self._make_summaries_response(
                    {self.einstein_id: "Recovered summary"}
                )
            )
            mock_datetime.now.return_value = datetime(1905, 3, 3, 11, 30, tzinfo=timezone.utc)
            await self.memory_manager._daily_summary(self.physics_guild_id, self.today)
            await asyncio.sleep(0.1)
            self.mock_gemini_client.generate_content.assert_called_once()

    async def test_async_rebuild_handles_generic_exception(self):
        """Test that async rebuild handles generic exceptions gracefully and releases lock."""
        with patch("memory_manager.datetime") as mock_datetime:
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # Populate cache at 9:00
            await self._populate_cache_via_rebuild(
                mock_datetime,
                at_time=datetime(1905, 3, 3, 9, 0, tzinfo=timezone.utc),
                ai_response=self._make_summaries_response(
                    {self.einstein_id: "Stale summary"}
                ),
            )

            # Rebuild will fail with generic exception
            self.mock_gemini_client.generate_content = AsyncMock(
                side_effect=Exception("Network error")
            )

            # Act - Stale call triggers rebuild that fails
            mock_datetime.now.return_value = datetime(1905, 3, 3, 10, 30, tzinfo=timezone.utc)
            result = await self.memory_manager._daily_summary(self.physics_guild_id, self.today)
            self.assertEqual(result, {self.einstein_id: "Stale summary"})
            await asyncio.sleep(0.1)

            # Assert - Lock released despite error: a new rebuild can proceed
            self.mock_gemini_client.generate_content = AsyncMock(
                return_value=self._make_summaries_response(
                    {self.einstein_id: "Recovered summary"}
                )
            )
            mock_datetime.now.return_value = datetime(1905, 3, 3, 11, 30, tzinfo=timezone.utc)
            await self.memory_manager._daily_summary(self.physics_guild_id, self.today)
            await asyncio.sleep(0.1)
            self.mock_gemini_client.generate_content.assert_called_once()

    async def test_async_rebuild_triggers_memory_recompute_for_affected_users(self):
        """Test that async rebuild calls get_memories to precompute merged contexts for affected users."""
        bohr_id = self.physicist_ids["Bohr"]

        with patch("memory_manager.datetime") as mock_datetime:
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # Populate cache at 9:00
            await self._populate_cache_via_rebuild(
                mock_datetime,
                at_time=datetime(1905, 3, 3, 9, 0, tzinfo=timezone.utc),
                ai_response=self._make_summaries_response(
                    {self.einstein_id: "Old summary", bohr_id: "Old Bohr summary"}
                ),
            )

            # Prepare fresh response for rebuild
            self.mock_gemini_client.generate_content = AsyncMock(
                return_value=self._make_summaries_response(
                    {self.einstein_id: "Fresh summary", bohr_id: "Fresh Bohr summary"}
                )
            )

            with patch.object(
                self.memory_manager, "get_memories", new_callable=AsyncMock
            ) as mock_get_memories:
                mock_get_memories.return_value = {}

                # Act - Stale call triggers rebuild
                mock_datetime.now.return_value = datetime(1905, 3, 3, 10, 30, tzinfo=timezone.utc)
                await self.memory_manager._daily_summary(self.physics_guild_id, self.today)
                await asyncio.sleep(0.1)

                # Assert - get_memories called with affected user IDs
                mock_get_memories.assert_called_once()
                call_args = mock_get_memories.call_args
                self.assertEqual(call_args[0][0], self.physics_guild_id)
                self.assertSetEqual(set(call_args[0][1]), {self.einstein_id, bohr_id})


class TestMemoryManagerAliasExtraction(unittest.IsolatedAsyncioTestCase):
    """Test alias extraction from factual memory for identity resolution."""

    def setUp(self):
        self.telemetry = NullTelemetry()
        self.test_store = TestStore()
        self.user_resolver = self.test_store.user_resolver
        self.redis_cache = NullRedisCache()
        self.mock_gemini_client = Mock(spec=AIClient)
        self.mock_gemma_client = Mock(spec=AIClient)
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.test_store,
            gemini_client=self.mock_gemini_client,
            gemma_client=self.mock_gemma_client,
            user_resolver=self.user_resolver,
            redis_cache=self.redis_cache,
        )
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids

    async def test_extract_aliases_returns_aliases_for_each_user(self):
        """Test that aliases are extracted concurrently for all users with facts."""
        einstein_id = self.physicist_ids["Einstein"]
        bohr_id = self.physicist_ids["Bohr"]

        async def gemma_side_effect(message, **kwargs):
            if "Albert" in message:
                return UserAliases(aliases=["Albert"])
            return UserAliases(aliases=["Niels"])

        self.mock_gemma_client.generate_content = AsyncMock(side_effect=gemma_side_effect)

        user_facts = {
            einstein_id: "He is Albert, a theoretical physicist",
            bohr_id: "He is Niels, works on atomic structure",
        }
        result = await self.memory_manager._extract_aliases(user_facts)

        self.assertEqual(result[einstein_id], ["Albert"])
        self.assertEqual(result[bohr_id], ["Niels"])
        self.assertEqual(self.mock_gemma_client.generate_content.call_count, 2)

    async def test_extract_aliases_caches_by_facts_hash(self):
        """Test that repeated extraction with same facts hits cache."""
        einstein_id = self.physicist_ids["Einstein"]
        facts = "He is Albert, a theoretical physicist"

        self.mock_gemma_client.generate_content = AsyncMock(
            return_value=UserAliases(aliases=["Albert"])
        )

        await self.memory_manager._extract_aliases({einstein_id: facts})
        await self.memory_manager._extract_aliases({einstein_id: facts})

        self.mock_gemma_client.generate_content.assert_called_once()

    async def test_extract_aliases_cache_invalidates_on_facts_change(self):
        """Test that different facts produce a cache miss."""
        einstein_id = self.physicist_ids["Einstein"]

        self.mock_gemma_client.generate_content = AsyncMock(
            return_value=UserAliases(aliases=["Albert"])
        )

        await self.memory_manager._extract_aliases({einstein_id: "He is Albert"})
        await self.memory_manager._extract_aliases({einstein_id: "He is Albert Einstein, also known as Al"})

        self.assertEqual(self.mock_gemma_client.generate_content.call_count, 2)

    async def test_extract_aliases_empty_input_returns_empty(self):
        """Test that empty user_facts returns empty dict."""
        result = await self.memory_manager._extract_aliases({})
        self.assertEqual(result, {})
        self.mock_gemma_client.generate_content.assert_not_called()

    async def test_extract_aliases_partial_failure_returns_successful(self):
        """Test that one user's extraction failure doesn't block others."""
        einstein_id = self.physicist_ids["Einstein"]
        bohr_id = self.physicist_ids["Bohr"]

        call_count = 0

        async def gemma_side_effect(message, **kwargs):
            nonlocal call_count
            call_count += 1
            if "fails" in message:
                raise ValueError("AI service unavailable")
            return UserAliases(aliases=["Albert"])

        self.mock_gemma_client.generate_content = AsyncMock(side_effect=gemma_side_effect)

        user_facts = {
            einstein_id: "He is Albert",
            bohr_id: "This one fails",
        }
        result = await self.memory_manager._extract_aliases(user_facts)

        self.assertIn(einstein_id, result)
        self.assertEqual(result[einstein_id], ["Albert"])
        self.assertNotIn(bohr_id, result)

    async def test_daily_summaries_include_aliases_in_prompt(self):
        """Test that _create_daily_summaries injects aliases into the target_users XML."""
        einstein_id = self.physicist_ids["Einstein"]
        test_date = date(1905, 3, 3)

        # Pre-populate facts for Einstein
        self.test_store._user_facts[einstein_id] = "He is Albert, a theoretical physicist"

        # Mock alias extraction
        self.mock_gemma_client.generate_content = AsyncMock(
            return_value=UserAliases(aliases=["Albert"])
        )

        # Mock daily summary generation
        self.mock_gemini_client.generate_content = AsyncMock(
            return_value=DailySummaries(
                summaries=[UserSummary(user_id=einstein_id, summary="Einstein discussed physics")]
            )
        )

        await self.memory_manager._create_daily_summaries(self.physics_guild_id, test_date)

        # Check that the Gemini prompt contains the also_known_as tag
        gemini_call_args = self.mock_gemini_client.generate_content.call_args
        prompt = gemini_call_args[1]["message"]
        self.assertIn("<also_known_as>Albert</also_known_as>", prompt)

    async def test_extract_aliases_concurrent_processing(self):
        """Test that alias extraction for multiple users happens concurrently."""

        async def slow_extract(message, **kwargs):
            await asyncio.sleep(0.1)
            return UserAliases(aliases=["Name"])

        self.mock_gemma_client.generate_content = AsyncMock(side_effect=slow_extract)

        user_facts = {
            self.physicist_ids["Einstein"]: "Facts A",
            self.physicist_ids["Bohr"]: "Facts B",
            self.physicist_ids["Planck"]: "Facts C",
        }

        start_time = asyncio.get_event_loop().time()
        await self.memory_manager._extract_aliases(user_facts)
        elapsed = asyncio.get_event_loop().time() - start_time

        self.assertEqual(self.mock_gemma_client.generate_content.call_count, 3)
        self.assertLess(elapsed, 0.2)


if __name__ == "__main__":
    unittest.main()
