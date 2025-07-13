"""
Tests for the TestStore test double to verify it correctly represents the physics chat history.
"""

import unittest
from datetime import date, datetime, timezone
from test_store import TestStore


class TestTestStore(unittest.IsolatedAsyncioTestCase):
    """Test the TestStore test double functionality."""
    
    def setUp(self):
        self.test_store = TestStore()
    
    def test_physics_guild_setup(self):
        """Test that TestStore is properly configured for physics guild."""
        self.assertEqual(self.test_store.physics_guild_id, 19001930)
        self.assertIn("Einstein", self.test_store.physicist_ids)
        self.assertIn("Planck", self.test_store.physicist_ids)
        self.assertIn("Bohr", self.test_store.physicist_ids)
    
    def test_total_message_count(self):
        """Test that all physics chat messages are loaded."""
        # Should have 60 messages from the full physics chat history (10+8+8+8+7+8+11)
        self.assertEqual(self.test_store.total_message_count, 60)
    
    def test_date_range(self):
        """Test that the chat history spans the expected date range."""
        start_date, end_date = self.test_store.date_range
        self.assertEqual(start_date, date(1905, 3, 3))  # Monday
        self.assertEqual(end_date, date(1905, 3, 9))    # Sunday
    
    async def test_get_messages_for_first_day(self):
        """Test getting messages for the first day (Monday, March 3rd)."""
        messages = await self.test_store.get_chat_messages_for_date(
            self.test_store.physics_guild_id, 
            date(1905, 3, 3)
        )
        
        # Monday should have 10 messages
        self.assertEqual(len(messages), 10)
        
        # First message should be Einstein's opening
        first_message = messages[0]
        self.assertEqual(first_message.user_id, self.test_store.physicist_ids["Einstein"])
        self.assertIn("photoelectric effect", first_message.message_text)
        self.assertEqual(first_message.timestamp.hour, 9)
        self.assertEqual(first_message.timestamp.minute, 15)
    
    async def test_get_messages_for_last_day(self):
        """Test getting messages for the last day (Sunday, March 9th)."""
        messages = await self.test_store.get_chat_messages_for_date(
            self.test_store.physics_guild_id,
            date(1905, 3, 9)
        )
        
        # Sunday should have 11 messages (evening reflection)
        self.assertEqual(len(messages), 11)
        
        # Last message should be Bohr's closing
        last_message = messages[-1]
        self.assertEqual(last_message.user_id, self.test_store.physicist_ids["Bohr"])
        self.assertIn("quantum world is strange", last_message.message_text)
    
    async def test_get_messages_empty_date(self):
        """Test getting messages for a date with no activity."""
        messages = await self.test_store.get_chat_messages_for_date(
            self.test_store.physics_guild_id,
            date(1905, 3, 10)  # Day after the chat ended
        )
        
        self.assertEqual(len(messages), 0)
    
    async def test_get_messages_wrong_guild(self):
        """Test getting messages for wrong guild returns empty."""
        messages = await self.test_store.get_chat_messages_for_date(
            12345,  # Wrong guild ID
            date(1905, 3, 3)
        )
        
        self.assertEqual(len(messages), 0)
    
    def test_active_physicists_monday(self):
        """Test getting active physicist names for Monday."""
        active_physicists = self.test_store.get_active_physicists_for_date(date(1905, 3, 3))
        
        # Monday had Einstein, Planck, Bohr, Thomson, Rutherford, Schrödinger, Heisenberg, Born
        expected_physicists = {"Einstein", "Planck", "Bohr", "Thomson", "Rutherford", "Schrödinger", "Heisenberg", "Born"}
        self.assertEqual(set(active_physicists), expected_physicists)
    
    def test_active_physicists_tuesday(self):
        """Test getting active physicist names for Tuesday."""
        active_physicists = self.test_store.get_active_physicists_for_date(date(1905, 3, 4))
        
        # Tuesday had Curie, Einstein, Lorentz, Minkowski, Planck, Rutherford, Bohr
        expected_physicists = {"Curie", "Einstein", "Lorentz", "Minkowski", "Planck", "Rutherford", "Bohr"}
        self.assertEqual(set(active_physicists), expected_physicists)
    
    def test_message_count_by_date(self):
        """Test message count distribution across the week."""
        # Monday: 10 messages (busy opening day)
        self.assertEqual(self.test_store.get_message_count_for_date(date(1905, 3, 3)), 10)
        
        # Tuesday: 8 messages (continued discussions)
        self.assertEqual(self.test_store.get_message_count_for_date(date(1905, 3, 4)), 8)
        
        # Wednesday: 8 messages (matrix vs wave mechanics debate)
        self.assertEqual(self.test_store.get_message_count_for_date(date(1905, 3, 5)), 8)
        
        # Thursday: 8 messages (experimental results and matter waves)
        self.assertEqual(self.test_store.get_message_count_for_date(date(1905, 3, 6)), 8)
        
        # Friday: 7 messages (general relativity discussions)
        self.assertEqual(self.test_store.get_message_count_for_date(date(1905, 3, 7)), 7)
        
        # Saturday: 8 messages (measurement problem and interpretation)
        self.assertEqual(self.test_store.get_message_count_for_date(date(1905, 3, 8)), 8)
        
        # Sunday: 11 messages (reflective conclusions)
        self.assertEqual(self.test_store.get_message_count_for_date(date(1905, 3, 9)), 11)
        
        # Total should be 60 (10+8+8+8+7+8+11)
        total = sum(self.test_store.get_message_count_for_date(date(1905, 3, d)) for d in range(3, 10))
        self.assertEqual(total, self.test_store.total_message_count)
    
    def test_mention_processing(self):
        """Test that physicist name mentions are converted to Discord mentions."""
        # Find Einstein's message that mentions Max and Werner
        target_date = date(1905, 3, 3)
        einstein_id = self.test_store.physicist_ids["Einstein"]
        
        # Look for the specific message: "Max, Werner, I remain uncomfortable..."
        target_message = None
        for message in self.test_store._messages:
            if (message.timestamp.date() == target_date and 
                message.user_id == einstein_id and
                "remain uncomfortable" in message.message_text):
                target_message = message
                break
        
        # Should find Einstein's message
        self.assertIsNotNone(target_message, "Should find Einstein's message about being uncomfortable")
        
        # Should contain Discord mentions instead of plain names
        self.assertIn(f"<@{self.test_store.physicist_ids['Planck']}>", target_message.message_text)  # Max -> Planck
        self.assertIn(f"<@{self.test_store.physicist_ids['Heisenberg']}>", target_message.message_text)  # Werner -> Heisenberg
    
    async def test_user_facts_storage(self):
        """Test user facts storage and retrieval."""
        einstein_id = self.test_store.physicist_ids["Einstein"]
        
        # Initially no facts
        facts = await self.test_store.get_user_facts(self.test_store.physics_guild_id, einstein_id)
        self.assertIsNone(facts)
        
        # Set facts
        test_facts = "Einstein is known for relativity theory and quantum mechanics contributions"
        self.test_store.set_user_facts(self.test_store.physics_guild_id, einstein_id, test_facts)
        
        # Retrieve facts
        facts = await self.test_store.get_user_facts(self.test_store.physics_guild_id, einstein_id)
        self.assertEqual(facts, test_facts)
    
    async def test_add_message(self):
        """Test adding new messages to the store."""
        initial_count = self.test_store.total_message_count
        
        # Add a new message
        await self.test_store.add_chat_message(
            self.test_store.physics_guild_id,
            1905,
            99999,
            self.test_store.physicist_ids["Einstein"],
            "This is a test message",
            datetime(1905, 3, 10, 12, 0, tzinfo=timezone.utc)
        )
        
        # Should have one more message
        self.assertEqual(self.test_store.total_message_count, initial_count + 1)
        
        # Should be able to retrieve it
        messages = await self.test_store.get_chat_messages_for_date(
            self.test_store.physics_guild_id,
            date(1905, 3, 10)
        )
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].message_text, "This is a test message")


if __name__ == '__main__':
    unittest.main()