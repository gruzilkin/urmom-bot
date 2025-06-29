import datetime
import unittest
from unittest.mock import Mock

from conversation_graph import ConversationGraphBuilder, MessageGraph, CachedHistoryFetcher
from message_node import MessageNode
from tests.null_telemetry import NullTelemetry


class BaseMessageGraphTest(unittest.IsolatedAsyncioTestCase):
    """Base class with shared mock functionality for message graph tests."""
    
    def setUp(self):
        self.test_time = datetime.datetime.now()
        self.messages = {}
    
    def create_mock_discord_message(self, msg_id: int, content: str, author_id: int, 
                                   created_at: datetime.datetime, reference_id: int = None,
                                   mentioned_user_ids: list[int] = None) -> Mock:
        """Create a mock Discord message object."""
        message = Mock()
        message.id = msg_id
        message.content = content
        message.author = Mock()
        message.author.id = author_id
        message.created_at = created_at
        message.embeds = []
        
        # Mock mentions
        if mentioned_user_ids:
            message.mentions = [Mock(id=uid) for uid in mentioned_user_ids]
        else:
            message.mentions = []
        
        # Mock reference
        if reference_id:
            message.reference = Mock()
            message.reference.message_id = reference_id
        else:
            message.reference = None
            
        return message
    
    async def mock_fetch_message(self, msg_id):
        return self.messages.get(msg_id)
    
    async def mock_fetch_history(self, message_id):
        # Simple mock: return messages older than message_id (fixed at 100 limit)
        if not message_id:
            return []
        
        before_msg = self.messages.get(message_id)
        if not before_msg:
            return []
        
        older_messages = [
            msg for msg in self.messages.values()
            if msg.created_at < before_msg.created_at
        ]
        older_messages.sort(key=lambda m: m.created_at, reverse=True)
        return older_messages[:100]
    
    def create_test_message(self, msg_id, content, minutes_ago=0, reference_id=None):
        """Helper to create test messages."""
        return self.create_mock_discord_message(
            msg_id=msg_id,
            content=content,
            author_id=msg_id * 1000,  # Use a predictable author_id based on msg_id
            created_at=self.test_time - datetime.timedelta(minutes=minutes_ago),
            reference_id=reference_id
        )


class TestMessageGraph(BaseMessageGraphTest):
    
    def setUp(self):
        super().setUp()
        self.graph = MessageGraph()
    
    def test_add_node_deduplication(self):
        """Test that adding the same message twice only creates one node."""
        msg = self.create_mock_discord_message(
            msg_id=1,
            content="test message",
            author_id=1000,
            created_at=self.test_time
        )
        
        # First add should succeed
        self.assertTrue(self.graph.add_node(msg))
        self.assertEqual(len(self.graph), 1)
        
        # Second add should fail (deduplication)
        self.assertFalse(self.graph.add_node(msg))
        self.assertEqual(len(self.graph), 1)
    
    def test_reference_tracking(self):
        """Test that messages with references are tracked for exploration."""
        msg_with_ref = self.create_mock_discord_message(
            msg_id=1,
            content="reply message",
            author_id=1000,
            created_at=self.test_time,
            reference_id=123
        )
        
        msg_without_ref = self.create_mock_discord_message(
            msg_id=2,
            content="regular message",
            author_id=2000,
            created_at=self.test_time
        )
        
        self.graph.add_node(msg_with_ref)
        self.graph.add_node(msg_without_ref)
        
        unexplored = self.graph.get_unexplored_references()
        self.assertEqual(len(unexplored), 1)
        self.assertEqual(unexplored[0].id, 1)
    
    def test_temporal_frontier_sorting(self):
        """Test that temporal frontier is sorted by recency."""
        older_msg = self.create_mock_discord_message(
            msg_id=1,
            content="older",
            author_id=1000,
            created_at=self.test_time - datetime.timedelta(minutes=10)
        )
        
        newer_msg = self.create_mock_discord_message(
            msg_id=2,
            content="newer",
            author_id=2000,
            mentioned_user_ids=[],
            created_at=self.test_time
        )
        
        self.graph.add_node(older_msg)
        self.graph.add_node(newer_msg)
        
        frontier = self.graph.get_temporal_frontier()
        self.assertEqual(len(frontier), 2)
        # Should be sorted by recency (newest first)
        self.assertEqual(frontier[0].id, 2)
        self.assertEqual(frontier[1].id, 1)
    
    def test_chronological_conversation(self):
        """Test conversion to chronological conversation format."""
        msg1 = self.create_mock_discord_message(
            msg_id=1,
            content="first message",
            author_id=1000,
            created_at=self.test_time - datetime.timedelta(minutes=5)
        )
        
        msg2 = self.create_mock_discord_message(
            msg_id=2,
            content="second message",
            author_id=2000,
            created_at=self.test_time
        )
        
        self.graph.add_node(msg2)  # Add in reverse order
        self.graph.add_node(msg1)
        
        def mock_discord_to_message_node(discord_msg):
            return MessageNode(
                id=discord_msg.id,
                content=discord_msg.content,
                author_id=discord_msg.author.id,
                mentioned_user_ids=[user.id for user in discord_msg.mentions],
                created_at=discord_msg.created_at,
                reference_id=discord_msg.reference.message_id if discord_msg.reference else None
            )
        
        conversation = self.graph.to_chronological_conversation(mock_discord_to_message_node)
        
        self.assertEqual(len(conversation), 2)
        # Should be in chronological order
        self.assertEqual(conversation[0].author_id, 1000)
        self.assertEqual(conversation[0].content, "first message")
        self.assertEqual(conversation[1].author_id, 2000)
        self.assertEqual(conversation[1].content, "second message")
        # Verify timestamps are formatted correctly
        self.assertIsInstance(conversation[0].timestamp, str)
        self.assertIsInstance(conversation[1].timestamp, str)


class TestConversationGraphBuilder(BaseMessageGraphTest):
    
    def setUp(self):
        super().setUp()
        self.builder = ConversationGraphBuilder(
            fetch_message=self.mock_fetch_message,
            fetch_history=self.mock_fetch_history,
            telemetry=NullTelemetry()
        )
    
    async def test_linear_history_fetching(self):
        """Test fetching guaranteed linear history."""
        # Create chain of messages
        self.messages[3] = self.create_test_message(3, "newest", 0)
        self.messages[2] = self.create_test_message(2, "middle", 5)
        self.messages[1] = self.create_test_message(1, "oldest", 10)
        
        trigger = self.messages[3]
        linear = await self.builder.get_linear_history(trigger, 3)
        
        self.assertEqual(len(linear), 3)
        self.assertEqual(linear[0].id, 3)  # trigger message first
    
    async def test_reference_exploration(self):
        """Test exploring reference connections."""
        # Create message with reference
        self.messages[1] = self.create_test_message(1, "original", 10)
        self.messages[2] = self.create_test_message(2, "reply", 5, reference_id=1)
        
        graph = MessageGraph()
        graph.add_node(self.messages[2])
        
        # Should find and add referenced message
        added = await self.builder.explore_references(graph)
        
        self.assertTrue(added)
        self.assertEqual(len(graph), 2)
        self.assertIn(1, graph.nodes)
        self.assertIn(2, graph.nodes)
    
    async def test_temporal_sealing(self):
        """Test that large time gaps seal temporal paths."""
        # Create messages with large time gap
        self.messages[2] = self.create_test_message(2, "recent", 5)
        self.messages[1] = self.create_test_message(1, "very old", 60)  # 60 minutes ago
        
        graph = MessageGraph()
        graph.add_node(self.messages[2])
        
        # Should seal path due to large time gap (threshold = 10 minutes)
        added = await self.builder.explore_temporal_neighbors(graph, time_threshold_minutes=10)
        
        self.assertFalse(added)  # No messages added due to sealing
        self.assertEqual(len(graph), 1)
        # Message 2 should be removed from temporal frontier
        self.assertNotIn(2, graph.temporal_frontier)
    
    async def test_api_efficiency_with_200_messages(self):
        """Test that 200 messages are read efficiently with exactly 2 API calls."""
        # Create a chain of 200 messages: 200 -> 199 -> 198 -> ... -> 2 -> 1
        for i in range(1, 201):
            self.messages[i] = self.create_test_message(i, f"message {i}", minutes_ago=200-i)
        
        # Track API calls by composing existing mock with counting wrapper
        api_call_count = 0
        
        async def counting_fetch_history(message_id):
            nonlocal api_call_count
            api_call_count += 1
            return await self.mock_fetch_history(message_id)
        
        # Create new builder with counting fetch_history
        counting_builder = ConversationGraphBuilder(
            fetch_message=self.mock_fetch_message,
            fetch_history=counting_fetch_history,
            telemetry=NullTelemetry()
        )
        
        # Build conversation graph requesting all 200 messages starting from message 200
        trigger_message = self.messages[200]
        def mock_discord_to_message_node(discord_msg):
            return MessageNode(
                id=discord_msg.id,
                content=discord_msg.content,
                author_id=discord_msg.author.id,
                mentioned_user_ids=[user.id for user in discord_msg.mentions],
                created_at=discord_msg.created_at,
                reference_id=discord_msg.reference.message_id if discord_msg.reference else None
            )
        
        conversation = await counting_builder.build_conversation_graph(
            trigger_message=trigger_message,
            min_linear=10,
            max_total=200,  # Request all 200 messages
            time_threshold_minutes=300,  # Large threshold to include all messages
            discord_to_message_node_func=mock_discord_to_message_node
        )
        
        # Verify that conversation was built successfully with all 200 messages
        self.assertEqual(len(conversation), 200)
        
        # Key assertion: Should make exactly 2 API calls
        # 1. First fetch_history call gets messages 199-100 (100 messages)
        # 2. Second fetch_history call during temporal exploration gets messages 99-1 (99 messages)
        # Total: 199 messages + trigger message = 200 messages
        self.assertEqual(api_call_count, 2, 
                        f"Expected exactly 2 API calls but got {api_call_count}. "
                        f"With 200 messages, should need exactly 2 calls of 100 messages each.")


class TestCachedHistoryFetcher(BaseMessageGraphTest):
    
    def setUp(self):
        super().setUp()
        self.call_count = 0
        
        # Create mock fetch_history function that tracks call count
        async def counting_mock_fetch_history(message_id):
            self.call_count += 1
            return await self.mock_fetch_history(message_id)
        
        self.cached_fetcher = CachedHistoryFetcher(counting_mock_fetch_history, self.mock_fetch_message)
    
    async def test_intermediate_cache_population(self):
        """Test that bulk fetch populates cache for intermediate messages."""
        # Set up message chain: 5 -> 4 -> 3 -> 2 -> 1
        self.messages[5] = self.create_test_message(5, "message 5", 0)
        self.messages[4] = self.create_test_message(4, "message 4", 1)
        self.messages[3] = self.create_test_message(3, "message 3", 2)
        self.messages[2] = self.create_test_message(2, "message 2", 3)
        self.messages[1] = self.create_test_message(1, "message 1", 4)
        
        # Request for message 5 should fetch bulk messages and populate cache
        await self.cached_fetcher.get_previous_message(5)
        initial_call_count = self.call_count
        
        # Subsequent requests for messages 4 and 3 should hit cache
        result4 = await self.cached_fetcher.get_previous_message(4)
        result3 = await self.cached_fetcher.get_previous_message(3)
        
        self.assertEqual(self.call_count, initial_call_count)  # No additional API calls
        self.assertEqual(result4.id, 3)  # Previous message to 4 is 3
        self.assertEqual(result3.id, 2)  # Previous message to 3 is 2
    
    async def test_no_previous_message(self):
        """Test handling when no previous message exists."""
        # Set up single message with no older messages
        self.messages[1] = self.create_test_message(1, "message 1", 0)
        
        result = await self.cached_fetcher.get_previous_message(1)
        
        self.assertIsNone(result)
        self.assertEqual(self.call_count, 1)  # API was called
    
    async def test_nonexistent_message(self):
        """Test handling when requested message doesn't exist."""
        result = await self.cached_fetcher.get_previous_message(999)
        
        self.assertIsNone(result)
        self.assertEqual(self.call_count, 1)  # API was called
    


if __name__ == '__main__':
    unittest.main()