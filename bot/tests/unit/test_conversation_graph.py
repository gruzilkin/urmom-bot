import datetime
import unittest

from conversation_graph import ConversationGraphBuilder, MessageGraph, MessageNode


class TestMessageGraph(unittest.TestCase):
    
    def setUp(self):
        self.graph = MessageGraph()
        self.test_time = datetime.datetime.now()
    
    def test_add_node_deduplication(self):
        """Test that adding the same message twice only creates one node."""
        msg = MessageNode(
            id=1,
            content="test message",
            author_name="user1",
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
        msg_with_ref = MessageNode(
            id=1,
            content="reply message",
            author_name="user1",
            created_at=self.test_time,
            reference_id=123
        )
        
        msg_without_ref = MessageNode(
            id=2,
            content="regular message",
            author_name="user2",
            created_at=self.test_time
        )
        
        self.graph.add_node(msg_with_ref)
        self.graph.add_node(msg_without_ref)
        
        unexplored = self.graph.get_unexplored_references()
        self.assertEqual(len(unexplored), 1)
        self.assertEqual(unexplored[0].id, 1)
    
    def test_temporal_frontier_sorting(self):
        """Test that temporal frontier is sorted by recency."""
        older_msg = MessageNode(
            id=1,
            content="older",
            author_name="user1",
            created_at=self.test_time - datetime.timedelta(minutes=10)
        )
        
        newer_msg = MessageNode(
            id=2,
            content="newer",
            author_name="user2",
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
        msg1 = MessageNode(
            id=1,
            content="first message",
            author_name="user1",
            created_at=self.test_time - datetime.timedelta(minutes=5)
        )
        
        msg2 = MessageNode(
            id=2,
            content="second message",
            author_name="user2",
            created_at=self.test_time
        )
        
        self.graph.add_node(msg2)  # Add in reverse order
        self.graph.add_node(msg1)
        
        conversation = self.graph.to_chronological_conversation()
        
        self.assertEqual(len(conversation), 2)
        # Should be in chronological order
        self.assertEqual(conversation[0], ("user1", "first message"))
        self.assertEqual(conversation[1], ("user2", "second message"))


class TestConversationGraphBuilder(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        self.test_time = datetime.datetime.now()
        self.messages = {}
        
        # Create mock message fetching functions
        async def mock_fetch_message(msg_id):
            return self.messages.get(msg_id)
        
        async def mock_fetch_history(message_id, limit):
            # Simple mock: return messages older than message_id
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
            return older_messages[:limit]
        
        self.builder = ConversationGraphBuilder(
            fetch_message=mock_fetch_message,
            fetch_history=mock_fetch_history
        )
    
    def create_test_message(self, msg_id, content, minutes_ago=0, reference_id=None):
        """Helper to create test messages."""
        return MessageNode(
            id=msg_id,
            content=content,
            author_name=f"user{msg_id}",
            created_at=self.test_time - datetime.timedelta(minutes=minutes_ago),
            reference_id=reference_id
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


if __name__ == '__main__':
    unittest.main()