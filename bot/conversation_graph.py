import datetime
import logging
from typing import List, Tuple, Callable, Awaitable, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MessageNode:
    """Simplified message representation for graph operations."""
    id: int
    content: str
    author_name: str
    created_at: datetime.datetime
    reference_id: int | None = None
    embeds: List[Any] = None
    
    def __post_init__(self) -> None:
        if self.embeds is None:
            self.embeds = []


class MessageGraph:
    """
    Graph data structure for conversation messages with deduplication.
    """
    def __init__(self) -> None:
        self.nodes = {}  # message_id -> MessageNode
        self.unexplored_references = set()  # message_ids with unprocessed references
        self.temporal_frontier = set()  # message_ids at temporal exploration boundary
    
    def add_node(self, message: MessageNode) -> bool:
        """Add message node with deduplication."""
        if message.id not in self.nodes:
            self.nodes[message.id] = message
            if message.reference_id:
                self.unexplored_references.add(message.id)
            self.temporal_frontier.add(message.id)
            return True
        return False
    
    
    def mark_reference_explored(self, message_id: int) -> None:
        """Mark a message's reference as explored."""
        self.unexplored_references.discard(message_id)
    
    def remove_from_temporal_frontier(self, message_id: int) -> None:
        """Remove message from temporal exploration frontier."""
        self.temporal_frontier.discard(message_id)
    
    def get_unexplored_references(self) -> List[MessageNode]:
        """Get messages with unexplored references."""
        return [self.nodes[msg_id] for msg_id in self.unexplored_references]
    
    def get_temporal_frontier(self) -> List[MessageNode]:
        """Get messages at temporal exploration frontier, sorted by recency."""
        frontier_messages = [self.nodes[msg_id] for msg_id in self.temporal_frontier]
        return sorted(frontier_messages, key=lambda m: m.created_at, reverse=True)
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    def to_chronological_conversation(self, article_extractor: Callable[[str], str] | None = None) -> List[Tuple[str, str]]:
        """Convert graph to chronological conversation format."""
        messages = sorted(self.nodes.values(), key=lambda m: m.created_at)
        conversation_messages = []
        
        for message in messages:
            if message.content.strip():
                conversation_messages.append((message.author_name, message.content))
            
            # Process embeds for article extraction if extractor provided
            if article_extractor and message.embeds:
                for embed in message.embeds:
                    if hasattr(embed, 'url') and embed.url:
                        article_content = article_extractor(embed.url)
                        if article_content:
                            conversation_messages.append(("article", article_content))
        
        return conversation_messages


class ConversationGraphBuilder:
    """
    Builds conversation graphs using tik/tok alternating exploration.
    Abstracted to work with any message fetching interface.
    """
    
    def __init__(self, 
                 fetch_message: Callable[[int], Awaitable[MessageNode | None]],
                 fetch_history: Callable[[int | None, int], Awaitable[List[MessageNode]]]):
        """
        Initialize with message fetching abstractions.
        
        Args:
            fetch_message: async function(message_id: int) -> MessageNode | None
            fetch_history: async function(message_id: int | None, limit: int) -> List[MessageNode]
        """
        self.fetch_message = fetch_message
        self.fetch_history = fetch_history
    
    
    async def explore_references(self, graph: MessageGraph) -> bool:
        """Explore one level of reference connections."""
        references_to_explore = graph.get_unexplored_references()
        if not references_to_explore:
            return False
        
        added_any = False
        for message in references_to_explore:
            try:
                if message.reference_id:
                    referenced_msg = await self.fetch_message(message.reference_id)
                    if referenced_msg and graph.add_node(referenced_msg):
                        added_any = True
            except Exception as e:
                logger.error(f"Could not fetch referenced message {message.reference_id}: {e}", exc_info=True)
            
            graph.mark_reference_explored(message.id)
        
        return added_any
    
    async def explore_temporal_neighbors(self, graph: MessageGraph, time_threshold_minutes: int) -> bool:
        """Explore temporal neighbors with sealing."""
        frontier = graph.get_temporal_frontier()
        if not frontier:
            return False
        
        added_any = False
        time_threshold = datetime.timedelta(minutes=time_threshold_minutes)
        
        for current_message in frontier:
            try:
                # Get the next message before this one in channel history
                prev_messages = await self.fetch_history(current_message.id, 1)
                
                if prev_messages:
                    prev_message = prev_messages[0]
                    time_gap = current_message.created_at - prev_message.created_at
                    
                    if time_gap <= time_threshold:
                        if graph.add_node(prev_message):
                            added_any = True
                    
            except Exception as e:
                logger.error(f"Error exploring temporal neighbors for message {current_message.id}: {e}", exc_info=True)
            
            # Always remove from frontier after processing (success, failure, or time gap exceeded)
            graph.remove_from_temporal_frontier(current_message.id)
        
        return added_any
    
    async def get_linear_history(self, trigger_message: MessageNode, min_linear: int) -> List[MessageNode]:
        """Get guaranteed linear message history."""
        messages = [trigger_message]
        
        if min_linear > 1:
            prev_messages = await self.fetch_history(trigger_message.id, min_linear - 1)
            messages.extend(prev_messages)
        
        return messages
    
    async def build_conversation_graph(self, 
                                     trigger_message: MessageNode,
                                     min_linear: int = 10,
                                     max_total: int = 30, 
                                     time_threshold_minutes: int = 10,
                                     article_extractor: Callable[[str], str] | None = None) -> List[Tuple[str, str]]:
        """
        Build conversation graph using tik/tok alternating exploration.
        
        Args:
            trigger_message: Message that triggered the bot
            min_linear: Minimum linear messages to include regardless of time gaps
            max_total: Maximum total messages in graph
            time_threshold_minutes: Time threshold for temporal connections
        
        Returns:
            List of (author_name, content) tuples in chronological order
        """
        graph = MessageGraph()
        
        # Seed with guaranteed linear history
        linear_messages = await self.get_linear_history(trigger_message, min_linear)
        for msg in linear_messages:
            graph.add_node(msg)
        
        # Tik/tok alternating exploration
        while len(graph) < max_total:
            # TIK: Reference step - follow all unexplored references
            reference_step_added = await self.explore_references(graph)
            
            if len(graph) >= max_total:
                break
            
            # TOK: Temporal step - explore neighbors with sealing
            temporal_step_added = await self.explore_temporal_neighbors(graph, time_threshold_minutes)
            
            # Exit if neither step found new messages
            if not reference_step_added and not temporal_step_added:
                break
        
        return graph.to_chronological_conversation(article_extractor)