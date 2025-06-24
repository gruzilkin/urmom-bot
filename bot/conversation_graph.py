import datetime
import logging
from typing import List, Tuple, Callable, Awaitable, Any, Dict
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


@dataclass
class ConversationMessage:
    """Represents a message in chronological conversation format."""
    author_name: str
    content: str
    timestamp: str  # Pre-formatted LLM-friendly timestamp


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
    
    def to_chronological_conversation(self, article_extractor: Callable[[str], str] | None = None) -> List[ConversationMessage]:
        """Convert graph to chronological conversation format."""
        messages = sorted(self.nodes.values(), key=lambda m: m.created_at)
        conversation_messages = []
        
        for message in messages:
            if message.content.strip():
                conversation_messages.append(ConversationMessage(
                    author_name=message.author_name,
                    content=message.content,
                    timestamp=message.created_at.strftime('%Y-%m-%d %H:%M:%S')
                ))
            
            # Process embeds for article extraction if extractor provided
            if article_extractor and message.embeds:
                for embed in message.embeds:
                    if hasattr(embed, 'url') and embed.url:
                        article_content = article_extractor(embed.url)
                        if article_content:
                            conversation_messages.append(ConversationMessage(
                                author_name="article",
                                content=article_content,
                                timestamp=message.created_at.strftime('%Y-%m-%d %H:%M:%S')
                            ))
        
        return conversation_messages


class ConversationGraphBuilder:
    """
    Builds conversation graphs using tik/tok alternating exploration.
    Abstracted to work with any message fetching interface.
    """
    
    def __init__(self, 
                 fetch_message: Callable[[int], Awaitable[MessageNode | None]],
                 fetch_history: Callable[[int | None], Awaitable[List[MessageNode]]],
                 telemetry=None):
        """
        Initialize with message fetching abstractions.
        
        Args:
            fetch_message: async function(message_id: int) -> MessageNode | None
            fetch_history: async function(message_id: int | None) -> List[MessageNode]
            telemetry: Optional telemetry service for spans
        """
        self.fetch_message = fetch_message
        self.fetch_history = fetch_history
        self.telemetry = telemetry
        
        # Set up cached history fetching for performance optimization
        self.cached_fetcher = CachedHistoryFetcher(fetch_history, fetch_message)
        self._get_history = self.cached_fetcher.get_previous_message
    
    
    async def explore_references(self, graph: MessageGraph) -> bool:
        """Explore one level of reference connections."""
        references_to_explore = graph.get_unexplored_references()
        
        if not references_to_explore:
            return False
        
        added_any = False
        for message in references_to_explore:
            try:
                if message.reference_id:
                    # Use cached fetcher for all message fetching
                    referenced_msg = await self.cached_fetcher.get_message_by_id(message.reference_id)
                    
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
                # Get the next message before this one in channel history (cached)
                prev_message = await self._get_history(current_message.id)
                
                if prev_message:
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
        min_linear = max(min_linear, 1)
        
        # Use cached fetcher for bulk linear history to populate cache for subsequent temporal exploration
        prev_messages = await self.cached_fetcher.get_bulk_history(trigger_message.id)
        messages = [trigger_message] + prev_messages[:min_linear - 1]
        
        return messages
    
    async def build_conversation_graph(self, 
                                     trigger_message: MessageNode,
                                     min_linear: int,
                                     max_total: int, 
                                     time_threshold_minutes: int,
                                     article_extractor: Callable[[str], str] | None = None) -> List[ConversationMessage]:
        """
        Build conversation graph using tik/tok alternating exploration.
        
        Args:
            trigger_message: Message that triggered the bot
            min_linear: Minimum linear messages to include regardless of time gaps
            max_total: Maximum total messages in graph
            time_threshold_minutes: Time threshold for temporal connections
        
        Returns:
            List of ConversationMessage objects in chronological order
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
        
        # Convert to conversation format
        return graph.to_chronological_conversation(article_extractor)
    


class CachedHistoryFetcher:
    """
    Cached wrapper for fetch_history that reads in bulk but returns single messages.
    Maintains a cache of message_id -> previous message to minimize Discord API calls.
    """
    
    def __init__(self, fetch_history: Callable[[int | None], Awaitable[List[MessageNode]]], 
                 fetch_message: Callable[[int], Awaitable[MessageNode | None]]):
        """
        Initialize cached history fetcher.
        
        Args:
            fetch_history: Original fetch_history function
            fetch_message: Original fetch_message function for individual messages
        """
        self.fetch_history = fetch_history
        self.fetch_message = fetch_message
        self.cache: Dict[int, MessageNode] = {}  # message_id -> previous message
        self.message_cache: Dict[int, MessageNode] = {}  # message_id -> message (for individual fetches)
        
    async def get_previous_message(self, message_id: int) -> MessageNode | None:
        """
        Get the previous message, using cache when possible.
        
        Args:
            message_id: ID of message to get history before
            
        Returns:
            Previous message or None if not found
        """
        # Check if we have cached data for this message
        if message_id in self.cache:
            return self.cache[message_id]
        
        # Cache miss - use bulk fetch to populate cache and return first result
        bulk_messages = await self.get_bulk_history(message_id)
        
        if bulk_messages:
            return bulk_messages[0]
        else:
            return None
    
    async def get_bulk_history(self, message_id: int | None) -> List[MessageNode]:
        """
        Get bulk history messages, populating cache along the way.
        
        Args:
            message_id: ID of message to get history before (None for channel start)
            
        Returns:
            List of previous messages (fixed at 100)
        """
        # Always call the original fetch_history and populate cache with the results
        try:
            bulk_messages = await self.fetch_history(message_id)
            
            if bulk_messages:
                # Populate both caches
                # 1. Cache individual messages
                for msg in bulk_messages:
                    self.message_cache[msg.id] = msg
                
                # 2. Cache previous-message relationships
                for current_msg, prev_msg in zip(bulk_messages, bulk_messages[1:]):
                    if current_msg.id not in self.cache:
                        self.cache[current_msg.id] = prev_msg
                
                # If we have a message_id, cache the relationship from it to first result
                if message_id is not None:
                    self.cache[message_id] = bulk_messages[0]
                
            return bulk_messages
                
        except Exception as e:
            logger.error(f"Error fetching bulk history for message {message_id}: {e}", exc_info=True)
            return []
    
    async def get_message_by_id(self, message_id: int) -> MessageNode | None:
        """
        Get a specific message by ID, using cache when possible.
        
        Args:
            message_id: ID of message to fetch
            
        Returns:
            MessageNode or None if not found
        """
        # Check if message is in individual message cache
        if message_id in self.message_cache:
            return self.message_cache[message_id]
        
        # Cache miss - fetch individual message
        try:
            message = await self.fetch_message(message_id)
            if message:
                self.message_cache[message_id] = message
            return message
        except Exception as e:
            logger.error(f"Error fetching message {message_id}: {e}", exc_info=True)
            return None
    
