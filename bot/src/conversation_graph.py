import datetime
import logging
from typing import Callable, Awaitable, Any
from dataclasses import dataclass
import nextcord
from message_node import MessageNode

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """Represents a message in chronological conversation format."""
    message_id: int
    author_id: int
    content: str
    timestamp: str  # Pre-formatted LLM-friendly timestamp
    mentioned_user_ids: list[int]
    reply_to_id: int | None = None


class MessageGraph:
    """Graph data structure for conversation messages with deduplication."""
    def __init__(self) -> None:
        self.nodes: dict[int, nextcord.Message] = {}
        self.unexplored_references: set[int] = set()
        self.temporal_frontier: set[int] = set()
    
    def add_node(self, message: nextcord.Message) -> bool:
        """Add message with deduplication."""
        if message.id not in self.nodes:
            self.nodes[message.id] = message
            if message.reference and message.reference.message_id:
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
    
    def get_unexplored_references(self) -> list[nextcord.Message]:
        """Get messages with unexplored references."""
        return [self.nodes[msg_id] for msg_id in self.unexplored_references]
    
    def get_temporal_frontier(self) -> list[nextcord.Message]:
        """Get messages at temporal exploration frontier, sorted by recency."""
        frontier_messages = [self.nodes[msg_id] for msg_id in self.temporal_frontier]
        return sorted(frontier_messages, key=lambda m: m.created_at, reverse=True)
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    async def to_chronological_conversation(self, discord_to_message_node_func: Callable[[nextcord.Message], Awaitable['MessageNode']]) -> list[ConversationMessage]:
        """Convert to chronological conversation format."""
        messages = sorted(self.nodes.values(), key=lambda m: m.created_at)
        conversation_messages = []
        
        for message in messages:
            message_node = await discord_to_message_node_func(message)
            
            conversation_messages.append(ConversationMessage(
                message_id=message_node.id,
                author_id=message_node.author_id,
                content=message_node.content,
                timestamp=message_node.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                mentioned_user_ids=message_node.mentioned_user_ids,
                reply_to_id=message_node.reference_id
            ))
        
        return conversation_messages


class ConversationGraphBuilder:
    """
    Builds conversation graphs using tik/tok alternating exploration.
    Abstracted to work with any message fetching interface.
    """
    
    def __init__(self, 
                 fetch_message: Callable[[int], Awaitable[nextcord.Message | None]],
                 fetch_history: Callable[[int | None], Awaitable[list[nextcord.Message]]],
                 telemetry: Any = None) -> None:
        """
        Initialize with message fetching abstractions.
        
        Args:
            fetch_message: async function(message_id: int) -> nextcord.Message | None
            fetch_history: async function(message_id: int | None) -> list[nextcord.Message]
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
                reference_id = None
                if message.reference and message.reference.message_id:
                    reference_id = message.reference.message_id
                    
                if reference_id:
                    # Use cached fetcher for all message fetching
                    referenced_msg = await self.cached_fetcher.get_message_by_id(reference_id)
                    
                    if referenced_msg and graph.add_node(referenced_msg):
                        added_any = True
            except Exception as e:
                logger.error(f"Could not fetch referenced message {reference_id}: {e}", exc_info=True)
            
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
    
    async def get_linear_history(self, trigger_message: nextcord.Message, min_linear: int) -> list[nextcord.Message]:
        """Get guaranteed linear message history."""
        min_linear = max(min_linear, 1)
        
        prev_messages = await self.cached_fetcher.get_bulk_history(trigger_message.id)
        messages = [trigger_message] + prev_messages[:min_linear - 1]
        
        return messages
    
    async def build_conversation_graph(self, 
                                     trigger_message: nextcord.Message,
                                     min_linear: int,
                                     max_total: int, 
                                     time_threshold_minutes: int,
                                     discord_to_message_node_func: Callable[[nextcord.Message], Awaitable['MessageNode']]) -> list[ConversationMessage]:
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
        
        return await graph.to_chronological_conversation(discord_to_message_node_func)
    


class CachedHistoryFetcher:
    """
    Cached wrapper for fetch_history that reads in bulk but returns single messages.
    Maintains a cache of message_id -> previous message to minimize Discord API calls.
    """
    
    def __init__(self, fetch_history: Callable[[int | None], Awaitable[list[nextcord.Message]]], 
                 fetch_message: Callable[[int], Awaitable[nextcord.Message | None]]) -> None:
        self.fetch_history = fetch_history
        self.fetch_message = fetch_message
        self.cache: dict[int, nextcord.Message] = {}
        self.message_cache: dict[int, nextcord.Message] = {}
        
    async def get_previous_message(self, message_id: int) -> nextcord.Message | None:
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
    
    async def get_bulk_history(self, message_id: int | None) -> list[nextcord.Message]:
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
    
    async def get_message_by_id(self, message_id: int) -> nextcord.Message | None:
        """
        Get a specific message by ID, using cache when possible.
        
        Args:
            message_id: ID of message to fetch
            
        Returns:
            nextcord.Message or None if not found
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
    
