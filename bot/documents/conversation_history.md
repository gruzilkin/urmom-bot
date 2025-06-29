# Graph-Based Conversation History

## Overview

The conversation history system uses a graph-based approach to intelligently gather contextual messages for AI responses. This system preserves bot messages, follows explicit reply relationships, and handles temporal conversation boundaries effectively with efficient lazy processing.

## Core Architecture

### Graph-Based Message Clustering

The system builds a conversation graph where Discord messages are nodes connected by two types of edges:

1. **Temporal edges**: Messages within a time threshold (30 minutes)
2. **Reference edges**: Discord reply relationships (always followed regardless of age)

This approach ensures that both explicit user connections (replies) and implicit temporal continuity are preserved while maintaining logical conversation boundaries.

### Lazy Processing & Efficiency

**Key architectural decision**: The conversation graph works exclusively with `nextcord.Message` objects throughout the building process. Expensive operations like article extraction from embeds are deferred until materialization time, ensuring only the final conversation messages (typically 10-30) undergo processing rather than all fetched messages (100+).

## Algorithm: Tik/Tok Alternating Exploration

The conversation building algorithm uses alternating exploration phases:

```python
async def build_conversation_graph(trigger_message, min_linear, max_total, time_threshold_minutes, discord_to_message_node_func):
    graph = MessageGraph()  # Works with nextcord.Message objects
    
    # Seed with guaranteed linear history
    linear_messages = await get_linear_history(trigger_message, min_linear)
    for msg in linear_messages:
        graph.add_node(msg)  # Add Discord message directly
    
    while len(graph) < max_total:
        # TIK: Reference step - follow all unexplored references
        reference_step_added = await explore_references(graph)
        
        if len(graph) >= max_total:
            break
            
        # TOK: Temporal step - explore neighbors with sealing
        temporal_step_added = await explore_temporal_neighbors(graph, time_threshold)
        
        # Exit if neither step found new messages
        if not reference_step_added and not temporal_step_added:
            break
    
    # Materialization happens only here for final messages
    return graph.to_chronological_conversation(discord_to_message_node_func)
```

### Key Features

1. **Guaranteed Minimum Context**
   - Always includes `min_linear` recent messages regardless of time gaps
   - Handles implicit conversation continuity when users respond to older topics

2. **Reference Following**
   - All Discord reply references are followed (explicit user connections)
   - No circular references possible (past messages can't reference future ones)
   - Natural deduplication (same message reached via multiple paths = single node)

3. **Temporal Sealing**
   - When exploring temporal neighbors, if time gap > threshold, seals that path
   - Prevents exploration of unrelated conversation clusters
   - Allows algorithm to explore other directions
   - Default threshold: 30 minutes

4. **Balanced Exploration**
   - Alternates between reference exploration (TIK) and temporal exploration (TOK)
   - Ensures neither explicit connections nor temporal continuity dominate
   - References explored 1 level deep at a time

5. **Natural Termination**
   - Stops when max message limit reached
   - Stops when no new messages found in either exploration type
   - Bounded performance with predictable limits

6. **Rich Content Support & Lazy Processing**
   - Extracts article content from embedded URLs only during final materialization
   - Embeds article content directly into message text using XML format: `<embeddings><embedding type="article" url="...">content</embedding></embeddings>`
   - Ensures expensive article extraction only happens for final conversation messages

## Benefits

- **Preserves bot messages**: Enables iterative conversations with AI
- **Logical conversation boundaries**: Isolates related message clusters
- **Handles slow conversations**: Works well for small friend groups
- **Explicit connection preservation**: Reply chains always followed
- **Transparent logic**: Clear, predictable decision making
- **Performance bounded**: Max size limits prevent runaway exploration
- **Efficient content processing**: Lazy article extraction minimizes API calls and processing overhead

## Implementation Components

### MessageGraph Class

Provides graph data structure with:
- **Node deduplication**: Same message ID creates single node
- **Reference tracking**: Maintains unexplored reference connections  
- **Temporal frontier**: Tracks messages available for temporal exploration
- **Discord message storage**: Works exclusively with `nextcord.Message` objects
- **Lazy materialization**: Converts to ConversationMessage format only at the end via `discord_to_message_node_func`

### ConversationGraphBuilder Class

Implements the core algorithm with:
- **Dependency injection**: Abstract message fetching to work with any message source
- **Reference exploration**: Follows `message.reference` relationships 
- **Temporal exploration**: Channel history with time-based sealing
- **Discord message focus**: Operates entirely on Discord message objects during graph building

### Integration Points

- **get_recent_conversation()**: Main entry point in `app.py` that uses the graph system
- **create_conversation_fetcher()**: Creates parameterless lambda for AI generators
- **discord_to_message_node()**: Materialization function in `app.py` that processes embeds and extracts articles
- **MessageNode class**: Separate module (`message_node.py`) containing final materialized message representation

## Edge Cases Handled

- **Implicit continuity**: Minimum linear messages for non-referenced context
- **Explicit connections**: References always followed regardless of age
- **Graph size limits**: Tik/tok exploration with max message bounds
- **Conversation isolation**: Temporal sealing prevents cluster bleeding
- **Circular paths**: Natural deduplication handles multiple paths to same message
- **Article extraction**: Safely handles embed processing with error recovery at materialization time only

## Performance & Efficiency

### Lazy Processing Architecture

The system implements a two-phase approach for maximum efficiency:

1. **Graph Building Phase**: Works exclusively with lightweight `nextcord.Message` objects
   - Fast message fetching and graph construction
   - No expensive embed processing during exploration
   - Typical scenario: 100+ messages fetched, minimal processing overhead

2. **Materialization Phase**: Processes final conversation messages only
   - Article extraction from embeds using `goose3`
   - XML embedding of article content into message text
   - User mention resolution to display names
   - Typical scenario: 10-30 messages processed with full content enrichment

This approach provides **70-90% efficiency improvement** by avoiding unnecessary processing of messages that don't make it into the final conversation.

### Content Embedding Format

Articles are embedded directly into message content using XML structure:
```
Original message content <embeddings><embedding type="article" url="https://example.com/article">Article content here...</embedding></embeddings>
```

This format allows AI models to seamlessly access both the original message and referenced article content in a single coherent text stream.

## Configuration

Parameters are configured in `create_conversation_fetcher()` in `app.py`:
- `min_messages=10`: Minimum guaranteed linear messages
- `max_messages=30`: Maximum total messages in graph
- `max_age_minutes=30`: Temporal connection threshold