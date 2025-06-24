# Graph-Based Conversation History

## Overview

The conversation history system uses a graph-based approach to intelligently gather contextual messages for AI responses. This system preserves bot messages, follows explicit reply relationships, and handles temporal conversation boundaries effectively.

## Core Architecture

### Graph-Based Message Clustering

The system builds a conversation graph where messages are nodes connected by two types of edges:

1. **Temporal edges**: Messages within a time threshold (30 minutes)
2. **Reference edges**: Discord reply relationships (always followed regardless of age)

This approach ensures that both explicit user connections (replies) and implicit temporal continuity are preserved while maintaining logical conversation boundaries.

## Algorithm: Tik/Tok Alternating Exploration

The conversation building algorithm uses alternating exploration phases:

```python
async def build_conversation_graph(trigger_message, min_linear, max_total, time_threshold_minutes):
    graph = MessageGraph()
    
    # Seed with guaranteed linear history
    linear_messages = await get_linear_history(trigger_message, min_linear)
    for msg in linear_messages:
        graph.add_node(msg)
    
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
    
    return graph.to_chronological_conversation()
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

6. **Rich Content Support**
   - Extracts article content from embedded URLs in messages
   - Includes article content in conversation context for AI responses

## Benefits

- **Preserves bot messages**: Enables iterative conversations with AI
- **Logical conversation boundaries**: Isolates related message clusters
- **Handles slow conversations**: Works well for small friend groups
- **Explicit connection preservation**: Reply chains always followed
- **Transparent logic**: Clear, predictable decision making
- **Performance bounded**: Max size limits prevent runaway exploration
- **Rich content support**: Extracts and includes article content from embedded URLs

## Implementation Components

### MessageGraph Class

Provides graph data structure with:
- **Node deduplication**: Same message ID creates single node
- **Reference tracking**: Maintains unexplored reference connections
- **Temporal frontier**: Tracks messages available for temporal exploration
- **Chronological output**: Converts graph to time-ordered conversation format

### ConversationGraphBuilder Class

Implements the core algorithm with:
- **Dependency injection**: Abstract message fetching to work with any message source
- **Reference exploration**: Follows `message.reference` relationships
- **Temporal exploration**: Channel history with time-based sealing
- **Article extraction**: Processes embedded URLs for additional context

### Integration Points

- **get_recent_conversation()**: Main entry point in `app.py` that uses the graph system
- **create_conversation_fetcher()**: Creates parameterless lambda for AI generators
- **Discord API abstraction**: Converts Discord messages to MessageNode format

## Edge Cases Handled

- **Implicit continuity**: Minimum linear messages for non-referenced context
- **Explicit connections**: References always followed regardless of age
- **Graph size limits**: Tik/tok exploration with max message bounds
- **Conversation isolation**: Temporal sealing prevents cluster bleeding
- **Circular paths**: Natural deduplication handles multiple paths to same message
- **Article extraction**: Safely handles embed processing with error recovery

## Configuration

Parameters are configured in `create_conversation_fetcher()` in `app.py`:
- `min_messages=10`: Minimum guaranteed linear messages
- `max_messages=30`: Maximum total messages in graph
- `max_age_minutes=30`: Temporal connection threshold