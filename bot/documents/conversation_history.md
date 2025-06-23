# Conversation History Refactor

## Problem Statement

The current conversation fetching logic in `app.py` has several limitations:

1. **Bot message filtering**: Removes all bot messages, breaking iterative conversations where follow-up questions need previous AI responses for context
2. **Simple time-based cutoff**: Uses fixed time windows that don't capture logical conversation boundaries
3. **Inadequate for small friend groups**: Conversations can be slow with gaps, but still logically connected

## Current Implementation Issues

Located in `app.py:203-270` (`get_recent_conversation`):
- Filters out all bot messages at line 230
- Uses simple time cutoff that doesn't follow conversation threads
- Doesn't handle reply chains or referenced messages effectively

## Proposed Solution: Graph-Based Conversation Clustering

### Core Concept

Build a conversation graph where messages are nodes connected by two types of edges:
1. **Temporal edges**: Messages within a time threshold (10 minutes)
2. **Reference edges**: Discord reply relationships (always followed regardless of age)

### Algorithm Design

#### Tik/Tok Alternating Exploration

```python
async def build_conversation_graph(channel, trigger_message, min_linear=10, max_total=30, time_threshold_minutes=10):
    graph = MessageGraph()
    
    # Seed with guaranteed linear history
    linear_messages = await get_linear_history(channel, trigger_message, min_linear)
    for msg in linear_messages:
        graph.add_node(msg)
    
    while len(graph) < max_total:
        # TIK: Reference step - follow all unexplored references
        reference_step_added = await explore_references(graph, channel)
        
        if len(graph) >= max_total:
            break
            
        # TOK: Temporal step - explore neighbors with sealing
        temporal_step_added = await explore_temporal_neighbors(graph, channel, time_threshold)
        
        # Exit if neither step found new messages
        if not reference_step_added and not temporal_step_added:
            break
    
    return graph.to_chronological_conversation()
```

#### Key Features

1. **Guaranteed Minimum Context**
   - Always include `min_linear` recent messages regardless of time gaps
   - Handles implicit conversation continuity (someone responding to yesterday's topic)

2. **Reference Following**
   - All Discord reply references are followed (explicit user connections)
   - No circular references possible (past messages can't reference future ones)
   - Natural deduplication (same message reached via multiple paths = single node)

3. **Temporal Sealing**
   - When exploring temporal neighbors, if time gap > threshold, seal that path
   - Prevents exploration of unrelated conversation clusters
   - Allows algorithm to explore other directions

4. **Balanced Exploration**
   - Alternates between reference exploration (TIK) and temporal exploration (TOK)
   - Ensures neither explicit connections nor temporal continuity dominate
   - References explored 1 level deep at a time

5. **Natural Termination**
   - Stops when max message limit reached
   - Stops when no new messages found in either exploration type
   - Bounded performance with predictable limits

### Benefits

- **Preserves bot messages**: Enables iterative conversations with AI
- **Logical conversation boundaries**: Isolates related message clusters
- **Handles slow conversations**: Works well for small friend groups
- **Explicit connection preservation**: Reply chains always followed
- **Transparent logic**: Clear, predictable decision making
- **Performance bounded**: Max size limits prevent runaway exploration

### Implementation Requirements

1. **MessageGraph class**: Node deduplication, edge management
2. **Reference exploration**: Follow `message.reference` relationships
3. **Temporal exploration**: Channel history with time-based sealing
4. **Chronological output**: Convert graph to time-ordered conversation format
5. **Integration**: Replace current `get_recent_conversation` and update `create_conversation_fetcher`

### Edge Cases Handled

- **Implicit continuity**: Minimum linear messages for non-referenced context
- **Explicit connections**: References always followed regardless of age
- **Graph size limits**: Tik/tok exploration with max message bounds
- **Conversation isolation**: Temporal sealing prevents cluster bleeding
- **Circular paths**: Natural deduplication handles multiple paths to same message

## Next Steps

1. Implement `MessageGraph` data structure
2. Implement reference and temporal exploration functions  
3. Replace existing conversation fetching logic
4. Update conversation fetcher creation
5. Test with various conversation patterns
6. Ensure bot message inclusion works for iterative queries