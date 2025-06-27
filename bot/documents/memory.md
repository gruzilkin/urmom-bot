# Memory System Architecture

## Overview

The urmom-bot memory system is a sophisticated hierarchical AI-powered memory architecture that maintains both permanent factual knowledge and transient episodic memories about users. The system is designed to provide rich contextual awareness while maintaining efficiency through intelligent summarization and caching.

**Key Design Decisions:**
- Uses Gemma for cost-effective summarization (free tier with generous limits)
- Optimized for small-scale deployment (5-100 users)
- In-memory caching for minimal resource usage
- User ID storage with nick translation for LLM readability

## Architecture Principles

### Two-Track Memory System
1. **Factual Memory**: Permanent knowledge explicitly provided via commands
2. **Transient Memory**: Automatic extraction and summarization of chat interactions

### Hierarchical AI Summarization
- Raw messages → Daily summaries → Weekly summaries → Final context
- AI-powered merging at every level to resolve conflicts and maintain coherence
- Bounded information growth through intelligent compression

### Cache Efficiency
- Uses Discord message ID snowflake properties for stable cache keys
- Cache keys remain valid as time windows slide
- Minimal recomputation through strategic caching

## Database Schema

### User Facts Table
```sql
CREATE TABLE user_facts (
    guild_id BIGINT NOT NULL,
    user_nick TEXT NOT NULL,
    memory_blob TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (guild_id, user_nick)
);
```

### Messages Table
```sql
CREATE TABLE messages (
    guild_id BIGINT NOT NULL,
    message_id BIGINT NOT NULL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    message_text TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    INDEX idx_guild_user_time (guild_id, user_id, message_id)
);
```

### Daily Summaries Table
```sql
-- Note: May be stored in-memory cache instead of database
CREATE TABLE daily_summaries (
    guild_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    date DATE NOT NULL,
    start_message_id BIGINT NOT NULL,
    end_message_id BIGINT NOT NULL,
    summary_text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (guild_id, user_id, date)
);
```

### Weekly Summaries Table
```sql
-- Note: May be stored in-memory cache instead of database
CREATE TABLE weekly_summaries (
    guild_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    week_start DATE NOT NULL,
    summary_text TEXT NOT NULL,
    source_days_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (guild_id, user_id, week_start)
);
```

## AI Operations

### Factual Memory Operations
```python
def remember(current_memory_blob: str, new_fact: str) -> str:
    """
    Merge new factual information into existing memory blob.
    Resolves conflicts and maintains coherent narrative.
    """
    prompt = f"""
    Update this memory about a user:
    Current memory: {current_memory_blob}
    New fact: {new_fact}
    
    Return a coherent updated memory that incorporates the new information,
    resolving any conflicts intelligently.
    """
    return llm_call(prompt)

def forget(current_memory_blob: str, fact_to_forget: str) -> str:
    """
    Remove specific information from memory blob.
    Maintains narrative coherence after removal.
    """
    prompt = f"""
    Remove specific information from this memory:
    Current memory: {current_memory_blob}
    Information to forget: {fact_to_forget}
    
    Return updated memory with the specified information removed,
    maintaining narrative flow.
    """
    return llm_call(prompt)
```

### Transient Memory Operations
```python
def prepare_messages_for_llm(messages: List[Message]) -> str:
    """
    Convert stored messages with user IDs to readable format with nicks.
    """
    readable_messages = []
    for msg in messages:
        # Replace user IDs in message text with current display names
        readable_text = replace_user_ids_with_nicks(msg.message_text, msg.guild_id)
        readable_messages.append(f"{get_user_nick(msg.user_id)}: {readable_text}")
    return "\n".join(readable_messages)

def daily_summarize(messages: List[Message]) -> str:
    """
    Summarize a day's worth of messages into key insights and events.
    Uses Gemma for cost-effective summarization.
    """
    formatted_messages = prepare_messages_for_llm(messages)
    prompt = f"""
    Summarize these messages from a user's day into key insights:
    Messages: {formatted_messages}
    
    Focus on:
    - Notable events or experiences
    - Mood and emotional state
    - Important interactions or topics
    - Behavioral patterns
    
    Keep summary concise (~300 characters).
    """
    return gemma_call(prompt)  # Using Gemma specifically

def weekly_summarize(daily_summaries: List[str]) -> str:
    """
    Merge multiple daily summaries into a weekly behavioral/contextual summary.
    """
    prompt = f"""
    Create a weekly summary from these daily summaries:
    {format_daily_summaries(daily_summaries)}
    
    Identify:
    - Recurring patterns and themes
    - Overall mood trends
    - Significant events or changes
    - Behavioral insights
    
    Produce a coherent weekly narrative.
    """
    return llm_call(prompt)

def merge_context(facts: str, daily: str, weekly: str) -> str:
    """
    Create final context blob by intelligently merging all memory sources.
    """
    prompt = f"""
    Create a comprehensive but concise user context by merging:
    
    Permanent facts: {facts}
    Recent activity (yesterday): {daily}
    Historical context (past week): {weekly}
    
    Produce a unified context that:
    - Resolves any conflicts between sources
    - Prioritizes recent information over historical
    - Maintains factual accuracy
    - Provides relevant context for conversation
    """
    return llm_call(prompt)
```

## Processing Workflows

### Daily Processing Pipeline
1. **Message Collection**: Fetch previous day's messages by user
2. **Daily Summarization**: Generate daily summary using AI
3. **Cache Storage**: Store with message ID boundary cache keys
4. **Weekly Check**: If week boundary crossed, trigger weekly summarization

### Weekly Processing Pipeline
1. **Daily Summary Collection**: Fetch last 7 daily summaries
2. **Weekly Summarization**: Generate weekly summary using AI
3. **Cache Storage**: Store with hash-based cache key
4. **Cleanup**: Archive or remove old daily summaries as needed

### Context Assembly Pipeline
1. **Retrieve Components**:
   - User facts from `user_facts` table
   - Yesterday's summary from cache/database
   - Current week's summary from cache/database
2. **AI Merge**: Create final context using `merge_context()`
3. **Cache Result**: Cache final context for conversation use
4. **Inject Context**: Provide to conversation processors

## Cache Strategy

### Storage Approach
- **In-Memory Preferred**: LRU cache or Redis for summaries (small data size ~6KB per user)
- **Daily boundaries**: Fixed day boundaries prevent overlapping windows
- **No current-day cache**: Bot fetches recent conversation history directly

### Cache Key Design
- **Daily summaries**: `daily_{guild_id}_{user_id}_{start_msg_id}_{end_msg_id}`
- **Weekly summaries**: `weekly_{guild_id}_{user_id}_{hash(daily_summaries)}`
- **Final context**: `context_{guild_id}_{user_id}_{hash(facts+daily+weekly)}`

### Memory Footprint Analysis
- **Per user**: ~6KB (facts: 400 chars, daily: 300 chars, weekly: 500 chars, context: 800 chars)
- **5 users**: 30KB total
- **100 users**: 600KB total
- **Negligible memory usage** - can store entirely in-process

### Cache Benefits
- **Stable keys**: Message ID boundaries don't change as time progresses
- **Efficient lookups**: O(1) retrieval for processed summaries
- **Minimal recomputation**: Only process new time windows
- **Memory efficiency**: Tiny footprint allows simple in-memory storage

## Discord Integration

### Commands
```
@urmom-bot remember @user <fact>     - Add permanent fact to user's memory
@urmom-bot forget @user <fact>       - Remove specific fact from user's memory
@urmom-bot memory @user              - Display current memory about user
@urmom-bot clear-memory @user        - Clear all memory about user (admin only)
```

### Automated Processing
- **Message Ingestion**: All messages automatically stored with user IDs in `messages` table
- **User ID Translation**: IDs converted to readable nicks only for LLM processing
- **Batch Processing**: Daily/weekly summaries generated via async scheduled jobs
- **Context Injection**: Memory context automatically included in AI responses
- **Failure Handling**: Failed daily summarizations are skipped, system continues with available data

## Implementation Examples

### Memory Evolution Example
```
Day 1 Messages:
- "I'm so stressed about this project deadline"
- "Going to the gym after work to clear my head"

Daily Summary: "User experiencing work stress, using exercise as coping mechanism"

Day 2-7 Messages: [Similar pattern of stress + gym usage]

Weekly Summary: "User has consistent pattern of managing work stress through regular exercise, shows resilience and healthy coping strategies"

Factual Memory: "Software engineer at TechCorp, lives in Berlin, enjoys hiking"

Final Context: "Berlin-based software engineer at TechCorp who manages work stress through exercise and enjoys outdoor activities like hiking. Currently dealing with project pressures but has healthy coping mechanisms."
```

### Context Usage Example
```python
# During joke generation
user_context = get_user_context(guild_id, user_nick)
# Result: "Berlin-based software engineer who manages work stress through exercise..."

# This context helps generate more personalized and relevant humor:
# "Ur mom's so fit, she makes your post-gym selfies look like before photos"
```

## Performance Considerations

### Efficiency Optimizations
- **Lazy Loading**: Context generated only when needed for active conversations
- **Batch Processing**: Daily/weekly jobs process multiple users simultaneously
- **Cache Warming**: Precompute contexts for frequent participants
- **Message Pruning**: Archive old raw messages, keep only summaries

### Scalability
- **Target Scale**: Optimized for small deployments (5-100 users)
- **Async Processing**: Non-blocking summarization jobs don't affect bot responsiveness
- **Gemma Integration**: Free tier provides generous limits for hobby-scale usage
- **Simple Architecture**: Minimal infrastructure requirements
- **Data Retention**: Limited message history (1 week to 1 month) keeps storage bounded

## Privacy and Safety

### Data Protection
- **User Consent**: Memory features opt-in per guild
- **Data Retention**: Configurable retention periods for different memory types
- **Anonymization**: Option to hash user identifiers for privacy
- **Export/Delete**: Users can request memory export or deletion

### Content Filtering
- **Sensitive Information**: Filter out personal information (addresses, phone numbers)
- **Inappropriate Content**: Content moderation before memory storage
- **Conflict Resolution**: Handle contradictory or false information gracefully

## Future Enhancements

### Advanced Features
- **Cross-Guild Memory**: Optional memory sharing across servers (with consent)
- **Memory Importance Scoring**: Weight memories by relevance and user interaction
- **Temporal Queries**: "What was X doing last month?" type queries
- **Memory Relationships**: Track connections between users' memories
- **Personality Modeling**: Build personality profiles from memory patterns

### Technical Improvements
- **Vector Embeddings**: Semantic similarity for better memory retrieval
- **Graph Database**: Model complex memory relationships
- **Real-time Processing**: Streaming memory updates instead of batch processing
- **A/B Testing**: Compare different summarization strategies for effectiveness