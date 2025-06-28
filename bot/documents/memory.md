# Memory System Architecture

## Overview

The urmom-bot memory system is a sophisticated hierarchical AI-powered memory architecture that maintains both permanent factual knowledge and transient episodic memories about users. The system is designed to provide rich contextual awareness while maintaining efficiency through intelligent summarization and caching.

**Key Design Decisions:**
- Uses Gemma for cost-effective summarization (free tier with generous limits)
- Optimized for small-scale deployment (5-100 users)
- LRU cache decorators for efficient on-demand caching
- Discord user ID storage with nickname translation only for LLM calls

## Architecture Principles

### Two-Track Memory System
1. **Factual Memory**: Permanent knowledge explicitly provided via commands
2. **Transient Memory**: Automatic extraction and summarization of chat interactions

### Hierarchical AI Summarization
- Raw messages → Daily summaries → Weekly summaries → Final context
- AI-powered merging at every level to resolve conflicts and maintain coherence
- Bounded information growth through intelligent compression

### Cache Efficiency
- Uses LRU cache decorators with absolute date keys for immutability
- Sliding cache keys for efficient temporal queries
- On-demand calculation with extensive caching to avoid redundant work

## Database Schema

### User Facts Table
```sql
CREATE TABLE user_facts (
    guild_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    memory_blob TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (guild_id, user_id)
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

### Summaries Storage
```python
# Daily and weekly summaries are cached using LRU decorators
# No database tables needed - computed on-demand with caching
# Cache keys based on absolute dates for immutability
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

@lru_cache(maxsize=1000)
def daily_summarize(guild_id: int, date: str, messages_hash: str) -> Dict[int, str]:
    """
    Summarize a complete day's worth of messages for all users in a guild.
    Uses absolute date as cache key for immutability.
    Returns mapping of user_id -> summary.
    """
    messages = get_messages_for_date(guild_id, date)
    user_messages = group_messages_by_user(messages)
    
    summaries = {}
    for user_id, user_msgs in user_messages.items():
        formatted_messages = prepare_messages_for_llm(user_msgs, guild_id)
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
        summaries[user_id] = gemma_call(prompt)
    
    return summaries

@lru_cache(maxsize=500)
def weekly_summarize(guild_id: int, user_id: int, week_start_date: str) -> str:
    """
    Create weekly summary from daily summaries for a specific user.
    Uses week start date as cache key for immutability.
    """
    daily_summaries = []
    for i in range(7):
        date = get_date_offset(week_start_date, i)
        day_summaries = daily_summarize(guild_id, date, get_messages_hash(guild_id, date))
        if user_id in day_summaries:
            daily_summaries.append(day_summaries[user_id])
    
    if not daily_summaries:
        return ""
    
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

@lru_cache(maxsize=500)
def merge_context(guild_id: int, user_id: int, facts: str, transient: str) -> str:
    """
    Create final context blob by intelligently merging factual and transient memory.
    """
    if not facts and not transient:
        return ""
    
    prompt = f"""
    Create a comprehensive but concise user context by merging:
    
    Permanent facts: {facts}
    Recent behavioral context: {transient}
    
    Produce a unified context that:
    - Resolves any conflicts between sources
    - Prioritizes factual information for accuracy
    - Integrates recent behavioral insights
    - Provides relevant context for conversation
    """
    return llm_call(prompt)
```

## On-Demand Processing

### Context Assembly Pipeline
1. **Retrieve Components**:
   - User facts from `user_facts` table by user_id
   - Raw messages from `messages` table for recent period
   - Convert user IDs to nicknames only for LLM processing
2. **Generate Transient Context**:
   - Calculate weekly summary using cached daily summaries
   - Use absolute date keys for immutable caching
3. **AI Merge**: Create final context using `merge_context(facts, transient)`
4. **Cache Result**: LRU cache stores final context
5. **Inject Context**: Provide to conversation processors

### Memory Operations Integration
- **AI Router Integration**: Memory operations (remember/forget/query) integrated into AI routing
- **Freeform Support**: Natural language memory commands
- **Temperature 0**: Precise instruction following without creative interpretation

## Cache Strategy

### Storage Approach
- **LRU Cache Decorators**: Automatic caching with configurable max sizes
- **Absolute Date Keys**: Immutable cache keys based on calendar dates
- **Sliding Windows**: Efficient temporal queries using date arithmetic

### Cache Key Design
- **Daily summaries**: Absolute date string (e.g., "2024-01-15")
- **Weekly summaries**: Week start date (e.g., "2024-01-15" for Monday)
- **Final context**: Hash of facts + transient context components

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
@urmom-bot clear-memory @user        - Clear all memory about user
```

### Automated Processing
- **Message Ingestion**: All messages automatically stored with user IDs in `messages` table
- **User ID Translation**: IDs converted to readable nicks only for LLM processing
- **On-Demand Calculation**: Summaries and context generated when requested
- **LRU Caching**: Extensive caching prevents redundant AI calls
- **Context Injection**: Memory context automatically included in AI responses

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
- **LRU Caching**: Automatic cache management with size limits
- **Immutable Keys**: Absolute date keys ensure cached results never invalidate
- **Message Pruning**: Archive old raw messages, keep only summaries

### Scalability
- **Target Scale**: Optimized for small deployments (5-100 users)
- **On-Demand Processing**: No background jobs, everything calculated when needed
- **Gemma Integration**: Free tier provides generous limits for hobby-scale usage
- **Simple Architecture**: Minimal infrastructure requirements
- **Data Retention**: Limited message history (1 week to 1 month) keeps storage bounded

## Privacy and Safety

### Data Protection
- **Private Bot**: No consent needed - users are aware of functionality
- **Data Retention**: Configurable retention periods for different memory types
- **User ID Storage**: Discord user IDs in database, nicknames only for LLM calls

### Content Processing
- **Direct Storage**: All message content stored without filtering
- **Temperature 0**: AI operations use deterministic responses
- **Conflict Resolution**: Handle contradictory information through AI merging

