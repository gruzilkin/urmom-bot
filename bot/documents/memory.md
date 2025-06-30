# Memory System Architecture

## Implementation Status

**✅ MILESTONE 1 COMPLETE: Factual Memory System**
- Factual memory operations (remember/forget) with AI-powered merging
- Database schema and storage
- Discord integration with freeform commands
- User resolution with proper Discord API integration
- Memory injection into AI responses via XML-structured prompts
- Natural memory queries handled by general query generator using injected context

**🚧 MILESTONE 2 PLANNED: Transient Memory System**
- Chat history ingestion and storage (TODO)
- Daily/weekly summarization pipeline with sliding window (TODO)  
- Context assembly and merging with MemoryManager (TODO)
- Cost-optimized AI client strategy (Gemma for volume, Gemini for quality) (TODO)

## Overview

The urmom-bot memory system is a sophisticated hierarchical AI-powered memory architecture that maintains both permanent factual knowledge and transient episodic memories about users. The system is designed to provide rich contextual awareness while maintaining efficiency through intelligent summarization and caching.

**Key Design Decisions:**
- Uses Gemma AI client for all memory operations (free tier with generous limits)
- Optimized for small-scale deployment (5-100 users)
- `cachetools.LRUCache` instances for efficient async caching (not built-in `@lru_cache`)
- Discord user ID storage with nickname translation only for LLM calls
- Clean MemoryManager interface hiding internal complexity

## Architecture Principles

### Two-Track Memory System
1. **Factual Memory**: Permanent knowledge explicitly provided via commands
2. **Transient Memory**: Automatic extraction and summarization of chat interactions

### Hierarchical AI Summarization
- Raw chat history → Daily summaries (Gemma) → Weekly summary (Gemma, sliding 7-day window) → Final context (Gemma, merging weekly summary + facts)
- AI-powered merging at every level to resolve conflicts and maintain coherence
- Bounded information growth through intelligent compression
- Weekly summary calculated daily with sliding window using Gemma for consistency

### Cache Efficiency
- Uses `cachetools.LRUCache` instances with absolute date keys for immutability
- Manual async cache management for async methods
- On-demand calculation with extensive caching to avoid redundant work
- Key-based eviction capability when needed

## Database Schema

### User Facts Table ✅ IMPLEMENTED
```sql
CREATE TABLE user_facts (
    guild_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    memory_blob TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (guild_id, user_id)
);
```

### Chat History Table 🚧 TODO (for transient memory)
```sql
CREATE TABLE chat_history (
    guild_id BIGINT NOT NULL,
    message_id BIGINT NOT NULL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    message_text TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    INDEX idx_guild_user_timestamp (guild_id, user_id, timestamp)
);
```

## MemoryManager Architecture 🚧 TODO

### Clean Interface Design
The transient memory system will be encapsulated in a `MemoryManager` class that provides a simple interface while hiding internal complexity:

```python
class MemoryManager:
    async def ingest_message(self, guild_id: int, message_id: int, user_id: int, message_text: str, timestamp: datetime) -> None:
        """Store message in chat_history table"""
    
    async def get_memories(self, guild_id: int, user_id: int) -> tuple[str | None, str | None]:
        """Returns (facts, observations) for a user
        
        facts: from existing factual memory system (user_facts table)
        observations: from transient memory pipeline (daily→weekly summaries)
        
        All complex summarization pipeline happens internally with caching.
        """
```

### Internal Implementation
- **Message Storage**: Automatic ingestion to `chat_history` table
- **Daily Summarization**: Gemma client for cost-effective high-volume processing
- **Weekly Summary**: Gemma client for reasoning over sliding 7-day window (calculated daily)
- **Context Merging**: Gemma client for intelligent factual + behavioral context combination
- **Caching Strategy**: `cachetools.LRUCache` instances for aggressive caching
- **Data Retention**: Automatic cleanup of old chat history outside retention window

### AI Client Strategy (Cost-Optimized)
- **All Memory Operations (Gemma - Free)**: Daily summaries, weekly summary, and context merging for all users

## AI Operations

### Factual Memory Operations ✅ IMPLEMENTED
```
Remember Operation:
- Input: Current memory blob (or empty if new user) + new fact to remember
- Process: AI-powered intelligent merging of new information with existing memory
- Conflict resolution: Prioritize newer information while maintaining narrative coherence
- Temperature: 0 (deterministic for consistency)
- Perspective normalization: Convert all facts to third-person declarative statements from external observer perspective
- Output: Updated memory blob incorporating the new fact in consistent third-person format
- Storage: Save merged result to user_facts table

Forget Operation:
- Input: Current memory blob + specific fact to remove
- Process: AI-powered selective removal while preserving narrative flow
- Fact detection: Determine if the specified information exists in memory
- Temperature: 0 (deterministic for consistency)
- Perspective consistency: Maintain third-person declarative statements from external observer perspective
- Output: Updated memory blob with specified information removed (or unchanged if not found)
- Storage: Save updated result to user_facts table
- Feedback: Indicate whether the fact was found and removed

Memory Merging Strategy:
- Resolve conflicts by prioritizing factual accuracy
- Maintain natural narrative flow and coherence
- Handle contradictions intelligently (newer info generally supersedes older)
- Preserve important context while incorporating updates
- Keep memory concise and relevant
- Enforce consistent third-person perspective throughout all stored facts
```

### Transient Memory Operations 🚧 TODO
```
Message Formatting:
- Convert stored messages with user IDs to readable format
- Replace user mentions with current display names for LLM processing
- Format as "username: message content" for each message

Daily Summarization (per-user approach):
- Input: All messages from a specific date, target user ID
- Process: Analyze full day's conversation focusing on specific user
- Cache: Use (guild_id, user_id, date) as cache key
- Benefits: Simpler prompts, better error isolation, independent caching
- Prompt focus areas:
  * Notable events or experiences mentioned
  * User's mood and emotional state
  * Important interactions and topics
  * Behavioral patterns observed
  * Information revealed through other users' messages
- Output: Concise daily summary (~300 chars) for the target user

Weekly Summary (sliding window):
- Input: 7 days of daily summaries ending on specific date
- Process: Create behavioral summary from daily summaries
- Cache: Use (guild_id, user_id, end_date) as cache key
- Sliding window: Calculate daily for last 7 days from end_date
- Prompt focus areas:
  * Recurring patterns and themes
  * Overall mood trends
  * Significant events or changes
  * Behavioral insights and personality traits
- Output: Coherent weekly narrative focusing on behavioral observations

Context Merging:
- Input: Factual memory blob + weekly behavioral summary
- Process: Intelligently merge permanent facts with transient observations
- Cache: Use hash of both input components as cache key
- Merge strategy:
  * Resolve conflicts between sources intelligently
  * Prioritize factual information for accuracy
  * Integrate recent behavioral insights
  * Provide relevant context for personalized conversation
- Output: Unified context for AI conversation processing
```

## On-Demand Processing

### Context Assembly Pipeline
1. **Retrieve Components**:
   - User facts from `user_facts` table by user_id
   - Raw messages from `chat_history` table for recent period
   - Convert user IDs to nicknames only for LLM processing
2. **Generate Transient Context**:
   - Calculate weekly summary using cached daily summaries
   - Use absolute date keys for immutable caching
3. **AI Merge**: Create final context using `merge_context(facts, transient)`
4. **Cache Result**: LRU cache stores final context
5. **Inject Context**: Provide to conversation processors

### Memory Operations Integration
- **AI Router Integration**: Memory operations (remember/forget) integrated into AI routing
- **Freeform Support**: Natural language memory commands
- **Temperature 0**: Precise instruction following without creative interpretation

## Cache Strategy

### Storage Approach
- **LRU Cache Decorators**: Automatic caching with configurable max sizes
- **Absolute Date Keys**: Immutable cache keys based on calendar dates
- **Sliding Windows**: Efficient temporal queries using date arithmetic

### Cache Key Design
All cache keys include `(guild_id, user_id, ...)` since all memory operations are per-user:

- **Daily summaries**: `(guild_id, user_id, date)` - e.g., "2024-01-15"
- **Weekly summary**: `(guild_id, user_id, end_date)` - e.g., "2024-01-21" for 7-day period ending that date  
- **Final context**: `(guild_id, user_id, hash(facts), weekly_end_date)` - combines fact hash with weekly summary date

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

### Freeform Commands ✅ IMPLEMENTED
```
Bot remember that gruzilkin is Sergey
Bot, remember this about Florent [conversation history]
Bot forget that gruzilkin likes pizza
Bot what do you know about @user
```

### Automated Processing
- **Chat History Ingestion**: All messages automatically stored with user IDs in `chat_history` table 🚧 TODO
- **User ID Translation**: IDs converted to readable nicks only for LLM processing ✅ IMPLEMENTED
- **On-Demand Calculation**: Summaries and context generated when requested via MemoryManager 🚧 TODO  
- **Caching with cachetools**: Extensive caching prevents redundant AI calls using `LRUCache` instances 🚧 TODO
- **Context Injection**: Memory context automatically included in AI responses ✅ IMPLEMENTED

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
- **Message Retention**: Use 7-day window for summarization

### Scalability
- **Target Scale**: Optimized for small deployments (5-100 users)
- **On-Demand Processing**: No background jobs, everything calculated when needed
- **Single AI Client**: Gemma provides free tier with generous limits for all memory operations
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

