# Memory System Architecture

## Implementation Status

**✅ MILESTONE 1 COMPLETE: Factual Memory System**
- Factual memory operations (remember/forget) with AI-powered merging
- Database schema and storage
- Discord integration with freeform commands
- User resolution with proper Discord API integration
- Memory injection into AI responses via XML-structured prompts
- Natural memory queries handled by general query generator using injected context

**✅ MILESTONE 2 COMPLETE: Transient Memory System**
- Chat history ingestion and storage
- Daily summarization pipeline with 3-way merge architecture
- Context assembly and merging with MemoryManager
- Dual caching strategy (TTL + LRU) with content-based hashing

## Overview

The urmom-bot memory system is a sophisticated hierarchical AI-powered memory architecture that maintains both permanent factual knowledge and transient episodic memories about users. The system is designed to provide rich contextual awareness while maintaining efficiency through intelligent summarization and caching.

**Key Design Decisions:**
- Uses Gemma AI client for all memory operations (free tier with generous limits)
- Optimized for small-scale deployment (5-100 users)
- `cachetools.LRUCache` instances for efficient async caching (not built-in `@lru_cache`)
- Discord user ID storage with nickname translation only for LLM calls
- Clean MemoryManager interface hiding internal complexity
- **Resilience-first approach**: System prioritizes reliability over completeness with graceful degradation

## Architecture Principles

### Two-Track Memory System
1. **Factual Memory**: Permanent knowledge explicitly provided via commands
2. **Transient Memory**: Automatic extraction and summarization of chat interactions

### Hierarchical AI Summarization
- Raw chat history → Daily summaries (Gemma) → Historical summary (Gemma, days 2-7) → Final context (Gemma, 3-way merge)
- **Strategy**: `merge(facts, current_day_summary, historical_summary)` for optimal caching
- Current day (day 1) updated hourly, historical days (2-7) cached permanently
- AI-powered merging at every level to resolve conflicts and maintain coherence
- Bounded information growth through intelligent compression

### Cache Efficiency
- **Dual Strategy**: TTL caching for current day, LRU caching for historical data
- Current day summaries: 1-hour TTL with hour-bucket keys for intraday updates
- Historical summaries: Permanent LRU cache with immutable date keys
- Content-based hashing for final context merge to ensure cache correctness
- Manual async cache management for async methods
- On-demand calculation with extensive caching to avoid redundant work

### Resilience & Best-Effort Design
- **Graceful Degradation**: System continues functioning even when AI services are unavailable
- **Database-First Reliability**: Factual memories from database always accessible regardless of AI quota
- **Fallback Hierarchy**: If AI operations fail, system returns available data in priority order (facts → current day → historical)
- **Best-Effort Philosophy**: Users always receive some memory context rather than complete failure
- **Comprehensive Observability**: Telemetry tracking provides visibility into system health and performance

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

### Chat History Table ✅ IMPLEMENTED
```sql
CREATE TABLE chat_history (
    guild_id BIGINT NOT NULL,
    channel_id BIGINT NOT NULL,
    message_id BIGINT NOT NULL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    message_text TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    INDEX idx_guild_user_timestamp (guild_id, user_id, timestamp)
);
```

## MemoryManager Architecture ✅ IMPLEMENTED

### Clean Interface Design
The transient memory system is encapsulated in a `MemoryManager` class that provides a simple interface while hiding internal complexity:

```python
class MemoryManager:
    async def ingest_message(self, guild_id: int, message: MessageNode) -> None:
        """Store message in chat_history table"""
    
    async def get_memories(self, guild_id: int, user_id: int) -> str | None:
        """Returns unified context for a user via 3-way merge with resilient fallback
        
        Combines: factual memory + current day summary + historical summary (days 2-7)
        Returns: Single unified context string or None if no memories exist
        
        Resilience guarantees:
        - Database facts always returned even if AI operations fail
        - Graceful fallback to partial context when AI quota exhausted
        - Best-effort approach ensures users get available memory data
        
        All complex summarization pipeline happens internally with dual caching strategy.
        """
```

### Internal Implementation ✅ IMPLEMENTED
- **Message Storage**: Automatic ingestion to `chat_history` table
- **Daily Summarization**: Gemma client for current day and historical day processing
- **Historical Summary**: Gemma client for reasoning over days 2-7 with behavioral focus
- **3-Way Context Merging**: Gemma client intelligently combines facts + current day + historical
- **Dual Caching Strategy**: TTLCache (1-hour) for current day, LRUCache for historical data
- **Content-Based Hashing**: Cache keys use content hashes for final context merge
- **Code Deduplication**: Shared daily summary generation with separate caching layers
- **Daily Summaries Batch Cache**: LRU cache for historical daily summaries per guild/date

### AI Client Strategy (Cost-Optimized)
- **Batch Daily Summaries (Gemini Flash)**: Multi-user daily summary generation using superior analysis capabilities
- **Historical Summaries & Context Merging (Gemma - Free)**: Individual operations using cost-efficient free tier
- **Optimization**: Reduces API calls from N per day to 1 per day per guild for daily summaries

## AI Operations

### Factual Memory Operations ✅ IMPLEMENTED
```
Third-Person Perspective:
- All factual memory operates with statements in third-person perspective 

Remember Operation:
- Input: Current memory blob (or empty if new user) + new fact to remember
- Process: AI-powered intelligent merging of new information with existing memory
- Conflict resolution: Prioritize newer information while maintaining narrative coherence
- Temperature: 0 (deterministic)
- Output: Updated memory blob incorporating the new fact in consistent third-person format
- Storage: Save merged result to user_facts table

Forget Operation:
- Input: Current memory blob + fact content to remove
- Process: AI-powered selective removal while preserving narrative flow
- Temperature: 0 (deterministic)
- Output: Updated memory blob in consistent third-person format with specified information removed
- Storage: Save updated result to user_facts table
- Feedback: Indicate whether the fact was found and removed

Memory Merging Strategy:
- Prioritize newer information while maintaining narrative coherence
- Handle contradictions intelligently
- Keep memory concise and relevant
```

### Transient Memory Operations ✅ IMPLEMENTED
```
Message Formatting:
- Convert stored messages with user IDs to readable format using XML structure
- Replace user mentions with current display names for LLM processing
- Format with structured XML: <message><timestamp/><author_id/><author/><content/></message>
- Target user identification: Include both nickname and user_id in prompts for precise identification

Daily Summarization (batch approach):
- Input: All messages from a specific date and list of all active users
- Process: Single Gemini API call analyzes entire day and generates summaries for all users
- Batch Cache: One cache entry per (guild_id, date) contains map of all user summaries
- Current day cache: Current day uses 1-hour TTL with hour buckets (guild_id, "date-HH")
- Historical day cache: Historical days use permanent LRU with date keys (guild_id, date)
- Prompt focus areas:
  * Notable events or experiences mentioned by each user
  * Each user's mood and emotional state
  * Important interactions and topics they discussed
  * Behavioral patterns they exhibited
  * Information revealed about them through messages from others
- Output: Map of user_id to concise daily summary (~300 chars) in third person

Historical Summary (days 2-7):
- Input: 6 days of daily summaries (yesterday and 5 days before that, relative to current date)
- Process: Create behavioral summary from historical daily summaries with actual date ranges
- Cache: Use (guild_id, user_id, historical_end_date) as cache key with permanent LRU
- Updates: Only when day transitions (current day becomes historical)
- Prompt focus areas:
  * Recurring patterns and themes from recent history
  * Overall mood trends over the period
  * Significant events or behavioral changes
  * Personality insights from consistent behaviors
- Output: Coherent historical narrative (~500 chars) focusing on behavioral observations

Context Merging (3-way merge):
- Input: Factual memory blob + current day summary + historical summary
- Process: Intelligently merge three sources using structured XML prompts
- Cache: Content-based hashing: (guild_id, user_id, hash(facts), hash(current), hash(historical))
- Merge strategy:
  * Prioritize factual information for accuracy
  * Balance current observations with historical patterns
  * Resolve conflicts intelligently, favoring factual data
  * Provide unified context for personalized conversation
- Output: Single unified context string for AI conversation processing
```

## On-Demand Processing

### Context Assembly Pipeline
1. **Retrieve Components**:
   - User facts from `user_facts` table by user_id
   - Raw messages from `chat_history` table for recent period
   - Convert user IDs to nicknames only for LLM processing
2. **Generate Transient Context**:
   - Get current day summary (1-hour cache with hour buckets)
   - Get historical summary (permanent LRU cache for days 2-7)
   - Use content-based caching for optimal efficiency
3. **AI Merge**: Create final context using `merge_context(facts, current_day, historical)`
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
Cache keys use different strategies based on operation type:

**Batch Operations (all users per guild):**
- **Current day summaries**: `(guild_id, hour_bucket)` - 1-hour TTL, stores `dict[user_id, summary]`
- **Historical daily summaries**: `(guild_id, date)` - permanent LRU, stores `dict[user_id, summary]`

**Per-User Operations:**
- **Historical summary**: `(guild_id, user_id, historical_end_date)` - permanent LRU, e.g., "2024-01-14" for days 2-7
- **Final context**: `(guild_id, user_id, hash(facts), hash(current_day), hash(historical))` - content-based hashing for encapsulation

### Memory Footprint Analysis
- **Per user**: ~6.5KB (facts: 400 chars, current day: 300 chars, historical: 500 chars, daily cache: 300×6, context: 800 chars)
- **5 users**: 32KB total
- **100 users**: 650KB total
- **Negligible memory usage** - can store entirely in-process
- **Current day overhead**: Minimal due to 1-hour TTL and single active bucket

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

### Automated Processing ✅ IMPLEMENTED
- **Chat History Ingestion**: All messages automatically stored in `chat_history` table
- **User ID Translation**: IDs converted to readable nicks only for LLM processing
- **On-Demand Calculation**: Summaries and context generated when requested via MemoryManager
- **Dual Caching Strategy**: TTL caching for current day, LRU caching for historical data using `cachetools`
- **Context Injection**: Memory context automatically included in AI responses via GeneralQueryGenerator

## Implementation Examples

### Memory Evolution Example
```
Day 1 Messages:
- "I'm so stressed about this project deadline"
- "Going to the gym after work to clear my head"

Daily Summary: "User experiencing work stress, using exercise as coping mechanism"

Day 2-7 Messages: [Similar pattern of stress + gym usage]

Historical Summary (Days 2-7): "User has consistent pattern of managing work stress through regular exercise, shows resilience and healthy coping strategies"

Current Day Summary: "User mentioned new project starting, seems excited but slightly anxious about timeline"

Factual Memory: "Software engineer at TechCorp, lives in Berlin, enjoys hiking"

Final Context: "Berlin-based software engineer at TechCorp who manages work stress through exercise and enjoys outdoor activities like hiking. Has consistent healthy coping strategies and is currently excited but slightly anxious about a new project timeline."
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

