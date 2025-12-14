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
- Daily summarization pipeline with multi-day merge architecture
- Context assembly and merging with MemoryManager
- Caching strategy (TTL + LRU) for daily summaries

## Overview

The urmom-bot memory system is a sophisticated AI-powered memory architecture that maintains both permanent factual knowledge and transient episodic memories about users. The system provides rich contextual awareness by preserving detailed memories from the full week of user interactions.

**Key Design Decisions:**
- Uses Gemini Flash (primary) with Ollama Kimi long-timeout fallback for batch daily summaries; Gemma for context merging and fact operations (free tier used where possible)
- Optimized for small-scale deployment (5-100 users)
- `cachetools` caches for async-friendly TTL/LRU behavior (not built-in `@lru_cache`)
- Discord user ID storage with nickname translation only for LLM calls
- Clean MemoryManager interface hiding internal complexity
- **Resilience-first approach**: System prioritizes reliability over completeness with graceful degradation

## Architecture Principles

### Two-Track Memory System
1. **Factual Memory**: Permanent knowledge explicitly provided via commands
2. **Transient Memory**: Automatic extraction and summarization of chat interactions

### Multi-Day AI Integration
- Raw chat history → Daily summaries (Gemini Flash composite) → Final context (Gemma, multi-day merge)
- **Strategy**: `merge(facts, day_0..day_6)` with full week context
- Current day (day 0) updated approximately hourly via TTL cache
- Historical days (1–6) persisted in database and cached in-process
- AI merging preserves specific events and conversations from all 7 days
- Full week context maintains rich detail for personalized responses

### Cache Efficiency
- **Strategy**: TTL caching for current day; DB-first storage + LRU read cache for historical daily summaries
- Current day summaries: 1-hour TTL keyed by `(guild_id, date)`
- Historical daily summaries: persisted in `daily_chat_summaries` and cached in-process in `Store`
- Content-based hashing for final context merge to ensure cache correctness
- Manual async cache management for async methods
- On-demand calculation with caching to avoid redundant work

### Resilience & Best-Effort Design
- **Graceful Degradation**: System continues functioning even when AI services are unavailable
- **Database-First Reliability**: Factual memories from database always accessible regardless of AI quota
- **Fallback Hierarchy**: If AI operations fail, system returns available data in priority order (facts → current day → historical)
- **Best-Effort Philosophy**: Users always receive some memory context rather than complete failure
- **Comprehensive Observability**: Telemetry tracking provides visibility into system health and performance

## Database Schema

To avoid duplication, the canonical SQL schema lives in `db/init.sql`.

- User facts: stores one row per `(guild_id, user_id)` with the current memory blob and timestamp.
- Chat history: stores raw messages per guild/channel with timestamps and optional reply linkage.
- Daily chat summaries: stores per-user daily summaries per guild/date.

## MemoryManager Architecture ✅ IMPLEMENTED

### Clean Interface Design
The transient memory system is encapsulated in a `MemoryManager` class that provides a simple interface while hiding internal complexity:

```python
class MemoryManager:
    async def ingest_message(self, guild_id: int, message: MessageNode) -> None:
        """Store message in chat_history table"""
    
    async def get_memories(self, guild_id: int, user_ids: list[int]) -> dict[int, str | None]:
        """Returns unified context for multiple users via concurrent multi-day merge with resilient fallback
        
        Combines: factual memory + 7 days of daily summaries (day 0 through day 6)
        Returns: Dictionary mapping user_id to unified context string or None if no memories exist
        
        Resilience guarantees:
        - Database facts always returned even if AI operations fail
        - Graceful fallback to partial context when AI quota exhausted
        - Best-effort approach ensures users get available memory data
        - Concurrent processing for optimal performance
        
        All complex summarization pipeline happens internally with dual caching strategy.
        """
    
    async def get_memory(self, guild_id: int, user_id: int) -> str | None:
        """Single-user wrapper around batch get_memories method for backward compatibility"""
```

### Internal Implementation ✅ IMPLEMENTED
- **Message Storage**: Automatic ingestion to `chat_history` table
- **Concurrent Batch Processing**: Multiple users processed simultaneously with `asyncio.gather`
- **Daily Summarization**: Gemini client for batch daily summaries across all users
- **Context Merging**: Gemma client merges user facts with all available daily summaries (days 0–6)
- **Caching**: TTLCache (1-hour) for current day; historical daily summaries persisted in DB with an LRU read-through cache in `Store`
- **Content-Based Hashing**: Cache keys use content hashes for daily-summary input and final context merge
- **Exception Handling**: Graceful degradation with concurrent exception handling
- **Daily Summaries Caching**: Current day in-process TTL; historical days DB + in-process LRU cache

- **Batch Daily Summaries (Gemini Flash → Kimi Long Timeout)**: Multi-user daily summary generation per guild/date with resilient fallback chain
- **Context Merging (Gemma - Free)**: Per-user merge of facts + daily summaries
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
- Format with structured XML including ids and reply relationships:
  <message>
    <id/>
    <reply_to/?>
    <timestamp/>
    <author_id/>
    <author/>
    <content/>
  </message>
- Target user identification: Include both display name and user_id in prompts for precise identification

Daily Summarization (batch approach):
- Input: All messages from a specific date and list of all active users
- Process: Single Gemini API call analyzes entire day and generates summaries for all users
- Batch Cache: One cache entry per (guild_id, date) contains map of all user summaries
- Current day cache: 1-hour TTL keyed by (guild_id, date)
- Historical day storage: Persisted in daily_chat_summaries with an LRU read-through cache in Store
- Prompt focus areas:
  * Notable events or experiences mentioned by each user
  * Each user's mood and emotional state
  * Important interactions and topics they discussed
  * Behavioral patterns they exhibited
  * Information revealed about them through messages from others
- Output: Map of user_id to daily summary in third person

Context Merging (facts + weekly summaries):
- Input: Factual memory blob + up to 7 daily summaries (days 0–6)
- Process: Merge all sources using structured XML prompts
- Cache: Content-based hashing: (guild_id, user_id, hash(facts), hash(all_summaries))
- Merge strategy:
  * Prioritize factual information for accuracy
  * Balance recent observations with patterns across the week
  * Resolve conflicts intelligently, favoring factual data
  * Provide unified context for personalized conversation
- Output: Single unified context string for AI conversation processing
```

## On-Demand Processing

### Context Assembly Pipeline
1. **Concurrent Daily Summary Fetching**:
   - Fetch daily summaries for dates day_0..day_6 simultaneously using `asyncio.gather`
   - Current day (TTL-backed) and historical days (DB-backed) processed concurrently
   - Exception handling ensures partial results don't block the entire operation
2. **Concurrent Context Merging**:
   - Retrieve user facts from `user_facts` table for all users
   - Merge facts + all available daily summaries for all users simultaneously
   - Content-based caching ensures optimal efficiency
3. **Result Assembly**: Return dictionary mapping user_id to final context

### Memory Operations Integration
- **AI Router Integration**: Memory operations (remember/forget) integrated into AI routing
- **Freeform Support**: Natural language memory commands
- **Temperature 0**: Precise instruction following without creative interpretation

## Cache Strategy

### Storage Approach
- **Current day TTL cache**: In-process TTL cache in MemoryManager keyed by (guild_id, date)
- **Historical summaries DB**: Persisted in daily_chat_summaries with in-process LRU read cache in Store
- **Absolute Date Keys**: Immutable cache keys based on calendar dates
- **Sliding Windows**: Efficient temporal queries using date arithmetic

### Cache Key Design
Cache keys use different strategies based on operation type:

**Batch Operations (all users per guild):**
- **Current day summaries**: `(guild_id, date)` - 1-hour TTL, stores `dict[user_id, summary]`
- **Historical daily summaries**: `(guild_id, date)` - persisted in DB; Store maintains an LRU cache of `dict[user_id, summary]`

**Per-User Operations:**
- **Final context**: `(guild_id, user_id, hash(facts), hash(all_summaries))` - content-based hashing for encapsulation

### Cache Benefits
- **Stable keys**: Message ID boundaries don't change as time progresses
- **Efficient lookups**: O(1) retrieval for processed summaries
- **Minimal recomputation**: Only process new time windows
- **Memory efficiency**: Tiny footprint allows simple in-memory storage

## Daily Summary and Memory Recompute Flow

The memory system uses staleness-based caching with async background rebuilds to optimize response times while keeping data reasonably fresh.

### Staleness Thresholds

**Fresh (< 1 hour):**
- Return cached summaries immediately

**Stale (1-6 hours):**
- Return slightly stale data immediately
- Trigger fire-and-forget async rebuild in background
- Background task regenerates daily summaries and updates cache
- Background task precomputes merged contexts for affected users

**Too Stale (≥ 6 hours):**
- Force synchronous rebuild

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
- **User ID Translation**: IDs converted to display names only for LLM processing
- **On-Demand Calculation**: Summaries and context generated when requested via MemoryManager
- **Caching Strategy**: TTL caching for current day (MemoryManager), DB + LRU caching for historical data (Store)
- **Context Injection**: Memory context automatically included in AI responses via GeneralQueryGenerator

## Implementation Examples

### Memory Evolution Example
```
Day 1 Messages:
- "I'm so stressed about this project deadline"
- "Going to the gym after work to clear my head"

Daily Summary: "User experiencing work stress, using exercise as coping mechanism"

Days 1–6 Messages: [Similar pattern of stress + gym usage]

Weekly Pattern: "Consistent pattern of managing work stress through regular exercise, showing resilience and healthy coping strategies"

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
- **Concurrent Processing**: Multiple users processed simultaneously with `asyncio.gather`
- **Batch Operations**: Single API call generates daily summaries for all users
- **Content-Based Caching**: Final context uses content hashes for cache correctness
- **LRU Caching**: Automatic cache management with size limits (Store-level read cache)
- **TTL + DB**: TTL for current day, persisted historical data with in-process LRU cache
- **Message Retention**: Use 7-day window for summarization

### Scalability
- **Target Scale**: Optimized for small deployments (5-100 users)
- **On-Demand Processing**: No background jobs, everything calculated when needed
- **AI Clients**: Gemini Flash (daily summaries) + Gemma (context merging and facts)
- **Simple Architecture**: Minimal infrastructure requirements
- **Data Retention**: Summarization uses a rolling 7-day window; database retention is currently unbounded unless externally managed

## Privacy and Safety

### Content Processing
- **Direct Storage**: All message content stored without filtering
- **Temperature 0**: AI operations use deterministic responses
- **Conflict Resolution**: Handle contradictory information through AI merging
