# urmom-bot Architecture Refactoring Plan

## Current Problems
- AIClient interface mixes low-level AI adapter methods with high-level domain-specific logic
- App class knows too much about request processor internals
- Manual routing logic that becomes unwieldy as processors are added
- Domain logic (joke detection, famous person detection) tightly coupled to AI client

## Target Architecture

### 1. Clean AI Client Interface
- Single method: `generate_content(message, prompt, samples)` 
- No domain-specific methods (remove `is_joke`, `is_famous_person_request`, etc.)
- Pure adapter pattern for AI services

### 2. Multiple AI Client Instances
Four instances of the same OpenAI client class:
- **Local**: OpenAI llama (http://localhost:1234/v1)
- **Cheap**: Gemini Flash (via OpenAI API format)
- **Witty**: Grok (https://api.x.ai/v1) 
- **Smart**: Gemini Pro 2.5 (via OpenAI API format)

### 3. AI Router with YAML Configuration
- Contains all 4 AI client instances
- Routes operations based on YAML config (MultiMap: operation -> ordered list of clients)
- Provides fallback mechanism (tries clients in order until one succeeds)
- Handles telemetry, logging, error handling

Example mapping:
```yaml
operation_mappings:
  IS_JOKE: [local, cheap]
  GENERATE_JOKE: [witty, smart]
  IS_FAMOUS_PERSON_REQUEST: [local, cheap]
  GENERATE_FAMOUS_PERSON_RESPONSE: [witty, smart]
  IS_QUERY: [local, cheap]
  QUERY: [smart]
  ROUTE_REQUEST: [local, cheap]
```

### 4. Request Processor Router
- Uses AI to intelligently route user messages to appropriate handlers
- Handlers register themselves with descriptions
- App doesn't need to know handler internals
- Easily extensible - just register new handlers

### 5. Domain-Specific Handlers
Each handler:
- Builds its own prompts
- Calls AIRouter with operation names (not specific clients)
- Handles its own logic (e.g., JokeGenerator handles joke detection and storage)
- Uses dependency injection for AIRouter

Examples:
- **JokeGenerator**: Handles `is_joke()`, `generate_joke()`
- **FamousPersonGenerator**: Handles `is_famous_person_request()`, `generate_famous_person_response()`
- **QueryHandler**: Handles `is_query()`, `handle_query()`

### 6. App Class Responsibilities
**Direct handling** (no AI routing):
- Emoji reactions
- Configuration commands (`!config`)

**AI-driven routing** for everything else:
- Use RequestProcessorRouter to determine appropriate handler
- Pass requests to handlers without knowing their internals

**Special case - Bot mentions/tags**:
- Validate with `IS_QUERY` first
- Handle as query with conversation history
- Include bot messages with `assistant` role, user messages with `user` role

### 7. QUERY Command
- Triggered by tagging bot user
- Fetches chat history including bot messages
- Bot messages get `assistant` role, user messages get `user` role
- Validates request with `IS_QUERY` operation before processing
- Uses Smart AI client for actual query processing
- Provides full conversation context to Smart model

## Key Benefits
1. **Separation of Concerns**: AI clients are pure adapters, domain logic in handlers
2. **Fallback Mechanisms**: Multiple AI clients with automatic failover
3. **Configuration-Driven**: Easy to change AI client priorities via YAML
4. **Extensible**: Add new handlers without changing app logic
5. **AI-Driven Routing**: Intelligent request routing without manual condition chains
6. **Testable**: Clean interfaces make unit testing easier
7. **Smart Query Handling**: Full conversation context with proper role assignment

## 8. Modular Feature Design
Instead of domain-specific handlers, implement a modular feature system:

### DiscordFeature Interface
Each feature implements:
- `handle_message(message, context)` -> Optional[str] 
- `handle_reaction(payload, context)` -> Optional[str]
- `get_description()` -> str (for AI routing)

### DiscordContext
Provides common Discord operations:
- `get_conversation_history()` - moves from app.py
- `fetch_message()` - common Discord API calls
- Access to bot instance for Discord operations
- **Note**: Features access store and telemetry directly, not through context

### Feature Examples
- **JokeFeature**: Handles clown emoji reactions, generates jokes, accesses store directly
- **FamousPersonFeature**: Handles "what would X say" requests
- **QueryFeature**: Handles general AI queries when bot is mentioned
- **CountryJokeFeature**: Handles flag emoji reactions

### App Class Responsibilities
- Discord boilerplate (event handlers, message parsing)
- Configuration commands (direct handling)
- Feature registration and routing
- Provides DiscordContext to features
- **App has zero knowledge of feature internals**

## Implementation Priority
1. Clean AIClient interface
2. YAML configuration for AI clients and operation mappings
3. AIRouter with fallback logic and telemetry
4. DiscordContext and DiscordFeature interface
5. Convert existing handlers to modular features
6. FeatureRouter with AI-based message routing
7. Update App class to use modular features
8. Implement QUERY command as QueryFeature