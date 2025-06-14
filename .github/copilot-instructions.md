# urmom-bot Architecture Refactoring Plan

## Current Problems
- App class knows too much about request processor internals
- Manual routing logic that becomes unwieldy as processors are added

## Target Architecture

### 1. Multiple AI Client Instances
Four instances of AiClient:
- **Local**: OpenAI llama (http://localhost:1234/v1)
- **Cheap**: Gemini Flash (via OpenAI API format)
- **Witty**: Grok (https://api.x.ai/v1) 
- **Smart**: Gemini Pro 2.5 (via OpenAI API format)

### 2. AI Router with YAML Configuration
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

### 3. Request Processor Router
- Uses AI to intelligently route user messages to appropriate handlers
- Handlers register themselves with descriptions
- App doesn't need to know handler internals

### 4. App Class Responsibilities
**Direct handling** (no AI routing):
- Configuration commands (`!config`)

**AI-driven routing** for everything else:
- Use RequestProcessorRouter to determine appropriate handler
- Pass requests to handlers without knowing their internals

## 5. Modular Feature Design
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