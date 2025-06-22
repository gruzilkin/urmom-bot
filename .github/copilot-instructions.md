# urmom-bot Architecture Refactoring Plan

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

### 3. Feature Base Interface
- Abstract base class for all feature handlers
- Unified interface for feature registration and execution
- Standardized method signatures across all features

### 4. Request Processor Router
- Uses AI to intelligently route user messages to appropriate handlers
- Handlers register themselves with descriptions
- App doesn't need to know handler internals

### 5. App Class Responsibilities
**Direct handling** (no AI routing):
- Configuration commands (`!config`)

**AI-driven routing** for everything else:
- Use RequestProcessorRouter to determine appropriate handler
- Pass requests to handlers without knowing their internals

## Testing Standards

**Use only `unittest.IsolatedAsyncioTestCase`** for async code - never pytest or other frameworks.

**Project-specific mocking patterns:**
```python
# AI Client mocking
self.ai_client = Mock()
self.ai_client.generate_content = AsyncMock(return_value="expected response")

# Telemetry - ALWAYS use NullTelemetry()
from tests.null_telemetry import NullTelemetry
self.telemetry = NullTelemetry()

# Integration tests - skip if API keys missing
api_key = os.getenv('API_KEY')
if not api_key:
    self.skipTest("API_KEY environment variable not set")
```