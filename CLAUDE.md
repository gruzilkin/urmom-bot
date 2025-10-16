# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## urmom-bot Project Information

urmom-bot is a Discord bot that generates AI-powered "ur mom" jokes, country-specific humor, impersonates famous people, and maintains conversational memory about users. The bot uses multiple AI providers (Gemini Flash, Gemma, Grok, Claude) with automatic retry and fallback logic.

## Development Setup

### Environment Setup
1. Create virtual environment: `python -m venv .venv`
2. Activate: `source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Set up environment variables in `.env` file (see README.md)

### Running the Bot
- Local development: `cd bot && python app.py`
- Production: `docker compose up -d`
- View logs: `docker compose logs -f`

## Testing

### Unit Tests
Run unit tests only (excludes integration tests):
```bash
source .venv/bin/activate && PYTHONPATH=bot/src:bot/tests python -m unittest discover -s bot/tests/unit -p "*test*.py" -v
```

### Integration Tests
Run integration tests (requires API keys):
```bash
source .venv/bin/activate && PYTHONPATH=bot/src:bot/tests python -m unittest discover -s bot/tests/integration -p "*test*.py" -v
```

### Test Structure
- Unit tests: `bot/tests/unit/`
- Integration tests: `bot/tests/integration/`
- Testing framework: `unittest.IsolatedAsyncioTestCase` for async code
- **Telemetry Guidelines**:
  - Use `NullTelemetry()` from `tests.null_telemetry` in tests for classes requiring telemetry
  - Telemetry is a required dependency - never None or optional
  - Don't add defensive checks like `if self.telemetry:` - assume it's always present
  - Classes can safely call telemetry methods without null checks

## Architecture Overview

### Core Components
- **app.py**: Main Discord bot entry point with event handlers
- **container.py**: Dependency injection container for all services (see `bot/documents/di_refactor.md`)
- **ai_router.py**: Routes user messages to appropriate AI handlers (FAMOUS, GENERAL, FACT, NONE)
- **store.py**: PostgreSQL database interactions and guild configuration with LRU caching
- **schemas.py**: Pydantic schemas for structured AI responses (see `bot/documents/structured_output.md`)

### AI Clients
- **ai_client.py**: Abstract base class for AI providers
- **gemini_client.py**: Google Gemini Flash (fast, general-purpose, grounding)
- **gemma_client.py**: Google Gemma (structured output, language tasks)
- **grok_client.py**: xAI Grok (creative tasks, jokes, celebrity impersonation)
- **claude_client.py**: Anthropic Claude (fallback option)
- **ai_client_wrappers.py**: `RetryAIClient` and `CompositeAIClient` for reliability
  - **See `bot/documents/llm_fallback.md` for complete fallback strategy**

### Generators & Handlers
- **joke_generator.py**: Joke generation with adaptive learning
- **famous_person_generator.py**: Celebrity impersonation
- **general_query_generator.py**: General queries with memory context
- **fact_handler.py**: "remember" and "forget" commands
- **country_resolver.py**: Flag emoji to country mapping

### Memory & Conversation
- **memory_manager.py**: User memory system (facts + daily summaries)
  - **See `bot/documents/memory.md` for complete memory architecture**
- **conversation_graph.py**: Graph-based message threading and reply chain tracking
  - **See `bot/documents/conversation_history.md` for algorithm details**
- **message_node.py**: Message data structure with reply metadata

### Supporting Services
- **open_telemetry.py**: Observability and metrics collection
- **config.py**: Centralized configuration from environment variables
- **language_detector.py**: AI-powered language detection
  - **See `bot/documents/language_detection.md` for language handling architecture**
- **user_resolver.py**: Resolves user IDs to display names, handles mentions
- **attachment_processor.py**: Processes images and attachments for AI analysis
- **response_summarizer.py**: Condenses long AI responses to fit Discord limits

## Project Structure
- Main code: `bot/src/`
- Tests: `bot/tests/`
- Documentation: `bot/documents/` (detailed design docs)
- Virtual environment: `.venv/`
- Database: PostgreSQL with Docker setup
- Use `PYTHONPATH=bot/src:bot/tests` to set Python path

## Code Style Guidelines

### Type Hints
- **Required**: All functions and methods must have complete type hints
- **Modern syntax**: Use modern union syntax (`int | None` instead of `Optional[int]`)
- **Return types**: Always specify return types, including `None` when applicable
- **Parameters**: Type hint all parameters including `self` context when needed
- **Collections**: Use specific collection types (`list[str]` instead of `List[str]` when possible)

### Examples
```python
# Good - Complete type hints with modern syntax
async def fetch_message(message_id: int) -> MessageNode | None:
    pass

def process_data(items: list[str], count: int = 10) -> dict[str, Any]:
    pass

def create_handler() -> Callable[[str], Awaitable[None]]:
    pass

# Avoid - Missing or old-style type hints
def fetch_message(message_id):  # Missing types
    pass

def process_data(items: Optional[List[str]]) -> Dict[str, Any]:  # Old syntax
    pass
```

### Import Guidelines
- Use `from typing import Callable, Awaitable` for function type hints
- Import specific types needed rather than importing everything from typing

### Exception Logging
- **All exceptions must be logged at ERROR level** with full stack trace
- **Format**: `logger.error(f"descriptive error based on operation: {e}", exc_info=True)`
- **Description**: Include what operation was being attempted when the error occurred
- **Examples**:
  ```python
  # Good - Descriptive error with operation context
  except ValueError as e:
      logger.error(f"Failed to parse user configuration: {e}", exc_info=True)

  except APIException as e:
      logger.error(f"OpenAI API call failed during joke generation: {e}", exc_info=True)

  # Avoid - Generic or missing error logging
  except Exception as e:
      logger.warning(f"Error: {e}")  # Wrong level, no context, no stack trace
  ```

### Telemetry Span Attributes
- **Reuse attribute names consistently** across different components when they represent the same concept
- **Common attribute names** should be standardized for easier aggregation and analysis in observability tools
- **Cache hit tracking**: Whenever there's caching involved, surrounding open telemetry span should have `cache_hit` attribute
- **Avoid inconsistent naming**:
  ```python
  # Avoid - Different names for same concept across components
  span.set_attribute("confidence", score)           # In one component
  span.set_attribute("detection_confidence", score) # In another component
  span.set_attribute("lang_confidence", score)      # In third component
  ```
- **When creating new attributes**, check existing codebase for similar concepts and reuse existing names

## Development Workflow

### When to Read Detailed Documentation

Refer to `bot/documents/` for detailed design documentation:

- **Memory System** (`memory.md`):
  - When working on user memory, daily summaries, or context merging
  - Understanding caching strategy (TTL vs LRU vs database)
  - Memory operations (remember/forget)
  - Batch processing and concurrent operations

- **LLM Fallback Strategy** (`llm_fallback.md`):
  - When adding new AI-powered features
  - Choosing which AI client to use for a specific task
  - Understanding retry policies (max_time vs max_tries)
  - Setting up CompositeAIClient with proper fallback chains

- **Conversation History** (`conversation_history.md`):
  - When working on message threading or reply chains
  - Understanding the tik/tok graph exploration algorithm
  - Lazy processing and message materialization
  - Article extraction from embeds

- **Language Detection** (`language_detection.md`):
  - When adding multilingual support to new features
  - Understanding language propagation through the request chain
  - Handling translation requests and mixed-language scenarios

- **Structured Output** (`structured_output.md`):
  - When creating new Pydantic schemas for AI responses
  - Understanding how structured output works across different AI providers

- **Dependency Injection** (`di_refactor.md`):
  - When adding new services to the container
  - Understanding how components are wired together

### AI Client Usage Patterns
- **Prefer using wrapped clients** from `container.py` (e.g., `retrying_gemma`, `gemma_with_grok_fallback`) over bare clients
- **CompositeAIClient** automatically tries fallback clients when:
  - An exception occurs
  - Response validation fails (via `is_bad_response` callback)
- **RetryAIClient** supports two retry strategies:
  - `max_time`: Retry for up to N seconds (good for rate limits)
  - `max_tries`: Retry up to N attempts (good for transient failures)
- **Client selection guidelines** - see `bot/documents/llm_fallback.md` for complete decision tree

### Code Quality Checks
- **After making file changes**: Always run `ruff` to check files for linting issues and formatting
- Use `ruff check` for linting and `ruff format` for code formatting
