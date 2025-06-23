# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## urmom-bot Project Information

urmom-bot is a Discord bot that generates AI-powered "ur mom" jokes, country-specific humor, and impersonates famous people. The bot uses multiple AI providers (Gemini, Grok, Claude) and includes adaptive learning from user reactions.

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
source .venv/bin/activate && PYTHONPATH=bot python -m unittest discover -s bot/tests/unit -p "*test*.py" -v
```

### Integration Tests
Run integration tests (requires API keys):
```bash
source .venv/bin/activate && PYTHONPATH=bot python -m unittest discover -s bot/tests/integration -p "*test*.py" -v
```

### Test Structure
- Unit tests: `bot/tests/unit/`
- Integration tests: `bot/tests/integration/`
- Testing framework: `unittest.IsolatedAsyncioTestCase` for async code
- Telemetry mocking: Use `NullTelemetry()` from `tests.null_telemetry`

## Architecture Overview

### Core Components
- **app.py**: Main Discord bot entry point with event handlers
- **container.py**: Dependency injection container for all services
- **ai_router.py**: Routes user messages to appropriate AI handlers
- **store.py**: PostgreSQL database interactions and guild configuration
- **schemas.py**: Pydantic schemas for structured AI responses

### AI Clients
- **gemini_client.py**: Google Gemini API integration
- **grok_client.py**: xAI Grok API integration  
- **claude_client.py**: Anthropic Claude API integration
- **ai_client.py**: Abstract base class for AI providers

### Generators
- **joke_generator.py**: Core joke generation with adaptive learning
- **famous_person_generator.py**: Celebrity impersonation responses
- **general_query_generator.py**: General AI query handling
- **country_resolver.py**: Flag emoji to country mapping

### Supporting Services
- **open_telemetry.py**: Observability and metrics collection
- **store.py**: Database operations and guild settings

### Database Schema
- **messages**: Stores Discord message content and language
- **jokes**: Links source messages to joke responses with reaction counts
- **guild_configs**: Per-server bot configuration

### Message Flow
1. Discord message → **app.py** event handler
2. **ai_router.py** determines handling strategy (famous person, general query, etc.)
3. Appropriate generator creates response using configured AI client
4. Response sent to Discord with optional archiving/auto-deletion

## Project Structure
- Main code: `bot/` directory
- Virtual environment: `.venv/`
- Database: PostgreSQL with Docker setup
- Use `PYTHONPATH=bot` to set Python path instead of `cd bot`

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