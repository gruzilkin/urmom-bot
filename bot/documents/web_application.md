# Joke Management Web Application

## Overview
A lightweight web interface for managing Discord bot jokes stored in PostgreSQL. Provides administrators the ability to view, edit, and delete joke pairs that have been automatically saved based on user reactions.

## Purpose
The Discord bot automatically saves message pairs as "jokes" based on user reactions, but this process can sometimes capture non-jokes or low-quality content. This web interface allows manual curation and cleanup of the joke database.

## Architecture

### Technology Stack
- **FastAPI**: Web framework for API endpoints and HTML serving
- **Jinja2**: Template engine for dynamic HTML generation  
- **HTMX**: Frontend interactivity without JavaScript complexity
- **PostgreSQL**: Shared database with Discord bot
- **Simplified telemetry**: Basic logging and optional tracing

### System Integration
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Discord Bot   │    │   PostgreSQL     │    │  Web Interface  │
│                 │───▶│                  │◀───│                 │
│   (Port N/A)    │    │   (Port 5432)    │    │ (Port via nginx)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

Both applications run as separate processes but share the same database instance.

## Core Features

### 1. Joke Table View
- **Two-column layout**: Source message | Joke response
- **Pagination**: Handle large datasets efficiently
- **Metadata display**: Reaction counts, message IDs
- **Search/Filter**: Basic text-based filtering

### 2. Inline Editing
- **Click-to-edit**: HTMX-powered inline editing
- **Real-time updates**: Changes saved without page refresh
- **Validation**: Input sanitization and length limits
- **Cancel/Save**: Standard editing controls

### 3. Joke Management
- **Delete functionality**: Remove inappropriate joke pairs
- **Confirmation dialogs**: Prevent accidental deletions

## Database Integration

### Dedicated WebStore Class
The web application uses a focused `WebStore` class designed specifically for web operations:

```python
class WebStore:
    def __init__(self, telemetry, **db_params):
        self.telemetry = telemetry
        self.connection_params = db_params
        self.conn = None
    
    async def get_all_jokes(self, limit: int = 50, offset: int = 0)
    async def update_joke_content(self, joke_message_id: int, new_content: str)
    async def delete_joke(self, source_message_id: int, joke_message_id: int)
    async def get_jokes_count(self) -> int
    async def search_jokes(self, query: str, limit: int = 50)
```

### WebStore Methods
Web-specific database operations without Discord bot complexity:

- **get_all_jokes()**: Paginated retrieval with source and joke content
- **update_joke_content()**: Edit joke text with validation
- **delete_joke()**: Remove joke pairs and orphaned messages
- **get_jokes_count()**: Total count for pagination
- **search_jokes()**: Text search through joke content

## User Interface Design

### Layout Principles
- **Minimal design**: Focus on functionality over aesthetics
- **Table-centric**: Primary interface is a data table

### HTMX Integration
- **Partial updates**: Only modified table rows refresh
- **Progressive enhancement**: Works without JavaScript as fallback
- **Server-side rendering**: All logic handled by FastAPI
- **Minimal client state**: Stateless interactions

## Security Considerations

### Input Validation
- **SQL injection prevention**: Parameterized queries via Store class
- **XSS protection**: Jinja2 auto-escaping enabled
- **Content sanitization**: Length limits and character validation
- **CSRF protection**: Built-in FastAPI security features

## Deployment

### Docker Integration
New service added to existing `docker-compose.yaml`:

```yaml
web:
  build: .
  command: python web/web_app.py
  # Port mapping handled at deployment level
  env_file:
    - .env
  environment:
    POSTGRES_HOST: db
    # ... same DB config as bot
  depends_on:
    - db
```

### Environment Configuration
Reuses existing environment variables:
- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`
- `POSTGRES_USER`, `POSTGRES_PASSWORD`
- `OTEL_EXPORTER_OTLP_ENDPOINT`
- `SAMPLE_JOKES_COEF`

## Observability

### Simplified Telemetry
- **Basic logging**: Standard Python logging for error tracking
- **Optional tracing**: Lightweight OpenTelemetry spans for database operations
- **Development mode**: NullTelemetry for local development without overhead

## Development Workflow

### Local Development
```bash
# Start database
docker compose up db -d

# Run web interface (uses default port)
cd web
python web_app.py

# Access via nginx proxy or container port mapping
```

### Testing Strategy
- **Unit tests**: FastAPI route testing
- **Integration tests**: Database operation validation
- **UI tests**: HTMX interaction verification

## Future Enhancements

### Basic Search
- **Text filtering**: Search through joke content for specific terms

## Benefits

### Operational Advantages
- **Quality control**: Manual curation of joke database
- **Pattern identification**: Spot issues in automatic classification
- **Quick cleanup**: Efficient removal of inappropriate content
- **Visibility**: Clear view of bot learning patterns