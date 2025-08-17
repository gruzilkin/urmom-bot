# General Request Processing Flow

This diagram shows the external interactions during GENERAL request processing in the urmom-bot Discord bot.

```mermaid
sequenceDiagram
    participant User as Discord User
    participant Discord as Discord API
    participant Bot as urmom-bot
    participant AIClient as AI Client (Gemini/Grok/Claude/Gemma)
    participant PostgreSQL as PostgreSQL Database

    User->>Discord: Send message mentioning bot
    Discord->>Bot: Webhook/WebSocket message event
    
    Bot->>PostgreSQL: Store message in transient memory
    
    Note over Bot: Internal routing & processing
    
    Bot->>AIClient: Route selection request
    AIClient-->>Bot: Route: "GENERAL"
    
    Bot->>AIClient: Language detection request
    AIClient-->>Bot: Language code & name
    
    Bot->>AIClient: Parameter extraction request
    AIClient-->>Bot: GeneralParams (ai_backend, temperature, cleaned_query)
    
    Bot->>Discord: Fetch conversation history
    Discord-->>Bot: Recent messages
    
    Bot->>Discord: Fetch user display names
    Discord-->>Bot: User information
    
    Bot->>PostgreSQL: Query user memories
    PostgreSQL-->>Bot: User memory data
    
    Bot->>AIClient: Generate response with context & memories
    AIClient-->>Bot: AI response text
    
    alt Response too long (>2000 chars)
        Bot->>AIClient: Summarization request
        AIClient-->>Bot: Summarized response
    end
    
    Bot->>Discord: Reply to user message
    Discord-->>User: Bot response delivered
```

## External Integrations

### Discord API
- **Inbound**: Webhook/WebSocket events for new messages
- **Outbound**: Fetch conversation history, user information, send replies

### AI Clients (Gemini/Grok/Claude/Gemma)
- **Route Selection**: Determine if request is GENERAL, FAMOUS, FACT, or NONE
- **Language Detection**: Identify message language for response localization
- **Parameter Extraction**: Extract ai_backend, temperature, and cleaned_query
- **Response Generation**: Generate final response with conversation context and memories
- **Summarization**: Compress responses exceeding Discord's 2000 character limit

### PostgreSQL Database
- **Message Storage**: Store messages in transient memory for context
- **User Memories**: Query stored facts about users for personalized responses

## AI Client Selection Logic

The extracted `ai_backend` parameter determines which AI service processes the request:
- **gemini_flash**: General questions, explanations, current events
- **grok**: Creative tasks, uncensored content  
- **claude**: Coding help, technical analysis, fact-checking
- **gemma**: Fallback option, explicit requests only