# Language Detection & Propagation Architecture

This document describes the language detection and propagation system that provides consistent multilingual handling across the urmom-bot Discord bot.

## Overview

The language detection system ensures that user messages are processed in their original language throughout the entire request chain. The system detects the user's language once at the entry point and propagates this information explicitly to all components, preventing language drift and ensuring consistent responses.

### Core Philosophy
- **Explicit over Implicit**: Language information is passed explicitly through the processing chain.
- **AI-Powered Accuracy**: A dedicated Large Language Model (LLM) is used for all language detection to ensure high accuracy, especially for short and ambiguous text common in chat.
- **Unambiguous Instructions**: Clear, full language names are used in prompts rather than ambiguous codes.
- **Single Source of Truth**: Language is detected once at the start of a request and propagated consistently.

## Architecture

### LLM-Powered Language Detection

The system uses a single, AI-powered strategy for all language detection, which is optimized for the chat-based nature of the bot.

- **Method**: A dedicated AI client performs all language detection.
- **Scope**: Applied to all incoming text, regardless of length. This provides consistent and accurate handling of short messages, which are common in Discord.
- **Enhanced Prompting**: The prompt sent to the LLM includes contextual guidance to improve accuracy for ambiguous cases.
- **Structured Output**: The LLM is constrained to return a valid, structured language code, ensuring the output is well-formed.
- **Validation**: The returned language code undergoes robust validation, with a fallback to English ('en') if invalid.

### Language Name Resolution
- **Purpose**: Convert language codes (e.g., "en", "ru") to full, unambiguous names (e.g., "English", "Russian") for clear instructions in generator prompts.
- **Cache**: A local cache stores mappings for the most common languages to reduce latency.
- **Fallback**: If a language code is not in the cache, the system queries the LLM to get the full language name. The result is then cached for future requests.

## Request Processing Flow

```
User Message → AI Router (Route Selection + Language Detection + Parameter Extraction) → Generators
```

**Processing Steps:**
1. **Route Selection**: Determine the request type (e.g., FAMOUS, GENERAL, FACT).
2. **Language Detection**: The `LanguageDetector` service uses an LLM to determine the language of the user's message.
3. **Parameter Extraction**: Extract route-specific parameters and populate the `language_code` and `language_name` fields in the request schema.
4. **Generator Processing**: The assigned generator uses the explicit `language_name` in its prompt to the LLM.

## Core Components

### Language Detection Service

This service provides two primary functions:
- **Language Detection**: Identifies the language of a given text using an LLM, with enhanced prompting for accuracy and robust validation.
- **Language Name Resolution**: Converts language codes into full, human-readable names, leveraging a cache for efficiency and falling back to an LLM when necessary.

### Schema Integration
All parameter schemas include:
- **language_code**: ISO 639-1 language code (e.g., "en", "ru", "de")
- **language_name**: Full language name (e.g., "English", "Russian", "German")
- **Population**: Fields populated automatically by the AI router after language detection.
- **Propagation**: Language information flows through the entire processing chain.

### Prompt Language Specification
- **Standard Pattern**: "Please respond in {language_name}"
- **Consistency**: Uniform language specification across all generators.
- **Clarity**: Simple, direct language instructions for reliable LLM behavior.

## Integration Points

- **AI Router**: Language detection occurs between route selection and parameter extraction.
- **All Generators**: Use explicit language parameter in prompts.
- **Components**: Centralized language detection across various components.
- **Container**: The `LanguageDetector` is integrated via the dependency injection container.

## Edge Cases & Language Handling

### Translation Requests
- **Example**: "BOT translate this into French"
- **Detection**: The primary language of the query is detected (e.g., English), and the LLM in the generator is responsible for interpreting the user's intent to translate.

### Mixed Language Scenarios
- **Example**: "BOT what does 'bonjour' mean?"
- **Strategy**: The system detects the primary language of the query. The generator's LLM is flexible enough to handle the mixed-language content and provide a relevant response.

### Ambiguous Cases
- **Fallback**: Detection failures or invalid language codes from the LLM default silently to English ('en').
- **Telemetry**: The language detection span is marked with an error status on failure, and relevant attributes track validation failures and fallbacks.
- **Graceful Degradation**: The system continues to function transparently, ensuring a response is always provided.

## Performance Considerations

### Performance Strategy
- **Optimized Prompts**: Prompts are designed to be efficient and get structured, accurate results from the LLM.
- **Caching**: Language name translations are cached locally to avoid repeated LLM queries for the same language codes, reducing latency.
- **Monitoring**: The system tracks key metrics, cache performance, and validation failures through telemetry.

## System Features

- **Language Preservation**: The user's original language is maintained throughout the complete request chain.
- **Override Support**: Translation and language change requests are handled correctly by the downstream generators.
- **Consistency**: The bot does not unexpectedly switch languages in its responses or memory operations.
- **Reliability**: Graceful degradation and fallback handling ensure stability.
- **Observability**: Comprehensive telemetry provides deep insights for monitoring and debugging.

## Conclusion

This architecture provides robust, consistent language handling while maintaining flexibility for complex language scenarios and edge cases. The system ensures reliable multilingual support across all bot features through explicit language propagation and an AI-powered detection strategy.