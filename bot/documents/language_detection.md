# Language Detection & Propagation Architecture

This document describes the language detection and propagation system that provides consistent multilingual handling across the urmom-bot Discord bot.

## Overview

The language detection system ensures that user messages are processed in their original language throughout the entire request chain. The system detects the user's language once at the entry point and propagates this information explicitly to all components, preventing language drift and ensuring consistent responses.

### Core Philosophy
- **Explicit over Implicit**: Language information is passed explicitly through the processing chain
- **Hybrid Detection**: Fast offline detection combined with LLM fallback for accuracy
- **Unambiguous Instructions**: Clear language names in prompts rather than ambiguous codes
- **Single Source of Truth**: Language detected once and propagated consistently

## Architecture

### Hybrid Language Detection with Short-Text Bypass

The system uses a two-tier detection strategy that optimizes for both accuracy and performance:

#### Tier 1: Offline Detection (for longer text)
- **Method**: `langdetect` library with `detect_langs()` function
- **Threshold**: 95% confidence ensures high accuracy
- **Performance**: Fast offline detection for majority of clear cases
- **Scope**: Applied to text longer than 15 characters
- **Output**: Language code (e.g., "en", "ru", "de")

#### Tier 2: LLM Fallback (short text or low confidence)
- **Trigger**: Text ≤ 15 characters OR ambiguous text with confidence < 95%
- **AI Backend**: Gemma exclusively for language detection fallback
- **Enhanced Prompting**: Contextual bias for ambiguous cases (Cyrillic→Russian, Latin→English)
- **Capability**: Handles complex language scenarios, edge cases, and very short text
- **Output**: Validated language code with fallback to English

### Language Name Resolution
- **Purpose**: Convert language codes to unambiguous names for prompts
- **Cache**: Language code to language name mapping cached to avoid repeated LLM queries
- **Fallback**: Query Gemma for unknown language codes
- **Validation**: Extended language code support (e.g., "en-us", "zh-cn") with fallback to English for invalid codes

## Request Processing Flow

```
User Message → AI Router (Route Selection + Language Detection + Parameter Extraction) → Generators
```

**Processing Steps:**
1. **Route Selection**: Determine request type (FAMOUS, GENERAL, FACT)
2. **Language Detection**: Hybrid detection with short-text bypass
3. **Parameter Extraction**: Extract route-specific parameters and populate language fields
4. **Generator Processing**: Use explicit language instructions in prompts

## Core Components

### Language Detection Service
```python
class LanguageDetector:
    async def detect_language(text: str) -> str:
        # Short text bypass: ≤ 15 chars goes directly to LLM
        # For longer text: try langdetect with 95% confidence threshold
        # Fall back to Gemma with enhanced prompting for ambiguous cases
        # Validate language codes and return with fallback to English
    
    async def get_language_name(code: str) -> str:
        # Return cached name or query Gemma
        # Cache results to avoid repeated queries
```

### Schema Integration
All parameter schemas (`FamousPersonParams`, `GeneralQueryParams`, `FactParams`) include:
- **language_code**: ISO 639-1 language code (e.g., "en", "ru", "de")
- **language_name**: Full language name (e.g., "English", "Russian", "German")
- **Population**: Fields populated automatically by AI router after language detection
- **Propagation**: Language information flows through entire processing chain

### Prompt Language Specification
- **Standard Pattern**: "Please respond in {language_name}"
- **Consistency**: Uniform language specification across all generators
- **Clarity**: Simple, direct language instructions for reliable LLM behavior

## Integration Points

- **AI Router**: Language detection occurs between route selection and parameter extraction
- **All Generators**: Use explicit language parameter in prompts
- **Components**: Centralized language detection across `joke_generator.py` and other components
- **Container**: LanguageDetector uses Gemma client exclusively for consistency

## Edge Cases & Language Handling

### Translation Requests
- **Example**: "BOT translate this into French"
- **Detection**: Query language = English, Response language = French
- **Handling**: LLM interprets user intent within language guidelines

### Mixed Language Scenarios
- **Example**: "BOT what does 'bonjour' mean?"
- **Strategy**: Detect primary query language, LLM handles response appropriately
- **Flexibility**: LLM interprets user intent within language guidelines

### Ambiguous Cases
- **Fallback**: Detection failures default to English silently
- **Validation**: Invalid language codes automatically fallback to English
- **Telemetry**: Language detection span marked with error status on failure
- **Graceful Degradation**: System continues functioning transparently with comprehensive error handling

## Performance Considerations

### Performance Strategy
- **Fast Path**: Short-text bypass and 95% confidence threshold optimize LLM usage
- **Caching**: Language name translation cached to avoid repeated queries
- **Monitoring**: Track detection confidence, fallback frequency, and performance metrics

### Telemetry & Observability
- **Comprehensive Tracking**: Detection method, confidence levels, cache performance
- **Attribute Consistency**: Standardized telemetry attribute names for aggregation and analysis
- **Error Logging**: Structured error handling with proper logging levels and stack traces
- **Validation Tracking**: Monitor validation failures and fallback behavior

## System Features

The language detection system provides:

- **Language Preservation**: User's original language maintained through complete request chain
- **Override Support**: Translation and language change requests handled correctly
- **Consistency**: No unexpected language switching in responses or memory operations
- **Performance**: Minimal latency impact with proper telemetry integration
- **Reliability**: Graceful degradation and fallback handling for edge cases
- **Observability**: Comprehensive telemetry for monitoring and debugging

## Conclusion

This architecture provides robust, consistent language handling while maintaining flexibility for complex language scenarios and edge cases. The system ensures reliable multilingual support across all bot features through explicit language propagation and hybrid detection strategies.