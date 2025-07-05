"""
Language detection service with two-tier detection strategy.

Implements hybrid detection using langdetect for high-confidence cases
and LLM fallback for ambiguous scenarios.
"""

import logging
import re
from typing import Optional
from langdetect import detect, detect_langs, LangDetectException, DetectorFactory
from ai_client import AIClient
from open_telemetry import Telemetry
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Set seed for deterministic results
DetectorFactory.seed = 0


class LanguageCode(BaseModel):
    """Schema for language code detection."""
    language_code: str = Field(description="ISO 639-1 language code (e.g., 'en', 'ru', 'de')")


class LanguageName(BaseModel):
    """Schema for language name resolution."""
    language_name: str = Field(description="Full name of the language in English (e.g., 'English', 'Russian', 'German')")


class LanguageDetector:
    """Two-tier language detection with offline and LLM fallback."""
    
    def __init__(self, ai_client: AIClient, telemetry: Telemetry):
        self.ai_client = ai_client
        self.telemetry = telemetry
        
        # Cache for language code to name mapping (10 most spoken languages)
        self._language_names = {
            "en": "English",
            "zh": "Chinese", 
            "es": "Spanish",
            "fr": "French",
            "ru": "Russian",
            "ja": "Japanese"
        }
    
    async def _detect_language_with_llm(self, text: str) -> str:
        """
        Detects language using only the LLM, with enhanced prompting for ambiguity.

        Args:
            text: Input text to analyze.

        Returns:
            The detected language code.
        """
        async with self.telemetry.async_create_span("detect_language_with_llm") as span:
            span.set_attribute("text", text)
            prompt = f"""Analyze the text and determine its primary language. Return only the ISO 639-1 code.

Text: "{text}"

IMPORTANT INSTRUCTIONS:
1.  If the text uses Cyrillic letters and is ambiguous, gravitate towards Russian ('ru').
2.  If the text uses Latin letters and is ambiguous (e.g., "ok", "ciao"), gravitate towards English ('en') or the most common language (e.g., Italian 'it' for "ciao").
"""

            response = await self.ai_client.generate_content(
                message=text,
                prompt=prompt,
                temperature=0.0,
                response_schema=LanguageCode
            )

            detected_lang = response.language_code.strip().lower()
            
            # Basic validation - should be at least 2 chars and follow language code patterns
            # Supports: "en", "eng", "en-us", "zh-cn", etc.
            if not re.match(r'^[a-z]{2,3}(-[a-z]{2,4})?$', detected_lang):
                logger.warning(f"Invalid language code from LLM: {detected_lang}, defaulting to 'en'")
                span.set_attribute("detected_language", "en")
                span.set_attribute("validation_failed", True)
                return "en"
            
            span.set_attribute("detected_language", detected_lang)
            return detected_lang

    async def detect_language(self, text: str) -> str:
        """
        Detect language using a hybrid approach with a short-text bypass.

        - For text < 15 chars, uses LLM directly.
        - For longer text, uses high-confidence offline detection first, then LLM fallback.

        Args:
            text: Input text to analyze.

        Returns:
            Language code (e.g., 'en', 'ru', 'de').
        """
        async with self.telemetry.async_create_span("detect_language") as span:
            span.set_attribute("text_length", len(text))

            if not text or not text.strip():
                raise ValueError("Text cannot be empty or whitespace-only")

            if len(text) > 15:
                try:
                    lang_probs = detect_langs(text)
                    if lang_probs:
                        result = lang_probs[0]
                        if result.prob >= 0.95:
                            span.set_attribute("confidence", result.prob)
                            span.set_attribute("detection_method", "offline")
                            span.set_attribute("detected_language", result.lang)
                            return result.lang
                except LangDetectException as e:
                    logger.warning(f"Offline language detection failed: {e}", exc_info=True)
                    span.record_exception(e)

            try:
                detected_lang = await self._detect_language_with_llm(text)
                span.set_attribute("detection_method", "llm")
                span.set_attribute("detected_language", detected_lang)
                return detected_lang
            except Exception as e:
                logger.error(f"LLM language detection failed: {e}", exc_info=True)
                span.set_attribute("detection_method", "default")
                span.set_attribute("detected_language", "en")
                span.record_exception(e)
                return "en"

    async def get_language_name(self, language_code: str) -> str:
        """
        Get full language name from language code.
        
        Args:
            language_code: ISO 639-1 language code (e.g., 'en', 'ru')
            
        Returns:
            Full language name (e.g., 'English', 'Russian')
        """
        async with self.telemetry.async_create_span("get_language_name") as span:
            span.set_attribute("language_code", language_code)
            
            # Check cache first
            if language_code in self._language_names:
                language_name = self._language_names[language_code]
                span.set_attribute("language_name", language_name)
                span.set_attribute("cache_hit", True)
                return language_name
            
            span.set_attribute("cache_hit", False)
            
            prompt = f"""What is the full name of the language with ISO 639-1 code '{language_code}'?
                        Provide the language name in English (e.g., 'German' for 'de', 'Russian' for 'ru')."""
            
            try:
                response = await self.ai_client.generate_content(
                    message=f"Language code: {language_code}",
                    prompt=prompt,
                    temperature=0.0,
                    response_schema=LanguageName
                )
                
                language_name = response.language_name.strip().title()
                
                self._language_names[language_code] = language_name            
                return language_name
            except Exception as e:
                logger.error(f"Failed to resolve language name for code {language_code}: {e}", exc_info=True)
                span.record_exception(e)
                return f"Language-{language_code}"