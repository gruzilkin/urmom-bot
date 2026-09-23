"""
Language detection service.

Detects the language of Discord messages with Jev (a single choice over the most spoken languages
plus OTHER) when a JevClient is configured. OTHER, a Jev failure, or no Jev client all fall back to
an LLM, which also resolves language names for codes outside the Jev list.
"""

import logging
import re

from pydantic import BaseModel, Field

from ai_client import AIClient
from jev_client import Choice, JevClient, choice
from open_telemetry import Telemetry

logger = logging.getLogger(__name__)

JEV_LANGUAGES: dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "ar": "Arabic",
    "ru": "Russian",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
    "pl": "Polish",
}
OTHER = "OTHER"

DETECTION_INSTRUCTIONS = """Detect the language the user is writing in.

Focus on the words and grammar the user uses to ask the question, give the instruction,
or make the statement. A message can contain names, foreign words, quotations, code,
URLs, or text from another language. Ignore that included content when the surrounding
message is written in a different language.

If the message is only a word or short phrase, identify the language in which it is
normally used. If there is still no clear answer, prefer Russian for Cyrillic text and
English for Latin text.

Detect only the language of the user's message. Do not answer the message, and do not
choose a language merely because the message mentions or discusses it."""


class LanguageCode(BaseModel):
    """Schema for LLM language code detection."""

    language_code: str = Field(description="ISO 639-1 language code (e.g., 'en', 'ru', 'de')")


class LanguageName(BaseModel):
    """Schema for language name resolution."""

    language_name: str = Field(
        description="Full name of the language in English (e.g., 'English', 'Russian', 'German')"
    )


class LanguageDecision(BaseModel):
    """Jev question: one choice over the most spoken languages, or OTHER."""

    language: Choice[str] = choice(
        instructions=DETECTION_INSTRUCTIONS,
        criteria={**JEV_LANGUAGES, OTHER: "Any language not listed above"},
    )


class LanguageDetector:
    """Detects the language of a message with Jev when available, otherwise an LLM."""

    def __init__(self, ai_client: AIClient, telemetry: Telemetry, jev_client: JevClient | None = None):
        self.ai_client = ai_client
        self.telemetry = telemetry
        self.jev_client = jev_client
        self._language_names: dict[str, str] = dict(JEV_LANGUAGES)

    async def _detect_language_with_jev(self, text: str) -> str | None:
        """Returns the language code, or None when Jev answers OTHER."""
        async with self.telemetry.async_create_span("detect_language_with_jev") as span:
            span.set_attribute("text", text)
            decision = await self.jev_client.ask(state={"message": text}, questions=LanguageDecision)
            answer = decision.language
            span.set_attribute("detected_language", answer.choice)
            span.set_attribute("confidence", answer.confidence)
            span.set_attribute("top_probability", answer.top_probability)
            return None if answer.choice == OTHER else answer.choice

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
            prompt = f"{DETECTION_INSTRUCTIONS}\n\nReturn its lowercase ISO 639-1 code in the required schema."

            response = await self.ai_client.generate_content(
                message=text, prompt=prompt, temperature=0.0, response_schema=LanguageCode
            )

            detected_lang = response.language_code.strip().lower()

            # Basic validation - should be at least 2 chars and follow language code patterns
            # Supports: "en", "eng", "en-us", "zh-cn", etc.
            if not re.match(r"^[a-z]{2,3}(-[a-z]{2,4})?$", detected_lang):
                logger.warning(f"Invalid language code from LLM: {detected_lang}, defaulting to 'en'")
                span.set_attribute("detected_language", "en")
                span.set_attribute("validation_failed", True)
                return "en"

            span.set_attribute("detected_language", detected_lang)
            return detected_lang

    async def detect_language(self, text: str) -> str:
        """
        Detect the language of a message.

        Tries Jev first when configured; falls back to the LLM when Jev answers OTHER or fails,
        and defaults to English if the LLM fails too.

        Args:
            text: Input text to analyze.

        Returns:
            Language code (e.g., 'en', 'ru', 'de').
        """
        async with self.telemetry.async_create_span("detect_language") as span:
            span.set_attribute("text_length", len(text))

            if not text or not text.strip():
                raise ValueError("Text cannot be empty or whitespace-only")

            if self.jev_client is not None:
                try:
                    detected_lang = await self._detect_language_with_jev(text)
                    if detected_lang is not None:
                        span.set_attribute("detection_method", "jev")
                        span.set_attribute("detected_language", detected_lang)
                        return detected_lang
                    logger.info("Jev answered OTHER for language, falling back to LLM")
                except Exception as e:
                    logger.error(f"Jev language detection failed, falling back to LLM: {e}", exc_info=True)
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
                    response_schema=LanguageName,
                )

                language_name = response.language_name.strip().title()

                self._language_names[language_code] = language_name
                span.set_attribute("language_name", language_name)
                return language_name
            except Exception as e:
                logger.error(f"Failed to resolve language name for code {language_code}: {e}", exc_info=True)
                span.record_exception(e)
                return f"Language-{language_code}"
