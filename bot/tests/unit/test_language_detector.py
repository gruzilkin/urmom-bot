import unittest
from unittest.mock import AsyncMock

from language_detector import LanguageCode, LanguageDecision, LanguageDetector
from null_telemetry import NullTelemetry


class TestLanguageDetector(unittest.IsolatedAsyncioTestCase):
    """Unit tests for LanguageDetector focusing on validation, caching, and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_ai_client = AsyncMock()
        self.telemetry = NullTelemetry()
        self.detector = LanguageDetector(ai_client=self.mock_ai_client, telemetry=self.telemetry)

    # Input Validation Tests

    async def test_empty_string_raises_error(self):
        """Empty string should raise ValueError."""
        with self.assertRaises(ValueError) as context:
            await self.detector.detect_language("")

        self.assertIn("empty", str(context.exception))
        self.mock_ai_client.generate_content.assert_not_called()

    async def test_whitespace_only_raises_error(self):
        """Whitespace-only text should raise ValueError."""
        with self.assertRaises(ValueError) as context:
            await self.detector.detect_language("   \n\t  ")

        self.assertIn("whitespace", str(context.exception))
        self.mock_ai_client.generate_content.assert_not_called()

    # Error Handling Tests

    async def test_ai_invalid_response_returns_english(self):
        """AI returning invalid response should fallback to English."""
        from language_detector import LanguageCode

        mock_response = LanguageCode(language_code="invalid123")  # Invalid code
        self.mock_ai_client.generate_content.return_value = mock_response

        result = await self.detector.detect_language("Text")

        self.assertEqual(result, "en")

    async def test_ai_exception_returns_english(self):
        """AI throwing exception should fallback to English."""
        self.mock_ai_client.generate_content.side_effect = Exception("AI failed")

        result = await self.detector.detect_language("Text")

        self.assertEqual(result, "en")

    # Language Name Caching Tests

    async def test_cached_language_names(self):
        """Cached language codes should return names without AI calls."""
        test_cases = [
            ("en", "English"),
            ("zh", "Chinese"),
            ("es", "Spanish"),
            ("fr", "French"),
            ("ru", "Russian"),
            ("ja", "Japanese"),
        ]

        for code, expected_name in test_cases:
            with self.subTest(code=code):
                result = await self.detector.get_language_name(code)
                self.assertEqual(result, expected_name)

        # AI should never be called for cached codes
        self.mock_ai_client.generate_content.assert_not_called()

    async def test_ai_name_resolution_failure_returns_fallback(self):
        """AI failing to resolve name should return fallback format."""
        self.mock_ai_client.generate_content.side_effect = Exception("AI failed")

        result = await self.detector.get_language_name("xyz")

        self.assertEqual(result, "Language-xyz")

    # Jev Tests

    async def test_jev_answer_is_used_without_llm(self):
        jev_client = AsyncMock()
        jev_client.ask.return_value = LanguageDecision(
            language={"choice": "de", "probabilities": {"de": 0.97, "en": 0.03}, "confidence": 0.96}
        )
        detector = LanguageDetector(ai_client=self.mock_ai_client, telemetry=self.telemetry, jev_client=jev_client)

        result = await detector.detect_language("Guten Tag")

        self.assertEqual(result, "de")
        self.mock_ai_client.generate_content.assert_not_called()

    async def test_jev_other_falls_back_to_llm(self):
        jev_client = AsyncMock()
        jev_client.ask.return_value = LanguageDecision(
            language={"choice": "OTHER", "probabilities": {"OTHER": 0.8, "en": 0.2}, "confidence": 0.7}
        )
        self.mock_ai_client.generate_content.return_value = LanguageCode(language_code="uk")
        detector = LanguageDetector(ai_client=self.mock_ai_client, telemetry=self.telemetry, jev_client=jev_client)

        result = await detector.detect_language("Привіт, як справи?")

        self.assertEqual(result, "uk")
        self.mock_ai_client.generate_content.assert_called_once()

    async def test_jev_failure_falls_back_to_llm(self):
        jev_client = AsyncMock()
        jev_client.ask.side_effect = RuntimeError("jev down")
        self.mock_ai_client.generate_content.return_value = LanguageCode(language_code="fr")
        detector = LanguageDetector(ai_client=self.mock_ai_client, telemetry=self.telemetry, jev_client=jev_client)

        result = await detector.detect_language("Bonjour")

        self.assertEqual(result, "fr")
        self.mock_ai_client.generate_content.assert_called_once()


if __name__ == "__main__":
    unittest.main()
