import unittest
from unittest.mock import AsyncMock
from language_detector import LanguageDetector
from null_telemetry import NullTelemetry


class TestLanguageDetector(unittest.IsolatedAsyncioTestCase):
    """Unit tests for LanguageDetector focusing on validation, caching, and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_ai_client = AsyncMock()
        self.telemetry = NullTelemetry()
        self.detector = LanguageDetector(
            ai_client=self.mock_ai_client,
            telemetry=self.telemetry
        )
    
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


if __name__ == '__main__':
    unittest.main()