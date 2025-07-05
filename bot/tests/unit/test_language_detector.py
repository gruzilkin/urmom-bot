import unittest
from unittest.mock import AsyncMock, patch
from language_detector import LanguageDetector
from tests.null_telemetry import NullTelemetry


class TestLanguageDetector(unittest.IsolatedAsyncioTestCase):
    """Unit tests for LanguageDetector with no real LLM calls."""
    
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
    
    # Offline Detection Tests
    
    async def test_high_confidence_offline_detection(self):
        """High confidence text should be detected offline without AI calls."""
        test_cases = [
            ("Hello, how are you today? This is a longer English text to ensure high confidence detection.", "en"),
            ("Hola, ¿cómo estás? Este es un texto en español más largo para asegurar una detección de alta confianza.", "es"),
            ("Bonjour, comment allez-vous? Ceci est un texte français plus long pour assurer une détection de haute confiance.", "fr"),
            ("Guten Tag, wie geht es Ihnen? Dies ist ein längerer deutscher Text für eine hochvertrauensvolle Erkennung.", "de"),
            ("Привет, как дела? Это более длинный русский текст для обеспечения высокоуверенного обнаружения.", "ru"),
            ("こんにちは、元気ですか？これは言語検出ライブラリによる高信頼度検出を確実にするためのより長い日本語のテキストです。", "ja"),
        ]
        
        for text, expected_lang in test_cases:
            with self.subTest(text=text[:50] + "..."):
                result = await self.detector.detect_language(text)
                self.assertEqual(result, expected_lang)
        
        # No AI calls should be made for high-confidence detection
        self.mock_ai_client.generate_content.assert_not_called()
    
    # AI Fallback Tests
    
    @patch('language_detector.detect_langs')
    async def test_langdetect_exception_triggers_ai_fallback(self, mock_detect_langs):
        """LangDetectException should trigger AI fallback."""
        from langdetect import LangDetectException
        from language_detector import LanguageCode
        
        mock_detect_langs.side_effect = LangDetectException("Detection failed", "Detection failed")
        mock_response = LanguageCode(language_code="fr")
        self.mock_ai_client.generate_content.return_value = mock_response
        
        result = await self.detector.detect_language("Some text")
        
        self.assertEqual(result, "fr")
        self.mock_ai_client.generate_content.assert_called_once()
    
    async def test_mixed_language_handling(self):
        """Mixed language text should be handled gracefully by the detection system."""
        from language_detector import LanguageCode
        
        # Text with mixed languages - English request with Spanish payload
        mixed_text = "Please translate this Spanish text: Hola me llamo María y vivo en España con mi familia"
        
        mock_response = LanguageCode(language_code="en")
        self.mock_ai_client.generate_content.return_value = mock_response
        
        result = await self.detector.detect_language(mixed_text)
        
        # Should return a valid language code (either offline detection or AI fallback)
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 2)
        self.assertTrue(result.isalpha())
        # The specific language detected depends on the offline detector's confidence
    
    @patch('language_detector.detect_langs')
    async def test_ai_fallback_with_invalid_response_returns_english(self, mock_detect_langs):
        """AI returning invalid response should fallback to English."""
        from language_detector import LanguageCode
        
        mock_detect_langs.return_value = []
        mock_response = LanguageCode(language_code="invalid123")  # Invalid code
        self.mock_ai_client.generate_content.return_value = mock_response
        
        result = await self.detector.detect_language("Text")
        
        self.assertEqual(result, "en")
    
    @patch('language_detector.detect_langs')
    async def test_ai_exception_returns_english(self, mock_detect_langs):
        """AI throwing exception should fallback to English."""
        mock_detect_langs.return_value = []
        self.mock_ai_client.generate_content.side_effect = Exception("AI failed")
        
        result = await self.detector.detect_language("Text")
        
        self.assertEqual(result, "en")
    
    # Language Name Resolution Tests
    
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
    
    async def test_unknown_language_code_calls_ai(self):
        """Unknown language code should call AI and cache result."""
        from language_detector import LanguageName
        
        mock_response = LanguageName(language_name="German")
        self.mock_ai_client.generate_content.return_value = mock_response
        
        # First call should trigger AI
        result1 = await self.detector.get_language_name("de")
        self.assertEqual(result1, "German")
        self.mock_ai_client.generate_content.assert_called_once()
        
        # Second call should use cache
        self.mock_ai_client.reset_mock()
        result2 = await self.detector.get_language_name("de")
        self.assertEqual(result2, "German")
        self.mock_ai_client.generate_content.assert_not_called()
    
    async def test_ai_name_resolution_failure_returns_fallback(self):
        """AI failing to resolve name should return fallback format."""
        self.mock_ai_client.generate_content.side_effect = Exception("AI failed")
        
        result = await self.detector.get_language_name("xyz")
        
        self.assertEqual(result, "Language-xyz")
    
    # Behavioral Tests
    
    async def test_deterministic_behavior(self):
        """Same input should always produce same output."""
        text = "Hello world, this is a test of deterministic language detection behavior."
        
        result1 = await self.detector.detect_language(text)
        result2 = await self.detector.detect_language(text)
        result3 = await self.detector.detect_language(text)
        
        self.assertEqual(result1, result2)
        self.assertEqual(result2, result3)
        self.assertEqual(result1, "en")


if __name__ == '__main__':
    unittest.main()