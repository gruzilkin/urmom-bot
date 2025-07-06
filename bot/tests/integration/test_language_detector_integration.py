import os
import unittest
from dotenv import load_dotenv
from gemma_client import GemmaClient
from language_detector import LanguageDetector
from tests.null_telemetry import NullTelemetry

# Load environment variables from .env file
load_dotenv()

class TestLanguageDetectorIntegration(unittest.IsolatedAsyncioTestCase):
    """
    Integration tests for LanguageDetector with real AI detection.
    These tests make real API calls to the Gemma service and will be skipped if
    the required environment variables are not set.
    
    Tests comprehensive language detection accuracy for Discord messages and
    various text types that caused issues with the previous langdetect approach.
    """

    def setUp(self):
        """Set up the test environment, skipping if API keys are missing."""
        self.gemma_api_key = os.getenv('GEMINI_API_KEY')
        self.gemma_model_name = os.getenv('GEMINI_GEMMA_MODEL')
        
        if not self.gemma_api_key:
            self.skipTest("GEMINI_API_KEY not found, skipping LanguageDetector integration tests.")
        if not self.gemma_model_name:
            self.skipTest("GEMINI_GEMMA_MODEL not found, skipping LanguageDetector integration tests.")

        telemetry = NullTelemetry()
        # This test specifically targets Gemma, so we instantiate it directly.
        gemma_client = GemmaClient(
            api_key=self.gemma_api_key,
            model_name=self.gemma_model_name,
            telemetry=telemetry
        )
        self.detector = LanguageDetector(
            ai_client=gemma_client,
            telemetry=telemetry
        )

    async def test_llm_fallback_for_short_ambiguous_text(self):
        """
        Test that very short or ambiguous text, which is hard for offline detectors,
        triggers the LLM fallback and gets a correct detection.
        """
        # These phrases often fail high-confidence offline detection.
        test_cases = {
            "ok": "en",
            "ciao": "it",
            "你好": "zh",
            "Привет": "ru"
        }

        for text, expected_lang in test_cases.items():
            with self.subTest(text=text):
                # This should trigger the LLM fallback.
                detected_lang = await self.detector.detect_language(text)
                self.assertEqual(detected_lang, expected_lang)

    async def test_llm_fallback_for_translation_request(self):
        """
        Test a tricky case where the user's instruction is in one language
        but the payload is in another. The language of the instruction should be detected.
        """
        # The user wants the bot to respond in French, so the detected language should be 'fr'.
        # The French instruction is the most important signal.
        text = "répond en français: 'I love programming and I want to write a lot of code in Python.'"
        expected_lang = "fr"

        detected_lang = await self.detector.detect_language(text)
        self.assertEqual(detected_lang, expected_lang)

    async def test_llm_fallback_for_mixed_language_query(self):
        """
        Test a case where the user asks a question in one language about a phrase
        in another. The primary language of the question should be detected.
        """
        # The user is asking a question in English, so the language should be 'en'.
        text = "what does 'wie geht es Ihnen?' mean in English?"
        expected_lang = "en"

        detected_lang = await self.detector.detect_language(text)
        self.assertEqual(detected_lang, expected_lang)

    async def test_comprehensive_language_detection(self):
        """
        Test AI language detection across multiple languages with real text.
        This replaces the old unit test that only mocked responses.
        """
        test_cases = [
            ("Hello, how are you today? This is a longer English text.", "en"),
            ("Hola, ¿cómo estás? Este es un texto en español más largo.", "es"),
            ("Bonjour, comment allez-vous? Ceci est un texte français.", "fr"),
            ("Guten Tag, wie geht es Ihnen? Dies ist ein deutscher Text.", "de"),
            ("Привет, как дела? Это русский текст для проверки.", "ru"),
            ("こんにちは、元気ですか？これは日本語のテキストです。", "ja"),
        ]
        
        for text, expected_lang in test_cases:
            with self.subTest(text=text[:50] + "..."):
                result = await self.detector.detect_language(text)
                self.assertEqual(result, expected_lang)
    
    async def test_discord_message_scenario(self):
        """
        Test the specific scenario that caused Indonesian misdetection.
        This was the original issue: "did Putin bomb Kiev again?" was
        incorrectly detected as Indonesian with 99.999% confidence by langdetect.
        """
        # The exact message that caused the problem
        text = "did Putin bomb Kiev again?"
        expected_lang = "en"
        
        result = await self.detector.detect_language(text)
        self.assertEqual(result, expected_lang)
    
    async def test_very_short_discord_messages(self):
        """
        Test very short Discord messages that langdetect struggled with.
        """
        test_cases = [
            ("lol", "en"),
            ("wtf", "en"), 
            ("omg", "en"),
            ("thx", "en"),
            ("ok", "en"),
        ]
        
        for text, expected_lang in test_cases:
            with self.subTest(text=text):
                result = await self.detector.detect_language(text)
                self.assertEqual(result, expected_lang)
    
    async def test_deterministic_behavior_with_real_ai(self):
        """
        Test that same input produces consistent output with real AI.
        Temperature is set to 0.1 in GemmaClient for deterministic results.
        """
        text = "Hello world, this is a test of deterministic language detection."
        
        result1 = await self.detector.detect_language(text)
        result2 = await self.detector.detect_language(text)
        result3 = await self.detector.detect_language(text)
        
        self.assertEqual(result1, result2)
        self.assertEqual(result2, result3)
        self.assertEqual(result1, "en")
    
    async def test_get_language_name_from_llm(self):
        """
        Test that get_language_name can resolve a non-cached language code
        by making a real call to the LLM.
        """
        # Use a language code that is unlikely to be in the small default cache.
        language_code = "sv"
        expected_name = "Swedish"

        # Ensure the code is not in the cache before the test.
        if language_code in self.detector._language_names:
            del self.detector._language_names[language_code]

        # This should trigger an LLM call.
        language_name = await self.detector.get_language_name(language_code)
        self.assertEqual(language_name, expected_name)

        # Verify that the result is now cached to prevent future LLM calls.
        self.assertIn(language_code, self.detector._language_names)
        self.assertEqual(self.detector._language_names[language_code], expected_name)
    
    async def test_caching_behavior_with_real_ai(self):
        """
        Test that unknown language code calls AI once then caches result.
        This replaces the unit test that only tested mocking behavior.
        """
        # Use a language code that's not in the default cache
        language_code = "it"
        expected_name = "Italian"
        
        # Ensure not cached
        if language_code in self.detector._language_names:
            del self.detector._language_names[language_code]
        
        # First call should trigger AI and cache result
        result1 = await self.detector.get_language_name(language_code)
        self.assertEqual(result1, expected_name)
        self.assertIn(language_code, self.detector._language_names)
        
        # Second call should use cache (we can't directly verify no AI call,
        # but we can verify the cached result is returned instantly)
        result2 = await self.detector.get_language_name(language_code)
        self.assertEqual(result2, expected_name)
        self.assertEqual(result1, result2)

if __name__ == '__main__':
    unittest.main()