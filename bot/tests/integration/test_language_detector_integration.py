import os
import unittest
from dataclasses import dataclass

from dotenv import load_dotenv

from gemma_client import GemmaClient
from language_detector import LanguageDetector
from null_telemetry import NullTelemetry
from ollama_client import OllamaClient

load_dotenv()


@dataclass(frozen=True)
class DetectorProfile:
    """Container describing a language detector AI client."""

    name: str
    client: object


class TestLanguageDetectorIntegration(unittest.IsolatedAsyncioTestCase):
    """
    Integration tests for LanguageDetector with real AI detection.

    Runs the same scenarios across Gemma and Kimi (when credentials are available)
    to validate language detection quality and caching behaviour.
    """

    async def asyncSetUp(self) -> None:
        self.telemetry = NullTelemetry()
        self.profiles: list[DetectorProfile] = []

        gemma_api_key = os.getenv("GEMMA_API_KEY")
        gemma_model = os.getenv("GEMMA_MODEL")

        if gemma_api_key and gemma_model:
            gemma_client = GemmaClient(
                api_key=gemma_api_key,
                model_name=gemma_model,
                telemetry=self.telemetry,
            )
            self.profiles.append(DetectorProfile(name="gemma", client=gemma_client))

        ollama_api_key = os.getenv("OLLAMA_API_KEY")
        if ollama_api_key:
            kimi_model = os.getenv("OLLAMA_KIMI_MODEL", "kimi-k2:1t-cloud")
            kimi_client = OllamaClient(
                api_key=ollama_api_key,
                model_name=kimi_model,
                telemetry=self.telemetry,
                temperature=0.0,
            )
            self.profiles.append(DetectorProfile(name="ollama_kimi", client=kimi_client))

        if not self.profiles:
            self.skipTest(
                "No language detector AI clients configured; ensure Gemma or Ollama credentials are set."
            )

    def _build_detector(self, profile: DetectorProfile) -> LanguageDetector:
        """Return a fresh LanguageDetector per profile to avoid cache sharing."""
        return LanguageDetector(ai_client=profile.client, telemetry=self.telemetry)

    async def test_llm_fallback_for_short_ambiguous_text(self):
        test_cases = {
            "ok": "en",
            "ciao": "it",
            "你好": "zh",
            "Привет": "ru",
        }

        for profile in self.profiles:
            detector = self._build_detector(profile)
            for text, expected_lang in test_cases.items():
                with self.subTest(profile=profile.name, text=text):
                    detected_lang = await detector.detect_language(text)
                    self.assertEqual(detected_lang, expected_lang)

    async def test_llm_fallback_for_translation_request(self):
        text = "répond en français: 'I love programming and I want to write a lot of code in Python.'"
        expected_lang = "fr"

        for profile in self.profiles:
            detector = self._build_detector(profile)
            with self.subTest(profile=profile.name):
                detected_lang = await detector.detect_language(text)
                self.assertEqual(detected_lang, expected_lang)

    async def test_llm_fallback_for_mixed_language_query(self):
        text = "what does 'wie geht es Ihnen?' mean in English?"
        expected_lang = "en"

        for profile in self.profiles:
            detector = self._build_detector(profile)
            with self.subTest(profile=profile.name):
                detected_lang = await detector.detect_language(text)
                self.assertEqual(detected_lang, expected_lang)

    async def test_english_with_embedded_non_latin_script(self):
        test_cases = [
            ("are 舞茸 and 黒アワビ茸 related?", "en"),
            ("what is the difference between 寿司 and 刺身?", "en"),
            ("how do you pronounce Привет?", "en"),
            ("is 北京 the same as Beijing?", "en"),
            (
                "explain what Bulgakov meant: "
                "'Никогда и ничего не просите! Никогда и ничего, "
                "и в особенности у тех, кто сильнее вас. "
                "Сами предложат и сами всё дадут!'",
                "en",
            ),
        ]

        for profile in self.profiles:
            detector = self._build_detector(profile)
            for text, expected_lang in test_cases:
                with self.subTest(profile=profile.name, text=text):
                    detected_lang = await detector.detect_language(text)
                    self.assertEqual(detected_lang, expected_lang)

    async def test_comprehensive_language_detection(self):
        test_cases = [
            ("Hello, how are you today? This is a longer English text.", "en"),
            ("Hola, ¿cómo estás? Este es un texto en español más largo.", "es"),
            ("Bonjour, comment allez-vous? Ceci est un texte français.", "fr"),
            ("Guten Tag, wie geht es Ihnen? Dies ist ein deutscher Text.", "de"),
            ("Привет, как дела? Это русский текст для проверки.", "ru"),
            ("こんにちは、元気ですか？これは日本語のテキストです。", "ja"),
        ]

        for profile in self.profiles:
            detector = self._build_detector(profile)
            for text, expected_lang in test_cases:
                with self.subTest(profile=profile.name, text=text[:50] + "..."):
                    result = await detector.detect_language(text)
                    self.assertEqual(result, expected_lang)

    async def test_discord_message_scenario(self):
        text = "did Putin bomb Kiev again?"
        expected_lang = "en"

        for profile in self.profiles:
            detector = self._build_detector(profile)
            with self.subTest(profile=profile.name):
                result = await detector.detect_language(text)
                self.assertEqual(result, expected_lang)

    async def test_very_short_discord_messages(self):
        test_cases = [
            ("lol", "en"),
            ("wtf", "en"),
            ("omg", "en"),
            ("thx", "en"),
            ("ok", "en"),
        ]

        for profile in self.profiles:
            detector = self._build_detector(profile)
            for text, expected_lang in test_cases:
                with self.subTest(profile=profile.name, text=text):
                    result = await detector.detect_language(text)
                    self.assertEqual(result, expected_lang)

    async def test_deterministic_behavior_with_real_ai(self):
        text = "Hello world, this is a test of deterministic language detection."

        for profile in self.profiles:
            detector = self._build_detector(profile)
            with self.subTest(profile=profile.name):
                result1 = await detector.detect_language(text)
                result2 = await detector.detect_language(text)
                result3 = await detector.detect_language(text)
                self.assertEqual(result1, result2)
                self.assertEqual(result2, result3)
                self.assertEqual(result1, "en")

    async def test_get_language_name_from_llm(self):
        language_code = "sv"
        expected_name = "Swedish"

        for profile in self.profiles:
            detector = self._build_detector(profile)
            with self.subTest(profile=profile.name):
                if language_code in detector._language_names:
                    del detector._language_names[language_code]

                language_name = await detector.get_language_name(language_code)
                self.assertEqual(language_name, expected_name)
                self.assertIn(language_code, detector._language_names)
                self.assertEqual(detector._language_names[language_code], expected_name)

    async def test_caching_behavior_with_real_ai(self):
        language_code = "it"
        expected_name = "Italian"

        for profile in self.profiles:
            detector = self._build_detector(profile)
            with self.subTest(profile=profile.name):
                detector._language_names.pop(language_code, None)

                result1 = await detector.get_language_name(language_code)
                self.assertEqual(result1, expected_name)
                self.assertIn(language_code, detector._language_names)

                result2 = await detector.get_language_name(language_code)
                self.assertEqual(result2, expected_name)
                self.assertEqual(result1, result2)


if __name__ == "__main__":
    unittest.main()
