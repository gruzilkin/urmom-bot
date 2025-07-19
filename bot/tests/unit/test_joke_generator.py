import unittest
from unittest.mock import AsyncMock, Mock
from joke_generator import JokeGenerator
from language_detector import LanguageDetector
from null_telemetry import NullTelemetry


class MockStore:
    """Mock store for testing"""
    def __init__(self):
        self.saved_jokes = []
        
    async def save(self, **kwargs):
        self.saved_jokes.append(kwargs)
        
    async def get_random_jokes(self, count):
        return [("test message", "test joke")]


class TestJokeGeneratorRefactored(unittest.IsolatedAsyncioTestCase):
    
    async def test_is_joke_returns_true_for_positive_response(self):
        """Test that is_joke returns True when AI responds 'yes'"""
        ai_client = Mock()
        ai_client.generate_content = AsyncMock(return_value="yes")
        store = MockStore()
        telemetry = NullTelemetry()
        
        # Use real LanguageDetector with mock AI client that raises error if accessed
        mock_language_ai = Mock()
        mock_language_ai.generate_content = AsyncMock(side_effect=Exception("AI should not be called in tests"))
        language_detector = LanguageDetector(ai_client=mock_language_ai, telemetry=telemetry)
        joke_generator = JokeGenerator(ai_client, store, telemetry, language_detector)
        
        result = await joke_generator.is_joke("original message", "funny response")
        
        self.assertTrue(result)

    async def test_is_joke_returns_false_for_negative_response(self):
        """Test that is_joke returns False when AI responds 'no'"""
        ai_client = Mock()
        ai_client.generate_content = AsyncMock(return_value="no")
        store = MockStore()
        telemetry = NullTelemetry()
        
        # Use real LanguageDetector with mock AI client that raises error if accessed
        mock_language_ai = Mock()
        mock_language_ai.generate_content = AsyncMock(side_effect=Exception("AI should not be called in tests"))
        language_detector = LanguageDetector(ai_client=mock_language_ai, telemetry=telemetry)
        joke_generator = JokeGenerator(ai_client, store, telemetry, language_detector)
        
        result = await joke_generator.is_joke("original message", "serious response")
        
        self.assertFalse(result)

    async def test_is_joke_handles_punctuation_in_response(self):
        """Test that is_joke handles punctuation correctly"""
        ai_client = Mock()
        ai_client.generate_content = AsyncMock(return_value="yes.")
        store = MockStore()
        telemetry = NullTelemetry()
        
        # Use real LanguageDetector with mock AI client that raises error if accessed
        mock_language_ai = Mock()
        mock_language_ai.generate_content = AsyncMock(side_effect=Exception("AI should not be called in tests"))
        language_detector = LanguageDetector(ai_client=mock_language_ai, telemetry=telemetry)
        joke_generator = JokeGenerator(ai_client, store, telemetry, language_detector)
        
        result = await joke_generator.is_joke("original message", "funny response")
        
        self.assertTrue(result)

    async def test_is_joke_caching(self):
        """Test that is_joke caches results properly"""
        ai_client = Mock()
        ai_client.generate_content = AsyncMock(return_value="yes")
        store = MockStore()
        telemetry = NullTelemetry()
        
        # Use real LanguageDetector with mock AI client that raises error if accessed
        mock_language_ai = Mock()
        mock_language_ai.generate_content = AsyncMock(side_effect=Exception("AI should not be called in tests"))
        language_detector = LanguageDetector(ai_client=mock_language_ai, telemetry=telemetry)
        joke_generator = JokeGenerator(ai_client, store, telemetry, language_detector)
        
        # First call should hit the AI
        result1 = await joke_generator.is_joke("original", "response", message_id=123)
        self.assertTrue(result1)
        self.assertEqual(ai_client.generate_content.call_count, 1)
        
        # Second call with same message_id should use cache
        result2 = await joke_generator.is_joke("original", "response", message_id=123)
        self.assertTrue(result2)
        self.assertEqual(ai_client.generate_content.call_count, 1)  # No additional call
        
        # Different message_id should hit AI again
        result3 = await joke_generator.is_joke("original", "response", message_id=456)
        self.assertTrue(result3)
        self.assertEqual(ai_client.generate_content.call_count, 2)

    async def test_save_joke(self):
        """Test that save_joke works correctly"""
        ai_client = Mock()
        ai_client.generate_content = AsyncMock(return_value="Test joke response")
        store = MockStore()
        telemetry = NullTelemetry()
        
        # Use real LanguageDetector with mock AI client that raises error if accessed
        mock_language_ai = Mock()
        mock_language_ai.generate_content = AsyncMock(side_effect=Exception("AI should not be called in tests"))
        language_detector = LanguageDetector(ai_client=mock_language_ai, telemetry=telemetry)
        joke_generator = JokeGenerator(ai_client, store, telemetry, language_detector)
        
        await joke_generator.save_joke(
            source_message_id=123,
            source_message_content="original message",
            joke_message_id=456,
            joke_message_content="funny joke",
            reaction_count=5
        )
        
        self.assertEqual(len(store.saved_jokes), 1)
        saved = store.saved_jokes[0]
        self.assertEqual(saved["source_message_id"], 123)
        self.assertEqual(saved["joke_message_id"], 456)
        self.assertEqual(saved["source_message_content"], "original message")
        self.assertEqual(saved["joke_message_content"], "funny joke")
        self.assertEqual(saved["reaction_count"], 5)
        self.assertIn("source_language", saved)
        self.assertIn("joke_language", saved)

    async def test_generate_joke_still_works(self):
        """Test that existing generate_joke functionality still works"""
        ai_client = Mock()
        ai_client.generate_content = AsyncMock(return_value="Test joke response")
        store = MockStore()
        telemetry = NullTelemetry()
        
        # Use real LanguageDetector with mock AI client that raises error if accessed
        mock_language_ai = Mock()
        mock_language_ai.generate_content = AsyncMock(side_effect=Exception("AI should not be called in tests"))
        language_detector = LanguageDetector(ai_client=mock_language_ai, telemetry=telemetry)
        joke_generator = JokeGenerator(ai_client, store, telemetry, language_detector)
        
        result = await joke_generator.generate_joke("test message", "en")
        
        self.assertEqual(result, "Test joke response")

    async def test_generate_country_joke_still_works(self):
        """Test that existing generate_country_joke functionality still works"""
        ai_client = Mock()
        ai_client.generate_content = AsyncMock(return_value="Test joke response")
        store = MockStore()
        telemetry = NullTelemetry()
        
        # Use real LanguageDetector with mock AI client that raises error if accessed
        mock_language_ai = Mock()
        mock_language_ai.generate_content = AsyncMock(side_effect=Exception("AI should not be called in tests"))
        language_detector = LanguageDetector(ai_client=mock_language_ai, telemetry=telemetry)
        joke_generator = JokeGenerator(ai_client, store, telemetry, language_detector)
        
        result = await joke_generator.generate_country_joke("test message", "USA")
        
        self.assertEqual(result, "Test joke response")


if __name__ == '__main__':
    unittest.main()
