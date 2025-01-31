import unittest
from unittest.mock import Mock
from country_resolver import CountryResolver

class TestCountryFlags(unittest.TestCase):
    def setUp(self):
        mock_gemini = Mock()
        self.resolver = CountryResolver(mock_gemini)

    def test_custom_names(self):
        test_cases = {
            "🇺🇸": "America",
            "🇬🇧": "Britain",
            "🇷🇺": "Soviet Russia",
        }

        for flag, expected in test_cases.items():
            with self.subTest(flag=flag):
                self.assertEqual(self.resolver.get_country_from_flag(flag), expected)

    def test_non_flag_emoji(self):
        self.assertIsNone(self.resolver.get_country_from_flag("😀"))
        
    def test_empty_string(self):
        self.assertIsNone(self.resolver.get_country_from_flag(""))
        
    def test_non_emoji_string(self):
        self.assertIsNone(self.resolver.get_country_from_flag("USA"))

    def test_is_flag_emoji(self):
        self.assertTrue(self.resolver._is_flag_emoji("🇺🇸"))
        self.assertTrue(self.resolver._is_flag_emoji("🇪🇬"))
        self.assertFalse(self.resolver._is_flag_emoji("😀"))
        self.assertFalse(self.resolver._is_flag_emoji(""))
        self.assertFalse(self.resolver._is_flag_emoji("🇺🇸🇬🇧"))

    def test_gemini_resolution(self):
        mock_response = Mock()
        mock_response.text = "Egypt"
        self.resolver.gemini_client.model.generate_content.return_value = mock_response
        
        result = self.resolver.get_country_from_flag("🇪🇬")
        self.assertEqual(result, "Egypt")
        self.assertEqual(self.resolver.flag_to_country["🇪🇬"], "Egypt")  # Check if cached

if __name__ == '__main__':
    unittest.main()
