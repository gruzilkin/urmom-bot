import unittest
from unittest.mock import Mock, AsyncMock
from country_resolver import CountryResolver
from null_telemetry import NullTelemetry

class TestCountryFlags(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ai_client = Mock()
        self.ai_client.generate_content = AsyncMock()
        self.resolver = CountryResolver(self.ai_client, NullTelemetry())

    async def test_custom_names(self):
        test_cases = {
            "🇺🇸": "America",
            "🇬🇧": "Britain",
            "🇷🇺": "Soviet Russia",
        }

        for flag, expected in test_cases.items():
            with self.subTest(flag=flag):
                self.assertEqual(await self.resolver.get_country_from_flag(flag), expected)

    async def test_non_flag_emoji(self):
        self.assertIsNone(await self.resolver.get_country_from_flag("😀"))
        
    async def test_empty_string(self):
        self.assertIsNone(await self.resolver.get_country_from_flag(""))
        
    async def test_non_emoji_string(self):
        self.assertIsNone(await self.resolver.get_country_from_flag("USA"))

    def test_is_flag_emoji(self):
        self.assertTrue(self.resolver._is_flag_emoji("🇺🇸"))
        self.assertTrue(self.resolver._is_flag_emoji("🇪🇬"))
        self.assertFalse(self.resolver._is_flag_emoji("😀"))
        self.assertFalse(self.resolver._is_flag_emoji(""))
        self.assertFalse(self.resolver._is_flag_emoji("🇺🇸🇬🇧"))

    async def test_gemini_resolution(self):
        self.ai_client.generate_content.return_value = "Egypt"
        
        result = await self.resolver.get_country_from_flag("🇪🇬")
        self.assertEqual(result, "Egypt")
        self.assertEqual(self.resolver.flag_to_country["🇪🇬"], "Egypt")  # Check if cached
        self.ai_client.generate_content.assert_called_once()

if __name__ == '__main__':
    unittest.main()
