import unittest
from utils import get_country_from_flag

class TestCountryFlags(unittest.TestCase):
    def test_common_flags(self):
        test_cases = {
            "🇺🇸": "America",
            "🇷🇺": "Soviet Russia",
            "🇨🇵": "France",
            "🇨🇦": "Canada",
            "🇬🇧": "Britain",
            "🇩🇪": "Germany",
            "🇮🇹": "Italy",
            "🇯🇵": "Japan",
            "🇨🇳": "China",
            "🇦🇺": "Australia"
        }
        
        for flag, expected in test_cases.items():
            with self.subTest(flag=flag):
                self.assertEqual(get_country_from_flag(flag), expected)
    
    def test_unknown_flag(self):
        # test Egypt flag
        self.assertIsNone(get_country_from_flag("🇪🇬"))
        
    def test_non_flag_emoji(self):
        self.assertIsNone(get_country_from_flag("😀"))
        
    def test_empty_string(self):
        self.assertIsNone(get_country_from_flag(""))
        
    def test_non_emoji_string(self):
        self.assertIsNone(get_country_from_flag("USA"))

if __name__ == '__main__':
    unittest.main()
