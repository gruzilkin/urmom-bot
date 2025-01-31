class CountryResolver:
    def __init__(self, gemini_client):
        self.gemini_client = gemini_client
        self.flag_to_country = {
            "🇺🇸": "America",
            "🇦🇺": "Australia", 
            "🇬🇧": "Britain",
            "🇨🇦": "Canada",
            "🇨🇳": "China",
            "🇨🇺": "Cuba",
            "🇨🇵": "France",
            "🇯🇵": "Japan",
            "🇵🇱": "Poland",
            "🇷🇺": "Soviet Russia",
            "🇺🇦": "Ukraine",
            "🇰🇵": "North Korea",
        }

    def _is_flag_emoji(self, emoji: str) -> bool:
        # Flag emojis consist of exactly 2 regional indicator symbols
        # Regional indicator symbols are in range U+1F1E6 to U+1F1FF
        if len(emoji) != 2:
            return False
            
        for char in emoji:
            # Check if character is in the regional indicator range
            code_point = ord(char)
            if not (0x1F1E6 <= code_point <= 0x1F1FF):
                return False
        return True

    def get_country_from_flag(self, emoji: str) -> str | None:
        # First check if this is actually a flag emoji
        if not self._is_flag_emoji(emoji):
            return None

        if emoji in self.flag_to_country:
            return self.flag_to_country[emoji]

        # Try to resolve unknown flag using Gemini
        country = self._resolve_flag_with_gemini(emoji)
        if country:
            # Cache the result for future use
            self.flag_to_country[emoji] = country
            return country
        
        print(f'Unknown flag emoji: "{emoji}"')
        return None

    def _resolve_flag_with_gemini(self, emoji: str) -> str | None:
        prompt = [
            "You are a flag emoji resolver. Given a flag emoji, respond only with the country name in a single word.",
            "For example: 🇺🇸 -> America",
            f"Emoji to resolve: {emoji}"
        ]

        try:
            response = self.gemini_client.model.generate_content(prompt)
            country = response.text.strip()
            return country if country else None
        except Exception as e:
            print(f"Error resolving flag with Gemini: {e}")
            return None
