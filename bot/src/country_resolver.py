import traceback

class CountryResolver:
    def __init__(self, ai_client):
        self.ai_client = ai_client
        
        # custom names for better humour
        self.flag_to_country = {
            "🇺🇸": "America",
            "🇬🇧": "Britain",
            "🇷🇺": "Soviet Russia",
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

    async def get_country_from_flag(self, emoji: str) -> str | None:
        # First check if this is actually a flag emoji
        if not self._is_flag_emoji(emoji):
            return None

        if emoji in self.flag_to_country:
            return self.flag_to_country[emoji]

        # Try to resolve unknown flag using AI
        country = await self._resolve_flag_with_ai(emoji)
        if country:
            # Cache the result for future use
            self.flag_to_country[emoji] = country
            return country
        
        print(f'Unknown flag emoji: "{emoji}"')
        return None

    async def _resolve_flag_with_ai(self, emoji: str) -> str | None:
        prompt = "You are a flag emoji resolver. Given a flag emoji, respond only with the country name."
        samples = [["🇺🇸", "America"]]

        try:
            return await self.ai_client.generate_content(emoji, prompt, samples)
        except Exception as e:
            print(f"Error resolving flag with AI: {str(e)}")
            print("Stack trace:")
            print(traceback.format_exc())
            return None
