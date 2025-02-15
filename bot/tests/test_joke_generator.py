import unittest
import os
from unittest.mock import Mock
from dotenv import load_dotenv
from joke_generator import JokeGenerator
from gemini_client import GeminiClient
from store import Store

load_dotenv()

class TestJokeGenerator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        ai_client = GeminiClient(
            api_key=os.getenv('GEMINI_API_KEY'),
            model_name=os.getenv('GEMINI_MODEL'),
            temperature=float(os.getenv('GEMINI_TEMPERATURE', '1.2'))
        )
        self.store = Mock(spec=Store)
        self.store.get_random_jokes.return_value = [
            ("The switch is hard to use", "ur mom is hard to use"),
            ("I need more space in my room", "I need more space in ur mom"),
            ("This game is too expensive", "ur mom is too expensive"),
            ("The graphics are amazing", "ur mom's graphics are amazing"),
            ("I can't handle this level", "I can't handle ur mom")
        ]
        
        self.joke_generator = JokeGenerator(
            ai_client=ai_client,
            store=self.store
        )

    async def test_generate_joke(self):
        # Test input
        test_message = "tldr, it's basically switch 1 with some extra coloured plastic"
        
        # Generate joke
        result = await self.joke_generator.generate_joke(test_message, "en")
        print(f"Generated joke: {result}")

    async def test_generate_country_joke(self):
        # Test input
        test_message = "I like to eat"
        country = "France"
        
        # Generate joke
        result = await self.joke_generator.generate_country_joke(test_message, country)
        print(f"Generated country joke: {result}")

if __name__ == '__main__':
    unittest.main()