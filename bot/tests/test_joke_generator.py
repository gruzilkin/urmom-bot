import unittest
import os
from dotenv import load_dotenv
from joke_generator import JokeGenerator

load_dotenv()  # Load environment variables from .env file

class TestJokeGenerator(unittest.TestCase):
    def setUp(self):
        self.joke_generator = JokeGenerator(
            api_key=os.getenv('GEMINI_API_KEY'),
            model_name=os.getenv('GEMINI_MODEL'),
            temperature=float(os.getenv('GEMINI_TEMPERATURE', '1.2'))
        )

    def test_generate_joke(self):
        # Test input
        test_message = "tldr, it's basically switch 1 with some extra coloured plastic"
        sample_jokes = [
            ("The switch is hard to use", "ur mom is hard to use"),
            ("I need more space in my room", "I need more space in ur mom")
        ]

        # Generate joke
        result = self.joke_generator.generate_joke(test_message, sample_jokes)

        expected_result = "tldr, its basically ur mom with some extra coloured plastic"
        self.assertEqual(result, expected_result)

    def test_generate_joke_2(self):
        # Test input
        test_message = "I remember it being too hard"
        sample_jokes = [
            ("The switch is hard to use", "ur mom is hard to use"),
            ("I need more space in my room", "I need more space in ur mom")
        ]

        # Generate joke
        result = self.joke_generator.generate_joke(test_message, sample_jokes)

        print(result)

if __name__ == '__main__':
    unittest.main()