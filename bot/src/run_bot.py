import unittest
from Gemini import generate_joke

class TestJokeGenerator(unittest.TestCase):
    def test_generate_joke(self):
        # Test input
        test_message = "tldr, it's basically switch 1 with some extra coloured plastic"

        # Generate joke
        result = generate_joke(test_message)

        expected_result = "tldr, its basically ur mom with some extra coloured plastic"
        self.assertEqual(result, expected_result)

    def test_generate_joke_2(self):
        # Test input
        test_message = "I remember it being too hard"

        # Generate joke
        result = generate_joke(test_message)

        print(result)

if __name__ == '__main__':
    unittest.main()