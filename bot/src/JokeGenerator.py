import re
import google.generativeai as genai

class JokeGenerator:
    def __init__(self, api_key: str, model_name: str, temperature: float = 1.2):
        if not api_key:
            raise ValueError("Gemini API key not provided!")
        if not model_name:
            raise ValueError("Gemini model name not provided!")

        genai.configure(api_key=api_key)

        # Create the model
        self.generation_config = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 512,
            "response_mime_type": "text/plain",
        }

        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.generation_config,
        )

        self.base_prompt = """You are a chatbot that receives a message and should generate a joke.
        One type of the jokes is ur mom joke, ur mom joke follows the pattern of replacing the subject or the object in a phrase with \"ur mom\" without adding much extra details.
        The second type is \"In Soviet Russia\" joke.
        You can combine the two patterns.
        Make it as lewd and preposterous as possible, carefully replace the subject and/or some objects in order to achieve the most outrageous result.
        Make sure that the joke is grammatically correct, check for subject-verb agreement, update pronouns after replacing subjects and objects.
        """

    def generate_joke(self, content: str, sample_jokes: list[tuple[str, str]]) -> str:
        prompt = [self.base_prompt]
        
        for (input, output) in sample_jokes:
            prompt.append(f"input: {input}")
            prompt.append(f"output: {output}")
        
        prompt.append("input: " + content)
        prompt.append("output: ")

        print(f"{prompt}")
        print(f"source: {content}")
        response = self.model.generate_content(prompt)
        print(f"response: {response.text}")

        return response.text

    def generate_country_joke(self, message: str, country: str) -> str:
        prompt = [  f"You are a chat bot and you need to turn a user message into a country joke.",
                            f"Your response should only contain the joke itself and it should start with 'In {country}'. Apply stereotypes and cliches about the country.",
                            f"Respond with a joke that plays with the following message: ", message ]

        print(f"{prompt}")
        response = self.model.generate_content(prompt)
        print(f"response: {response.text}")

        return response.text
