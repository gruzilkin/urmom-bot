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
        One of the jokes is ur mom joke, ur mom joke follows the pattern of replacing the subject or the object in a phrase with \"ur mom\" without adding much extra details.
        Make it as lewd and preposterous as possible, carefully replace the subject and/or some objects in order to achieve the most outrageous result.
        The second type is \"In Soviet Russia\" joke. \"In Soviet Russia\", also called the Russian reversal, is a joke template taking the general form \"In America you do X to/with Y; in Soviet Russia Y does X to/with you\".
        Typically the American clause describes a harmless ordinary activity and the inverted Soviet form something menacing or dysfunctional, satirizing life under communist rule, or in the \"old country\".
        Sometimes the first clause is omitted, and sometimes either clause or both are deliberately rendered with English grammatical errors stereotypical of Russians.
        You can combine the two patterns.
        Make sure that the joke is grammatically correct, check for subject-verb agreement, update pronouns after replacing subjects and objects.
        You should surround your response with *joke* tag before the joke and after so that I could automatically extract it with a script.
        """

    def generate_joke(self, content: str, sample_jokes: list[tuple[str, str]]) -> str:
        prompt = [self.base_prompt]
        
        for (input, output) in sample_jokes:
            prompt.append(f"input: {input}")
            prompt.append(f"output: *joke*{output}*joke*")
        
        prompt.append("input: " + content)
        prompt.append("output: ")

        print(f"{prompt}")
        print(f"source: {content}")
        response = self.model.generate_content(prompt)
        print(f"response: {response.text}")

        joke_match = re.search(r'\*joke\*(.*?)\*joke\*', response.text, re.DOTALL)
        if joke_match:
            joke = joke_match.group(1).strip()
            return joke
        else:
            return "I need some better urmom joke material to work with"
