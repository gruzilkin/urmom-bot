from google import genai
from google.genai.types import Content, Part, GenerateContentConfig
from ai_client import AIClient
from typing import List, Tuple

class GeminiClient(AIClient):
    def __init__(self, api_key: str, model_name: str, temperature: float = 1.2):
        if not api_key:
            raise ValueError("Gemini API key not provided!")
        if not model_name:
            raise ValueError("Gemini model name not provided!")

        self.client = genai.Client(api_key=api_key)
        self.temperature = temperature
        self.model_name = model_name

    async def generate_content(self, message: str, prompt: str = None, samples: List[Tuple[str, str]] = None) -> str:
        samples = samples or []
        contents = []

        for msg, joke in samples:
            contents.append(Content(parts=[Part(text=msg)], role="user"))
            contents.append(Content(parts=[Part(text=joke)], role="model"))

        contents.append(Content(parts=[Part(text=message)], role="user"))

        print(f"[GEMINI] system_instruction: {prompt}")
        print(f"[GEMINI] Request contents: {contents}")

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=GenerateContentConfig(
                temperature=self.temperature,
                system_instruction = prompt,
            )
        )

        print(response)
        print(f"[GEMINI] Response: {response}")
        return response.text

    async def is_joke(self, original_message: str, response_message: str) -> bool:
        prompt = f"""Tell me if the response is a joke, a wordplay or a sarcastic remark to the original message, reply in English with only yes or no:
original message: {original_message}
response: {response_message}
No? Think again carefully. The response might be a joke, wordplay, or sarcastic remark.
Is it actually a joke? Reply only yes or no."""
        
        print(f"[GEMINI] Checking if message is a joke:")
        print(f"[GEMINI] Original: {original_message}")
        print(f"[GEMINI] Response: {response_message}")

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config=GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1,
                stop_sequences=["\n", "."]
            )
        )
        
        print(f"[GEMINI] Raw response object: {response}")
        result = response.text.strip().lower() == "yes"
        print(f"[GEMINI] AI response: {response.text}")
        print(f"[GEMINI] Is joke: {result}")
        return result

    async def generate_famous_person_response(self, conversation: List[Tuple[str, str]], person: str) -> str:
        """
        Generate a response in the style of a famous person based on the conversation context.
        
        Args:
            conversation (List[Tuple[str, str]]): List of (username, message) tuples
            person (str): The name of the famous person
            
        Returns:
            str: A response in the style of the famous person
        """

        system_instruction = f"""You are {person}. Generate a response as if you were {person}, 
            using their communication style, beliefs, values, and knowledge.
            Make the response thoughtful, authentic to {person}'s character, and relevant to the conversation.
            Stay in character completely and respond directly as {person} would.
            Keep your response length to about a couple of paragraphs."""
        
        contents = []
        for username, content in conversation:
            contents.append(Content(parts=[Part(text=f"{username}: {content}")], role="user"))
        
        print(f"[GEMINI] Generating response as {person}")
        print(f"[GEMINI] Conversation: {conversation}")
        
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=GenerateContentConfig(
                temperature=self.temperature,
                system_instruction=system_instruction,
            )
        )
        
        print(f"[GEMINI] Famous person response: {response.text}")
        return response.text
