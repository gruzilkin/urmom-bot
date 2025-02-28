from openai import OpenAI
from ai_client import AIClient
from typing import List, Tuple

class GrokClient(AIClient):
    def __init__(self, api_key: str, model_name: str = "grok-2-latest", temperature: float = 0.7):
        if not api_key:
            raise ValueError("Grok API key not provided!")
        if not model_name:
            raise ValueError("Grok model name not provided!")

        self.model = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
        self.model_name = model_name
        self.temperature = temperature

    async def generate_content(self, message: str, prompt: str = None, samples: List[Tuple[str, str]] = None) -> str:
        messages = []
        if prompt:
            messages.append({"role": "system", "content": prompt})
        
        if samples:
            for user_msg, assistant_msg in samples:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})
            
        messages.append({"role": "user", "content": message})

        print(messages)

        completion = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature
        )

        print(completion)

        return completion.choices[0].message.content

    async def is_joke(self, original_message: str, response_message: str) -> bool:
        prompt = f"""Tell me if the response is a joke, a wordplay or a sarcastic remark to the original message, reply in English with only yes or no:
original message: {original_message}
response: {response_message}
No? Think again carefully. The response might be a joke, wordplay, or sarcastic remark.
Is it actually a joke? Reply only yes or no."""

        print(f"[GROK] Checking if message is a joke:")
        print(f"[GROK] Original: {original_message}")
        print(f"[GROK] Response: {response_message}")

        completion = self.model.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        print(f"[GROK] Raw completion object: {completion}")
        response_text = completion.choices[0].message.content.strip().lower()
        # Remove any punctuation and check if the response is 'yes'
        result = response_text.rstrip('.,!?') == "yes"
        print(f"[GROK] AI response: {response_text}")
        print(f"[GROK] Is joke: {result}")
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
        
        system_message = {
            "role": "system", 
            "content": f"""You are {person}. Generate a response as if you were {person}, 
            using their communication style, beliefs, values, and knowledge.
            Make the response thoughtful, authentic to {person}'s character, and relevant to the conversation.
            Stay in character completely and respond directly as {person} would.
            Keep your response length similar to the conversation context."""
        }
        
        messages = [system_message]
        
        for username, content in conversation:
            messages.append({
                "role": "user",
                "content": f"{username}: {content}"
            })
        
        print(f"[GROK] Generating response as {person}")
        print(f"[GROK] Messages: {messages}")
        
        completion = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature
        )
        
        print(f"[GROK] Raw completion object: {completion}")
        return completion.choices[0].message.content
