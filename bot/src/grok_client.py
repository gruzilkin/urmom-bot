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
