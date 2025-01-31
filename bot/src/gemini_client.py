import google.generativeai as genai
from ai_client import AIClient
from typing import List, Tuple

class GeminiClient(AIClient):
    def __init__(self, api_key: str, model_name: str, temperature: float = 1.2):
        if not api_key:
            raise ValueError("Gemini API key not provided!")
        if not model_name:
            raise ValueError("Gemini model name not provided!")

        genai.configure(api_key=api_key)

        generation_config = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 512,
            "response_mime_type": "text/plain",
        }

        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
        )

    async def generate_content(self, message: str, prompt: str = None, samples: List[Tuple[str, str]] = None) -> str:
        history = []
        
        # Add prompt as first message in history if exists
        if prompt:
            history.append({"role": "user", "parts": [prompt]})
        
        # Add samples to history
        if samples:
            for user_msg, assistant_msg in samples:
                history.append({"role": "user", "parts": [user_msg]})
                history.append({"role": "model", "parts": [assistant_msg]})

        print(history)

        # Start chat with history and send current message
        chat = self.model.start_chat(history=history)
        response = await chat.send_message_async(message)

        print(response)

        return response.text
