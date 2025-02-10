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

    async def is_joke(self, original_message: str, response_message: str) -> bool:
        prompt = f"""Tell me if the response is a joke, a wordplay or a sarcastic remark to the original message, reply in English with only yes or no:
original message: {original_message}
response: {response_message}
No? Think again carefully. The response might be a joke, wordplay, or sarcastic remark.
Is it actually a joke? Reply only yes or no."""
        
        print(f"[GEMINI] Checking if message is a joke:")
        print(f"[GEMINI] Original: {original_message}")
        print(f"[GEMINI] Response: {response_message}")

        response = await self.model.generate_content_async(
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 1,
                "stop_sequences": ["\n", "."]
            }
        )
        print(f"[GEMINI] Raw response object: {response}")
        result = response.text.strip().lower() == "yes"
        print(f"[GEMINI] AI response: {response.text}")
        print(f"[GEMINI] Is joke: {result}")
        return result
