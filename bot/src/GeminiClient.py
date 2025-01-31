import google.generativeai as genai

class GeminiClient:
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
