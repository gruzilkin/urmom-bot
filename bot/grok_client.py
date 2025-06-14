from openai import OpenAI
from openai.types.chat import ChatCompletion
from ai_client import AIClient
from typing import List, Tuple
from opentelemetry.trace import SpanKind
from open_telemetry import Telemetry

class GrokClient(AIClient):
    def __init__(self, api_key: str, model_name: str = "grok-2-latest", temperature: float = 0.7, telemetry: Telemetry = None):
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
        self.telemetry = telemetry
        
    def _track_completion_metrics(self, completion: ChatCompletion, method_name: str, **additional_attributes):
        """Track metrics from completion response with detailed attributes"""
        if self.telemetry:
            usage = completion.usage
            attributes = {
                "service": "GROK",
                "model": self.model_name,
                "method": method_name
            }
            
            # Add any additional attributes passed in
            attributes.update(additional_attributes)
            
            self.telemetry.track_token_usage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                attributes=attributes
            )

    async def generate_content(self, message: str, prompt: str = None, samples: List[Tuple[str, str]] = None) -> str:
        async with self.telemetry.async_create_span("generate_content", kind=SpanKind.CLIENT) as span:
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
            self._track_completion_metrics(completion, method_name="generate_content")

            return completion.choices[0].message.content

    async def is_joke(self, original_message: str, response_message: str) -> bool:
        async with self.telemetry.async_create_span("is_joke", kind=SpanKind.CLIENT) as span:
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
            self._track_completion_metrics(completion, method_name="is_joke", is_joke=result)
            print(f"[GROK] AI response: {response_text}")
            print(f"[GROK] Is joke: {result}")
            return result
