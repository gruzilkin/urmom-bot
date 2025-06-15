"""
Gemini Client with Google Search grounding support.

This client can use GoogleSearchRetrieval to provide more accurate and up-to-date responses
by grounding responses with current web information, when supported by the model and API.
"""

from google import genai
from google.genai.types import Content, Part, GenerateContentConfig, GenerateContentResponse, Tool, GoogleSearch
from ai_client import AIClient
from typing import List, Tuple
from opentelemetry.trace import SpanKind
from open_telemetry import Telemetry

class GeminiClient(AIClient):
    def __init__(self, api_key: str, model_name: str, temperature: float = 1.2, telemetry: Telemetry = None):
        if not api_key:
            raise ValueError("Gemini API key not provided!")
        if not model_name:
            raise ValueError("Gemini model name not provided!")

        self.client = genai.Client(api_key=api_key)
        self.temperature = temperature
        self.model_name = model_name
        self.telemetry = telemetry
        
    def _track_completion_metrics(self, response: GenerateContentResponse, method_name: str, **additional_attributes):
        """Track metrics from Gemini response with detailed attributes"""
        if self.telemetry:
            try:
                usage_metadata = response.usage_metadata
                
                prompt_tokens = usage_metadata.prompt_token_count
                completion_tokens = usage_metadata.candidates_token_count
                total_tokens = usage_metadata.total_token_count
                    
                attributes = {
                    "service": "GEMINI",
                    "model": self.model_name,
                    "method": method_name
                }
                
                # Add any additional attributes passed in
                attributes.update(additional_attributes)
                
                self.telemetry.track_token_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    attributes=attributes
                )
                print(f"[GEMINI] Token usage tracked - Method: {method_name}, Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
            except Exception as e:
                print(f"[GEMINI] Error tracking token usage: {e}")

    async def generate_content(self, message: str, prompt: str = None, samples: List[Tuple[str, str]] = None, enable_grounding: bool = False) -> str:
        async with self.telemetry.async_create_span("generate_content", kind=SpanKind.CLIENT) as span:
            samples = samples or []
            contents = []

            for msg, joke in samples:
                contents.append(Content(parts=[Part(text=msg)], role="user"))
                contents.append(Content(parts=[Part(text=joke)], role="model"))

            contents.append(Content(parts=[Part(text=message)], role="user"))

            print(f"[GEMINI] system_instruction: {prompt}")
            print(f"[GEMINI] Request contents: {contents}")

            # Configure grounding based on enable_grounding flag
            config = GenerateContentConfig(
                temperature=self.temperature,
                system_instruction=prompt
            )
            
            if enable_grounding:
                tools = [Tool(google_search=GoogleSearch())]
                config.tools = tools
                print(f"[GEMINI] Grounding enabled with Google Search")
            else:
                print(f"[GEMINI] Grounding disabled")

            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )

            print(response)
            print(f"[GEMINI] Response: {response}")
            self._track_completion_metrics(response, method_name="generate_content")
            return response.text
