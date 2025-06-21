"""
Gemini Client with Google Search grounding support.

This client can use GoogleSearchRetrieval to provide more accurate and up-to-date responses
by grounding responses with current web information, when supported by the model and API.
"""

from google import genai
from google.genai.types import Content, Part, GenerateContentConfig, GenerateContentResponse, Tool, GoogleSearch
from ai_client import AIClient
from typing import List, Tuple, Type, TypeVar
from opentelemetry.trace import SpanKind
from open_telemetry import Telemetry
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

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
                logger.info(f"Token usage tracked - Method: {method_name}, Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
            except Exception as e:
                logger.error(f"Error tracking token usage: {e}", exc_info=True)

    async def generate_content(self, message: str, prompt: str = None, samples: List[Tuple[str, str]] = None, enable_grounding: bool = False, response_schema: Type[T] | None = None, temperature: float | None = None) -> str | T:
        async with self.telemetry.async_create_span("generate_content", kind=SpanKind.CLIENT) as span:
            samples = samples or []
            contents = []

            for msg, joke in samples:
                contents.append(Content(parts=[Part(text=msg)], role="user"))
                contents.append(Content(parts=[Part(text=joke)], role="model"))

            contents.append(Content(parts=[Part(text=message)], role="user"))

            logger.info(f"system_instruction: {prompt}")
            logger.info(f"Request contents: {contents}")

            # Configure grounding based on enable_grounding flag
            # Use provided temperature or fallback to instance temperature
            actual_temperature = temperature if temperature is not None else self.temperature
            config = GenerateContentConfig(
                temperature=actual_temperature,
                system_instruction=prompt
            )
            
            # Configure structured output if response_schema is provided
            if response_schema:
                config.response_mime_type = "application/json"
                config.response_schema = response_schema
                logger.info(f"Structured output enabled with schema: {response_schema.__name__}")
            
            if enable_grounding:
                tools = [Tool(google_search=GoogleSearch())]
                config.tools = tools
                logger.info("Grounding enabled with Google Search")
            else:
                logger.info("Grounding disabled")

            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )

            logger.info(response)
            logger.info(f"Response: {response}")
            self._track_completion_metrics(response, method_name="generate_content")
            
            # Return parsed object if schema was provided, otherwise return text
            if response_schema:
                if response.parsed is None:
                    raise ValueError(f"Failed to parse response with schema {response_schema.__name__}: {response.text}")
                return response.parsed
            else:
                return response.text
