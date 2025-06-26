"""
Gemma Client using Google GenAI API with manual JSON parsing.

This client uses the same Google GenAI API as Gemini but handles structured output
manually like Claude client, since Gemma models don't support native structured output.
"""

import json
import logging
from typing import List, Tuple, Type, TypeVar

from google import genai
from google.genai.types import GenerateContentConfig, GenerateContentResponse
from bot.ai_client import AIClient
from bot.open_telemetry import Telemetry
from opentelemetry.trace import SpanKind
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class GemmaClient(AIClient):
    def __init__(self, api_key: str, model_name: str, telemetry: Telemetry, temperature: float = 0.1):
        if not api_key:
            raise ValueError("Gemma API key not provided!")
        if not model_name:
            raise ValueError("Gemma model name not provided!")

        self.client = genai.Client(api_key=api_key)
        self.temperature = temperature
        self.model_name = model_name
        self.telemetry = telemetry
        
    def _track_completion_metrics(self, response: GenerateContentResponse, method_name: str, **additional_attributes):
        """Track metrics from Gemma response with detailed attributes"""
        try:
            usage_metadata = response.usage_metadata
            
            prompt_tokens = usage_metadata.prompt_token_count or 0
            completion_tokens = usage_metadata.candidates_token_count or 0
            total_tokens = usage_metadata.total_token_count or 0
                
            attributes = {
                "service": "GEMMA",
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
        except Exception as e:
            logger.error(f"Error tracking token usage: {e}", exc_info=True)

    async def generate_content(self, message: str, prompt: str = None, samples: List[Tuple[str, str]] = None, enable_grounding: bool = False, response_schema: Type[T] | None = None, temperature: float | None = None) -> str | T:
        async with self.telemetry.async_create_span("generate_content", kind=SpanKind.CLIENT) as span:
            # Log unsupported features
            if samples:
                logger.warning("Samples/chat mode not supported by simplified Gemma client")
            if enable_grounding:
                logger.warning("Grounding not supported by Gemma models")

            # Build final message
            final_message = message
            
            # Add system prompt by prepending (Gemma doesn't support system instructions)
            if prompt:
                final_message = f"System: {prompt}\n\nUser: {final_message}"
            
            # Add schema instruction for structured output
            if response_schema:
                schema_instruction = f"\n\nPlease respond with a valid JSON object that matches this schema: {response_schema.model_json_schema()}"
                final_message += schema_instruction

            logger.info(f"Gemma request: {final_message}")

            # Use provided temperature or fallback to instance temperature
            actual_temperature = temperature if temperature is not None else self.temperature
            config = GenerateContentConfig(temperature=actual_temperature)

            # Simple text-to-text generation (no chat mode)
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=final_message,
                config=config
            )

            logger.info(response)
            self._track_completion_metrics(response, method_name="generate_content")
            
            response_text = response.text
            
            # Parse structured response if schema was provided
            if response_schema:
                try:
                    # Extract JSON from markdown block if present, otherwise use raw text
                    json_str = response_text.strip()
                    if "```json" in json_str:
                        start = json_str.find("```json") + 7
                        end = json_str.find("```", start)
                        json_str = json_str[start:end].strip()
                    
                    response_data = json.loads(json_str)
                    parsed_result = response_schema.model_validate(response_data)
                    return parsed_result
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Failed to parse structured response: {e}", exc_info=True)
                    raise ValueError(f"Failed to parse response with schema {response_schema.__name__}: {response_text}")
            
            return response_text