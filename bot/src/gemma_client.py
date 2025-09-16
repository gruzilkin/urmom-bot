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
from google.genai import types
from ai_client import AIClient
from open_telemetry import Telemetry
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

    async def generate_content(self, message: str, prompt: str = None, samples: List[Tuple[str, str]] = None, enable_grounding: bool = False, response_schema: Type[T] | None = None, temperature: float | None = None, image_data: bytes | None = None, image_mime_type: str | None = None) -> str | T:
        """
        Generate content using Gemma model.
        
        Args:
            message: The main user message/query
            prompt: System prompt to guide the response
            samples: Example conversations (not supported by Gemma)
            enable_grounding: Enable web grounding (not supported by Gemma)
            response_schema: Pydantic model for structured JSON output
            temperature: Override default temperature for this request
            image_data: Raw image bytes for multimodal input
            image_mime_type: MIME type of the image (e.g., 'image/jpeg', 'image/png')
            
        Returns:
            Generated text response or structured object if response_schema provided
        """
        async with self.telemetry.async_create_span("generate_content", kind=SpanKind.CLIENT):
            # Log unsupported features
            if samples:
                logger.warning("Samples/chat mode not supported by simplified Gemma client")
            if enable_grounding:
                logger.warning("Grounding not supported by Gemma models")

            # Build content parts for Gemma (text and optional image)
            content_parts = []
            
            # Add text content
            if response_schema:
                # Structured format for JSON responses
                text_content = f"""<system>{prompt or 'You are a helpful assistant.'}</system>
<user_message>{message}</user_message>
<response_format>Respond with a valid JSON object matching the provided schema. Do not include explanations or multiple JSON blocks - return only the requested parameter values as a single JSON object.</response_format>

Schema: {response_schema.model_json_schema()}"""
            else:
                # Simple format for text responses
                if prompt:
                    text_content = f"<system>{prompt or 'You are a helpful assistant.'}</system>\n<user_message>{message}</user_message>"
                else:
                    text_content = message
            
            content_parts.append(types.Part.from_text(text=text_content))
            
            # Add image if provided
            if image_data and image_mime_type:
                content_parts.append(types.Part.from_bytes(data=image_data, mime_type=image_mime_type))
                logger.info(f"Gemma multimodal request with image ({image_mime_type}): {text_content}")
            else:
                logger.info(f"Gemma text request: {text_content}")

            # Use provided temperature or fallback to instance temperature
            actual_temperature = temperature if temperature is not None else self.temperature
            config = GenerateContentConfig(temperature=actual_temperature)

            # Generate content with multimodal support
            timer = self.telemetry.metrics.timer()
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=content_parts,
                    config=config
                )
                attrs = {"service": "GEMMA", "model": self.model_name, "outcome": "success"}
                self.telemetry.metrics.llm_latency.record(timer(), attrs)
                self.telemetry.metrics.llm_requests.add(1, attrs)
            except Exception as e:
                attrs = {"service": "GEMMA", "model": self.model_name, "outcome": "error", "error_type": type(e).__name__}
                self.telemetry.metrics.llm_latency.record(timer(), attrs)
                self.telemetry.metrics.llm_requests.add(1, attrs)
                raise

            logger.info(response)
            # Track metrics with multimodal indicator
            additional_attributes = {}
            if image_data:
                additional_attributes["multimodal"] = True
                additional_attributes["image_mime_type"] = image_mime_type
            self._track_completion_metrics(response, method_name="generate_content", **additional_attributes)
            
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
                    self.telemetry.metrics.structured_output_failures.add(1, {"service": "GEMMA", "model": self.model_name})
                    raise ValueError(f"Failed to parse response with schema {response_schema.__name__}: {response_text}")
            
            return response_text
