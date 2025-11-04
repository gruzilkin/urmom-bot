"""
Gemini Client with Google Search grounding support.

This client can use GoogleSearchRetrieval to provide more accurate and up-to-date responses
by grounding responses with current web information, when supported by the model and API.
"""

import logging
from typing import TypeVar

from google import genai
from google.genai.types import Content, Part, GenerateContentConfig, GenerateContentResponse, Tool, GoogleSearch
from opentelemetry.trace import SpanKind
from pydantic import BaseModel

from ai_client import AIClient, BlockedException
from open_telemetry import Telemetry

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class GeminiClient(AIClient):
    def __init__(self, api_key: str, model_name: str, telemetry: Telemetry, temperature: float = 0.1, client=None):
        if not model_name:
            raise ValueError("Gemini model name not provided!")
        if not client and not api_key:
            raise ValueError("Gemini API key not provided!")

        self.client = client or genai.Client(api_key=api_key)
        self.temperature = temperature
        self.model_name = model_name
        self.telemetry = telemetry

    def _track_completion_metrics(self, response: GenerateContentResponse, method_name: str, **additional_attributes):
        """Track metrics from Gemini response with detailed attributes"""
        try:
            usage_metadata = response.usage_metadata

            prompt_tokens = usage_metadata.prompt_token_count
            completion_tokens = usage_metadata.candidates_token_count
            total_tokens = usage_metadata.total_token_count

            attributes = {
                "service": "GEMINI",
                "model": self.model_name,
            }

            # Add any additional attributes passed in
            attributes.update(additional_attributes)

            self.telemetry.track_token_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                attributes=attributes,
            )
            logger.info(
                f"Token usage tracked - Method: {method_name}, "
                f"Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}"
            )
        except Exception as e:
            logger.error(f"Error tracking token usage: {e}", exc_info=True)

    async def generate_content(
        self,
        message: str,
        prompt: str = None,
        samples: list[tuple[str, str]] = None,
        enable_grounding: bool = False,
        response_schema: type[T] | None = None,
        temperature: float | None = None,
        image_data: bytes | None = None,
        image_mime_type: str | None = None,
    ) -> str | T:
        if image_data:
            raise ValueError("GeminiClient does not support image data.")
        base_attrs = {"service": "GEMINI", "model": self.model_name}

        async with self.telemetry.async_create_span(
            "generate_content",
            kind=SpanKind.CLIENT,
            attributes=base_attrs,
        ):
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
            config = GenerateContentConfig(temperature=actual_temperature, system_instruction=prompt)

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

            timer = self.telemetry.metrics.timer()
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model_name, contents=contents, config=config
                )
            except Exception as e:
                attrs = {
                    **base_attrs,
                    "outcome": "error",
                    "error_type": type(e).__name__,
                }
                self.telemetry.metrics.llm_latency.record(timer(), attrs)
                self.telemetry.metrics.llm_requests.add(1, attrs)
                raise

            logger.info(response)

            block_reason = self._get_block_reason(response)
            if block_reason:
                attrs_blocked = {**base_attrs, "outcome": "blocked"}
                self.telemetry.metrics.llm_latency.record(timer(), attrs_blocked)
                self.telemetry.metrics.llm_requests.add(1, attrs_blocked)
                raise BlockedException(reason=str(block_reason))

            attrs_success = {**base_attrs, "outcome": "success"}
            self.telemetry.metrics.llm_latency.record(timer(), attrs_success)
            self.telemetry.metrics.llm_requests.add(1, attrs_success)

            self._track_completion_metrics(response, method_name="generate_content")

            # Return parsed object if schema was provided, otherwise return text
            if response_schema:
                if response.parsed is None:
                    # Count structured output failures
                    self.telemetry.metrics.structured_output_failures.add(
                        1, {"service": "GEMINI", "model": self.model_name}
                    )
                    raise ValueError(
                        f"Failed to parse response with schema {response_schema.__name__}: {response.text}"
                    )
                return response.parsed
            else:
                # Return only the first part's text (Gemini sometimes returns multiple parts)
                return response.candidates[0].content.parts[0].text

    def _get_block_reason(self, response: GenerateContentResponse):
        """Return the Gemini block reason if the response was rejected."""
        prompt_feedback = getattr(response, "prompt_feedback", None)
        if prompt_feedback is None:
            return None
        return getattr(prompt_feedback, "block_reason", None)
