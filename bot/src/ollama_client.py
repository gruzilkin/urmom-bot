"""
Ollama Client using native Ollama Python library.

Supports cloud models via api.ollama.com with native structured output,
web search/grounding, and multimodal (vision) capabilities.
"""

import base64
import json
import logging
from typing import Any, List, Tuple, Type, TypeVar

import httpx
from ollama import AsyncClient
from pydantic import BaseModel, ValidationError

from ai_client import AIClient
from open_telemetry import Telemetry
from opentelemetry.trace import SpanKind

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OllamaClient(AIClient):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        telemetry: Telemetry,
        base_url: str = "https://ollama.com",
        temperature: float = 0.1,
        timeout: float | httpx.Timeout | None = 20.0,
    ):
        if not api_key:
            raise ValueError("Ollama API key not provided!")
        if not model_name:
            raise ValueError("Ollama model name not provided!")

        # AsyncClient requires Authorization header for cloud API
        # The client's web_search and web_fetch methods will use this auth
        self.client = AsyncClient(
            host=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        self.api_key = api_key
        self.model_name = model_name
        self.telemetry = telemetry
        self.temperature = temperature
        self.service = "OLLAMA"
        self.timeout = timeout

    def _track_completion_metrics(
        self, response: dict, method_name: str, **additional_attributes
    ):
        """Track metrics from Ollama response with detailed attributes"""
        try:
            # Ollama response structure may vary, handle gracefully
            usage = response.get("usage", {})

            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            attributes = {
                "service": self.service,
                "model": self.model_name,
            }

            attributes.update(additional_attributes)

            self.telemetry.track_token_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                attributes=attributes,
            )
            logger.info(
                f"Token usage tracked - Method: {method_name}, "
                f"Prompt: {prompt_tokens}, Completion: {completion_tokens}, "
                f"Total: {total_tokens}"
            )
        except Exception as e:
            logger.error(f"Error tracking token usage: {e}", exc_info=True)

    async def generate_content(
        self,
        message: str,
        prompt: str | None = None,
        samples: List[Tuple[str, str]] | None = None,
        enable_grounding: bool = False,
        response_schema: Type[T] | None = None,
        temperature: float | None = None,
        image_data: bytes | None = None,
        image_mime_type: str | None = None,
    ) -> str | T:
        """
        Generate content using Ollama model.

        Args:
            message: The main user message/query
            prompt: System prompt to guide the response
            samples: Example conversations for few-shot learning
            enable_grounding: Enable web search grounding (not supported)
            response_schema: Pydantic model for structured JSON output
            temperature: Override default temperature for this request
            image_data: Raw image bytes for multimodal input
            image_mime_type: MIME type of the image (e.g., 'image/jpeg', 'image/png')

        Returns:
            Generated text response or structured object if response_schema provided
        """
        async with self.telemetry.async_create_span(
            "generate_content",
            kind=SpanKind.CLIENT,
            attributes={"service": self.service, "model": self.model_name},
        ):
            # Build messages list
            messages = []

            # Add system prompt if provided
            if response_schema:
                schema_prompt = prompt or "You are a helpful assistant."
                schema_prompt = (
                    f"{schema_prompt}\n\n"
                    "You must respond with a single valid JSON object that matches the "
                    "following schema exactly. Do not include extra fields, omit required "
                    "fields, add prose, or wrap the JSON in markdown code fences. Produce the "
                    "JSON directly—do not invoke any helper commands or special formatting utilities.\n"
                    f"Schema:\n{json.dumps(response_schema.model_json_schema(), indent=2)}"
                )
                messages.append({"role": "system", "content": schema_prompt})
            elif prompt:
                messages.append({"role": "system", "content": prompt})

            # Add few-shot samples if provided
            if samples:
                for user_msg, assistant_msg in samples:
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": assistant_msg})

            # Build the main user message
            user_message = {"role": "user", "content": message}

            # Add images if provided (multimodal support)
            # Images must be base64-encoded and included in the message
            if image_data:
                # Ollama expects images as base64-encoded strings
                encoded_image = base64.b64encode(image_data).decode("utf-8")
                user_message["images"] = [encoded_image]
                logger.info(
                    f"Ollama multimodal request with image ({image_mime_type}): {message}"
                )
            else:
                logger.info(f"Ollama text request: {message}")

            messages.append(user_message)

            # Prepare options
            actual_temperature = (
                temperature if temperature is not None else self.temperature
            )
            options: dict[str, Any] = {"temperature": actual_temperature}

            # Prepare chat parameters
            chat_params = {
                "model": self.model_name,
                "messages": messages,
                "options": options,
            }

            # Add structured output format if requested
            if response_schema:
                chat_params["format"] = response_schema.model_json_schema()
                logger.info(
                    f"Structured output enabled with schema: {response_schema.__name__}"
                )

            if enable_grounding:
                logger.warning(
                    "Grounding requested but tool support is disabled for OllamaClient."
                )

            max_validation_retries = 2
            validation_retry = 0

            def build_validation_error(error: ValueError) -> str:
                error_parts = ["Your response does not match the required schema."]
                if response_schema:
                    schema_dict = response_schema.model_json_schema()
                    properties = schema_dict.get("properties", {})
                    for field_name, field_schema in properties.items():
                        if "enum" in field_schema:
                            enum_values = field_schema["enum"]
                            values_list = ", ".join(f'"{v}"' for v in enum_values)
                            error_parts.append(
                                f"Field '{field_name}' must be EXACTLY one of: {values_list}"
                            )
                error_parts.append(f"\nOriginal error: {error}")
                return "\n".join(error_parts)

            base_attrs = {"service": self.service, "model": self.model_name}

            while True:
                loop_attrs = {**base_attrs, "validation_retry": validation_retry}
                timer = self.telemetry.metrics.timer()
                try:
                    async with self.telemetry.async_create_span(
                        "ollama_chat", kind=SpanKind.CLIENT, attributes=loop_attrs
                    ):
                        response = await self.client.chat(**chat_params)
                    attrs = {**loop_attrs, "outcome": "success"}
                    self.telemetry.metrics.llm_latency.record(timer(), attrs)
                    self.telemetry.metrics.llm_requests.add(1, attrs)
                except Exception as e:
                    attrs = {
                        **loop_attrs,
                        "outcome": "error",
                        "error_type": type(e).__name__,
                    }
                    self.telemetry.metrics.llm_latency.record(timer(), attrs)
                    self.telemetry.metrics.llm_requests.add(1, attrs)
                    raise

                logger.info(
                    f"Ollama response (validation_retry={validation_retry}): {response}"
                )
                self._track_completion_metrics(
                    response,
                    method_name="generate_content",
                    validation_retry=validation_retry,
                )

                messages.append(response["message"])

                if response_schema:
                    try:
                        return self._parse_structured_response(
                            response, response_schema
                        )
                    except ValueError as error:
                        self.telemetry.metrics.structured_output_failures.add(
                            1,
                            {
                                "service": self.service,
                                "model": self.model_name,
                                "retry_attempt": validation_retry + 1,
                            },
                        )

                        if validation_retry >= max_validation_retries:
                            logger.error(
                                "Validation failed after %s attempts: %s",
                                max_validation_retries + 1,
                                error,
                            )
                            raise

                        validation_retry += 1
                        logger.warning(
                            "Validation failed (attempt %s/%s): %s",
                            validation_retry,
                            max_validation_retries + 1,
                            error,
                        )

                        messages.append(
                            {
                                "role": "user",
                                "content": f"{build_validation_error(error)}\n\nPlease fix and respond with valid JSON.",
                            }
                        )
                        continue

                return response["message"]["content"]

    def _strip_markdown_code_fence(self, content: str) -> str:
        """
        Strip markdown code fence from JSON content if present.

        Some models (like deepseek) wrap JSON in ```json ... ```.
        This method removes those fences to get pure JSON.

        Args:
            content: Content that may contain markdown code fences

        Returns:
            Clean JSON string without markdown formatting
        """
        import re

        # Remove ```json ... ``` or ``` ... ``` fences
        # Pattern: optional language identifier, then content, then closing fence
        pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
        match = re.match(pattern, content.strip(), re.DOTALL)

        if match:
            return match.group(1).strip()

        return content.strip()

    def _parse_structured_response(self, response: dict, response_schema: Type[T]) -> T:
        """
        Parse structured response from Ollama.

        Ollama's native format support should return valid JSON,
        but we validate it against the Pydantic schema.

        Args:
            response: Ollama response dictionary
            response_schema: Pydantic model to validate against

        Returns:
            Validated Pydantic model instance
        """
        try:
            content = response["message"]["content"]

            # Strip markdown code fences if present (some models like deepseek add them)
            clean_content = self._strip_markdown_code_fence(content)

            # Parse JSON and validate with Pydantic
            response_data = json.loads(clean_content)
            parsed_result = response_schema.model_validate(response_data)
            return parsed_result
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            logger.error(f"Failed to parse structured response: {e}", exc_info=True)
            self.telemetry.metrics.structured_output_failures.add(
                1, {"service": self.service, "model": self.model_name}
            )
            # Preserve the original error message for retry feedback
            raise ValueError(str(e)) from e
