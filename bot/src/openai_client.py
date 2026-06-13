import logging
from collections.abc import Sequence
from typing import TypeVar

from openai import OpenAI, PermissionDeniedError
from openai.types.chat import ChatCompletion
from opentelemetry.trace import SpanKind
from pydantic import BaseModel

from ai_client import AIClient, BlockedException
from open_telemetry import Telemetry

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OpenAIClient(AIClient):
    """Base client for OpenAI-compatible chat completion APIs.

    Providers that expose an OpenAI-compatible endpoint (Grok, DeepSeek, ...) share
    the same request flow and differ only by base URL and telemetry service name.

    Structured output uses OpenAI's native structured outputs
    (``beta.chat.completions.parse``). Providers that do not support the strict
    ``json_schema`` response format can override ``_generate_structured``.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        telemetry: Telemetry,
        base_url: str,
        service: str,
        temperature: float = 0.1,
    ):
        if not api_key:
            raise ValueError(f"{service} API key not provided!")
        if not model_name:
            raise ValueError(f"{service} model name not provided!")

        self.model = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.temperature = temperature
        self.telemetry = telemetry
        self.service = service

    def _track_completion_metrics(self, completion: ChatCompletion, method_name: str, **additional_attributes):
        """Track metrics from completion response with detailed attributes"""
        usage = completion.usage
        attributes = {
            "service": self.service,
            "model": self.model_name,
        }

        attributes.update(additional_attributes)

        self.telemetry.track_token_usage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            attributes=attributes,
        )

    def _handle_api_exception(self, e: Exception, timer, base_attrs: dict) -> None:
        """Handle exceptions from API calls, record telemetry, and raise appropriate exception."""
        is_blocked = isinstance(e, PermissionDeniedError)
        attrs = {
            **base_attrs,
            "outcome": "blocked" if is_blocked else "error",
            "error_type": type(e).__name__,
        }
        self.telemetry.metrics.llm_latency.record(timer(), attrs)
        self.telemetry.metrics.llm_requests.add(1, attrs)

        if is_blocked:
            raise BlockedException(reason=f"Content violates safety guidelines: {str(e)}")
        raise

    async def generate_content(
        self,
        message: str,
        prompt: str = None,
        samples: Sequence[tuple[str, str]] | None = None,
        enable_grounding: bool = False,
        response_schema: type[T] | None = None,
        temperature: float | None = None,
        image_data: bytes | None = None,
        image_mime_type: str | None = None,
    ) -> str | T:
        if image_data:
            raise ValueError(f"{type(self).__name__} does not support image data.")
        base_attrs = {"service": self.service, "model": self.model_name}

        async with self.telemetry.async_create_span(
            "generate_content",
            kind=SpanKind.CLIENT,
            attributes=base_attrs,
        ):
            messages = []
            if prompt:
                messages.append({"role": "system", "content": prompt})

            if samples:
                for user_msg, assistant_msg in samples:
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": assistant_msg})

            messages.append({"role": "user", "content": message})

            logger.info(f"{self.service} input messages: {messages}")

            if enable_grounding:
                logger.warning(f"Grounding is disabled for {self.service}")

            actual_temperature = temperature if temperature is not None else self.temperature

            if response_schema:
                return self._generate_structured(messages, response_schema, actual_temperature, base_attrs)

            return self._generate_text(messages, actual_temperature, base_attrs)

    def _generate_text(self, messages: list[dict], temperature: float, base_attrs: dict) -> str:
        timer = self.telemetry.metrics.timer()
        try:
            completion = self.model.chat.completions.create(
                model=self.model_name, messages=messages, temperature=temperature
            )
            attrs = {**base_attrs, "outcome": "success"}
            self.telemetry.metrics.llm_latency.record(timer(), attrs)
            self.telemetry.metrics.llm_requests.add(1, attrs)
        except Exception as e:
            self._handle_api_exception(e, timer, base_attrs)

        logger.info(f"{self.service} completion: {completion}")
        self._track_completion_metrics(completion, method_name="generate_content")

        return completion.choices[0].message.content

    def _generate_structured(
        self, messages: list[dict], response_schema: type[T], temperature: float, base_attrs: dict
    ) -> T:
        """Native structured output via OpenAI's strict ``json_schema`` response format.

        Subclasses whose provider rejects this response format may override.
        """
        logger.info(f"Structured output enabled with schema: {response_schema.__name__}")
        timer = self.telemetry.metrics.timer()
        try:
            completion = self.model.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                response_format=response_schema,
            )
            attrs = {**base_attrs, "outcome": "success"}
            self.telemetry.metrics.llm_latency.record(timer(), attrs)
            self.telemetry.metrics.llm_requests.add(1, attrs)
        except Exception as e:
            self._handle_api_exception(e, timer, base_attrs)

        logger.info(f"{self.service} completion: {completion}")
        self._track_completion_metrics(completion, method_name="generate_content")

        parsed_result = completion.choices[0].message.parsed
        if parsed_result is None:
            self.telemetry.metrics.structured_output_failures.add(1, base_attrs)
            raise ValueError(
                f"Failed to parse response with schema {response_schema.__name__}:"
                f" {completion.choices[0].message.content}"
            )
        return parsed_result
