import logging
from openai import OpenAI
from openai.types.chat import ChatCompletion
from ai_client import AIClient
from typing import List, Tuple, Type, TypeVar
from opentelemetry.trace import SpanKind
from open_telemetry import Telemetry
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

class GrokClient(AIClient):
    def __init__(self, api_key: str, model_name: str, telemetry: Telemetry, temperature: float = 0.1):
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
        usage = completion.usage
        attributes = {
            "service": "GROK",
            "model": self.model_name,
        }
        
        # Add any additional attributes passed in
        attributes.update(additional_attributes)
        
        self.telemetry.track_token_usage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            attributes=attributes
        )

    async def generate_content(self, message: str, prompt: str = None, samples: List[Tuple[str, str]] = None, enable_grounding: bool = False, response_schema: Type[T] | None = None, temperature: float | None = None, image_data: bytes | None = None, image_mime_type: str | None = None) -> str | T:
        if image_data:
            raise ValueError("GrokClient does not support image data.")
        async with self.telemetry.async_create_span("generate_content", kind=SpanKind.CLIENT):
            messages = []
            if prompt:
                messages.append({"role": "system", "content": prompt})
            
            if samples:
                for user_msg, assistant_msg in samples:
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": assistant_msg})
                
            messages.append({"role": "user", "content": message})

            logger.info(f"Grok input messages: {messages}")

            # Log unsupported features
            if enable_grounding:
                logger.warning("Grounding is disabled for Grok")

            # Use provided temperature or fallback to instance temperature
            actual_temperature = temperature if temperature is not None else self.temperature
            
            # Use structured output if schema is provided
            timer = self.telemetry.metrics.timer()
            if response_schema:
                logger.info(f"Structured output enabled with schema: {response_schema.__name__}")
                try:
                    completion = self.model.beta.chat.completions.parse(
                        model=self.model_name,
                        messages=messages,
                        temperature=actual_temperature,
                        response_format=response_schema
                    )
                    attrs = {"service": "GROK", "model": self.model_name, "outcome": "success"}
                    self.telemetry.metrics.llm_latency.record(timer(), attrs)
                    self.telemetry.metrics.llm_requests.add(1, attrs)
                except Exception as e:
                    attrs = {"service": "GROK", "model": self.model_name, "outcome": "error", "error_type": type(e).__name__}
                    self.telemetry.metrics.llm_latency.record(timer(), attrs)
                    self.telemetry.metrics.llm_requests.add(1, attrs)
                    raise
                
                logger.info(f"Grok completion: {completion}")
                self._track_completion_metrics(completion, method_name="generate_content")
                
                parsed_result = completion.choices[0].message.parsed
                if parsed_result is None:
                    self.telemetry.metrics.structured_output_failures.add(1, {"service": "GROK", "model": self.model_name})
                    raise ValueError(f"Failed to parse response with schema {response_schema.__name__}: {completion.choices[0].message.content}")
                return parsed_result
            else:
                try:
                    completion = self.model.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=actual_temperature
                    )
                    attrs = {"service": "GROK", "model": self.model_name, "outcome": "success"}
                    self.telemetry.metrics.llm_latency.record(timer(), attrs)
                    self.telemetry.metrics.llm_requests.add(1, attrs)
                except Exception as e:
                    attrs = {"service": "GROK", "model": self.model_name, "outcome": "error", "error_type": type(e).__name__}
                    self.telemetry.metrics.llm_latency.record(timer(), attrs)
                    self.telemetry.metrics.llm_requests.add(1, attrs)
                    raise

                logger.info(f"Grok completion: {completion}")
                self._track_completion_metrics(completion, method_name="generate_content")

                return completion.choices[0].message.content
