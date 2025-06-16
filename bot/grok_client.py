import logging
from openai import OpenAI
from openai.types.chat import ChatCompletion
from ai_client import AIClient
from typing import List, Tuple
from opentelemetry.trace import SpanKind
from open_telemetry import Telemetry

logger = logging.getLogger(__name__)

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

    async def generate_content(self, message: str, prompt: str = None, samples: List[Tuple[str, str]] = None, enable_grounding: bool = False) -> str:
        async with self.telemetry.async_create_span("generate_content", kind=SpanKind.CLIENT) as span:
            messages = []
            if prompt:
                messages.append({"role": "system", "content": prompt})
            
            if samples:
                for user_msg, assistant_msg in samples:
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": assistant_msg})
                
            messages.append({"role": "user", "content": message})

            logger.info(f"Grok input messages: {messages}")

            # Configure search parameters based on grounding flag
            if enable_grounding:
                extra_body = {
                    "search_parameters": {
                        "mode": "on"
                    }
                }
                logger.info("[GROK] Grounding enabled with search mode: on")
            else:
                extra_body = None

            completion = self.model.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                extra_body=extra_body
            )

            logger.info(f"Grok completion: {completion}")
            self._track_completion_metrics(completion, method_name="generate_content")

            return completion.choices[0].message.content
