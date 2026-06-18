import json
import logging
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from open_telemetry import Telemetry
from openai_client import OpenAIClient

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class DeepSeekClient(OpenAIClient):
    """DeepSeek client over its OpenAI-compatible API.

    DeepSeek rejects the strict ``json_schema`` response format used by the base
    client, so structured output goes through JSON mode
    (``response_format={"type": "json_object"}``) with the schema injected into the
    prompt and validated client-side.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        telemetry: Telemetry,
        base_url: str = "https://api.deepseek.com",
        temperature: float = 0.7,
    ):
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            telemetry=telemetry,
            base_url=base_url,
            service="DEEPSEEK",
            temperature=temperature,
        )

    async def _generate_structured(
        self, messages: list[dict], response_schema: type[T], temperature: float, base_attrs: dict
    ) -> T:
        """Structured output via JSON mode, validated client-side with corrective retries."""
        logger.info(f"Structured output (json_object) enabled with schema: {response_schema.__name__}")
        instructions = self._schema_instructions(response_schema)
        if messages and messages[0]["role"] == "system":
            messages[0] = {"role": "system", "content": f"{messages[0]['content']}\n\n{instructions}"}
        else:
            messages.insert(0, {"role": "system", "content": f"You are a helpful assistant.\n\n{instructions}"})

        max_validation_retries = 2
        validation_retry = 0

        while True:
            loop_attrs = {**base_attrs, "validation_retry": validation_retry}
            timer = self.telemetry.metrics.timer()
            try:
                completion = await self.model.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                attrs = {**loop_attrs, "outcome": "success"}
                self.telemetry.metrics.llm_latency.record(timer(), attrs)
                self.telemetry.metrics.llm_requests.add(1, attrs)
            except Exception as e:
                self._handle_api_exception(e, timer, loop_attrs)

            logger.info(f"{self.service} completion (validation_retry={validation_retry}): {completion}")
            self._track_completion_metrics(
                completion, method_name="generate_content", validation_retry=validation_retry
            )

            content = completion.choices[0].message.content
            messages.append({"role": "assistant", "content": content})

            try:
                return self._parse_json_object(content, response_schema)
            except ValueError as error:
                self.telemetry.metrics.structured_output_failures.add(
                    1, {**base_attrs, "retry_attempt": validation_retry + 1}
                )

                if validation_retry >= max_validation_retries:
                    logger.error(
                        f"Structured output validation failed after {max_validation_retries + 1} attempts: {error}",
                        exc_info=True,
                    )
                    raise

                validation_retry += 1
                logger.warning(
                    f"Structured output validation failed "
                    f"(attempt {validation_retry}/{max_validation_retries + 1}): {error}"
                )

                feedback = self._build_validation_error(error, response_schema)
                messages.append({"role": "user", "content": f"{feedback}\n\nPlease fix and respond with valid JSON."})

    @staticmethod
    def _schema_instructions(response_schema: type[T]) -> str:
        return (
            "You must respond with a single valid JSON object that matches the "
            "following schema exactly. Do not include extra fields, omit required "
            "fields, add prose, or wrap the JSON in markdown code fences.\n"
            f"Schema:\n{json.dumps(response_schema.model_json_schema(), indent=2)}"
        )

    @staticmethod
    def _strip_markdown_code_fence(content: str) -> str:
        pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
        match = re.match(pattern, content.strip(), re.DOTALL)
        if match:
            return match.group(1).strip()
        return content.strip()

    def _parse_json_object(self, content: str, response_schema: type[T]) -> T:
        try:
            clean_content = self._strip_markdown_code_fence(content)
            response_data = json.loads(clean_content)
            return response_schema.model_validate(response_data)
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            raise ValueError(str(e)) from e

    @staticmethod
    def _build_validation_error(error: ValueError, response_schema: type[T]) -> str:
        error_parts = ["Your response does not match the required schema."]
        properties = response_schema.model_json_schema().get("properties", {})
        for field_name, field_schema in properties.items():
            if "enum" in field_schema:
                values_list = ", ".join(f'"{v}"' for v in field_schema["enum"])
                error_parts.append(f"Field '{field_name}' must be EXACTLY one of: {values_list}")
        error_parts.append(f"\nOriginal error: {error}")
        return "\n".join(error_parts)
