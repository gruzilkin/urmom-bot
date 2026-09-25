"""Claude Client using the Claude Code CLI tool."""

import asyncio
import json
import logging
import os
import tempfile
from typing import TypeVar

from ai_client import AIClient
from open_telemetry import Telemetry
from opentelemetry.trace import SpanKind
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_WEB_TOOLS = "WebSearch,WebFetch"


class ClaudeClient(AIClient):
    def __init__(
        self,
        telemetry: Telemetry,
        model_name: str,
        effort: str | None = None,
        enable_web_search: bool = True,
        timeout_seconds: float | None = None,
    ):
        self.model_name = model_name
        self.effort = effort
        self.telemetry = telemetry
        self.service = "CLAUDE"
        self.enable_web_search = enable_web_search
        self.timeout_seconds = timeout_seconds

    def __repr__(self) -> str:
        effort = self.effort or "default"
        return f"{type(self).__name__}({self.model_name}, effort={effort})"

    def _record(self, timer_value: float, attrs: dict[str, str], outcome: str, error_type: str | None = None) -> None:
        metric_attrs = {**attrs, "outcome": outcome}
        if error_type:
            metric_attrs["error_type"] = error_type
        self.telemetry.metrics.llm_latency.record(timer_value, metric_attrs)
        self.telemetry.metrics.llm_requests.add(1, metric_attrs)

    async def generate_content(
        self,
        message: str,
        prompt: str | None = None,
        samples: list[tuple[str, str]] | None = None,
        enable_grounding: bool = False,
        response_schema: type[T] | None = None,
        temperature: float | None = None,
        image_data: bytes | None = None,
        image_mime_type: str | None = None,
    ) -> str | T:
        if image_data:
            raise ValueError("ClaudeClient does not support image data.")

        base_attrs = {
            "service": self.service,
            "model": self.model_name,
            "reasoning_effort": self.effort or "default",
        }

        async with self.telemetry.async_create_span(
            "generate_content",
            kind=SpanKind.CLIENT,
            attributes=base_attrs,
        ):
            conversation_parts = []
            for user_msg, assistant_msg in samples or []:
                conversation_parts.append(f"Human: {user_msg}")
                conversation_parts.append(f"Assistant: {assistant_msg}")
            if conversation_parts:
                conversation_parts.append(f"Human: {message}")
                user_input = "\n\n".join(conversation_parts)
            else:
                user_input = message

            logger.info(f"Claude CLI system prompt: {prompt}")
            logger.info(f"Claude CLI input: {user_input}")

            claude_cmd = [
                "claude",
                "--print",
                "--output-format",
                "json",
                "--model",
                self.model_name,
                "--no-session-persistence",
                "--safe-mode",
                "--permission-prompts",
                "none",
            ]

            if self.enable_web_search or enable_grounding:
                claude_cmd.extend(["--tools", _WEB_TOOLS, "--allowedTools", _WEB_TOOLS])
            else:
                claude_cmd.extend(["--tools", ""])

            if self.effort:
                claude_cmd.extend(["--effort", self.effort])

            if response_schema:
                claude_cmd.extend(["--json-schema", json.dumps(response_schema.model_json_schema())])

            prompt_file = None
            if prompt:
                prompt_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8", delete=False)
                prompt_file.write(prompt)
                prompt_file.close()
                claude_cmd.extend(["--system-prompt-file", prompt_file.name])

            logger.info(f"Running Claude CLI command: {' '.join(claude_cmd)}")

            timer = self.telemetry.metrics.timer()
            try:
                process = await asyncio.create_subprocess_exec(
                    *claude_cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(input=user_input.encode("utf-8")),
                        timeout=self.timeout_seconds,
                    )
                except TimeoutError:
                    process.kill()
                    await process.wait()
                    logger.error(f"Claude CLI timed out after {self.timeout_seconds}s for model {self.model_name}")
                    self._record(timer(), base_attrs, "error", "Timeout")
                    raise
            finally:
                if prompt_file:
                    try:
                        os.unlink(prompt_file.name)
                    except OSError:
                        pass

            if process.returncode != 0:
                error_msg = stderr.decode().strip() or stdout.decode().strip() or "Unknown error"
                logger.error(f"Claude CLI command failed with return code {process.returncode}: {error_msg}")
                self._record(timer(), base_attrs, "error", "CLIError")
                raise RuntimeError(f"Claude CLI failed: {error_msg}")

            raw_output = stdout.decode().strip()
            logger.info(f"Claude CLI response: {raw_output}")

            try:
                result = json.loads(raw_output)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode Claude CLI JSON output: {e}", exc_info=True)
                self._record(timer(), base_attrs, "error", "InvalidOutput")
                raise RuntimeError(f"Claude CLI returned non-JSON output: {raw_output}") from e

            if result.get("is_error"):
                logger.error(f"Claude CLI reported an error result: {result.get('result')}")
                self._record(timer(), base_attrs, "error", "CLIError")
                raise RuntimeError(f"Claude CLI error result: {result.get('result')}")

            self._record(timer(), base_attrs, "success")

            if response_schema:
                try:
                    return response_schema.model_validate(result.get("structured_output"))
                except ValueError as e:
                    logger.error(f"Failed to parse structured Claude response: {e}", exc_info=True)
                    self.telemetry.metrics.structured_output_failures.add(1, base_attrs)
                    raise ValueError(
                        f"Failed to parse response with schema {response_schema.__name__}: {raw_output}"
                    ) from e

            response_text = (result.get("result") or "").strip()
            if not response_text:
                raise RuntimeError("Empty response from Claude CLI")
            return response_text
