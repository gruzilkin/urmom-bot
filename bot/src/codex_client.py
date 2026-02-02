"""
Codex Client using the OpenAI Codex CLI tool.

This client uses the Codex command-line interface to process requests.
"""

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


class CodexClient(AIClient):
    def __init__(
        self,
        telemetry: Telemetry,
        model_name: str = "gpt-5.2",
        enable_web_search: bool = True,
    ):
        self.model_name = model_name
        self.telemetry = telemetry
        self.service = "CODEX"
        self.enable_web_search = enable_web_search

    def _get_image_extension(self, mime_type: str | None) -> str:
        """Get file extension from MIME type."""
        mime_to_ext = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
        }
        return mime_to_ext.get(mime_type, ".png")

    async def generate_content(
        self,
        message: str,
        prompt: str = None,
        samples: list[tuple[str, str]] | None = None,
        enable_grounding: bool = False,
        response_schema: type[T] | None = None,
        temperature: float | None = None,
        image_data: bytes | None = None,
        image_mime_type: str | None = None,
    ) -> str | T:
        base_attrs = {"service": self.service, "model": self.model_name}

        async with self.telemetry.async_create_span(
            "generate_content",
            kind=SpanKind.CLIENT,
            attributes=base_attrs,
        ):
            samples = samples or []

            conversation_parts = []

            if prompt:
                conversation_parts.append(f"System: {prompt}")

            for user_msg, assistant_msg in samples:
                conversation_parts.append(f"Human: {user_msg}")
                conversation_parts.append(f"Assistant: {assistant_msg}")

            conversation_parts.append(f"Human: {message}")

            full_conversation = "\n\n".join(conversation_parts)

            logger.info(f"Codex CLI input: {full_conversation}")

            if temperature is not None:
                logger.warning("Temperature control not supported by Codex CLI")

            codex_cmd = [
                "codex",
                "exec",
                "-",
                "--skip-git-repo-check",
                "-s",
                "read-only",
                "-m",
                self.model_name,
                "--json",
            ]

            if self.enable_web_search or enable_grounding:
                codex_cmd.extend(["-c", 'web_search="live"'])

            image_file = None
            if image_data:
                ext = self._get_image_extension(image_mime_type)
                image_file = tempfile.NamedTemporaryFile(mode="wb", suffix=ext, delete=False)
                image_file.write(image_data)
                image_file.close()
                codex_cmd.extend(["-i", image_file.name])

            schema_file = None
            if response_schema:
                schema_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
                json.dump(response_schema.model_json_schema(), schema_file)
                schema_file.close()
                codex_cmd.extend(["--output-schema", schema_file.name])

            logger.info(f"Running Codex CLI command: {' '.join(codex_cmd)}")

            timer = self.telemetry.metrics.timer()
            try:
                process = await asyncio.create_subprocess_exec(
                    *codex_cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await process.communicate(input=full_conversation.encode("utf-8"))

                if process.returncode != 0:
                    error_msg = stderr.decode().strip() if stderr else "Unknown error"
                    logger.error(f"Codex CLI command failed with return code {process.returncode}: {error_msg}")
                    attrs_err = {**base_attrs, "outcome": "error", "error_type": "CLIError"}
                    self.telemetry.metrics.llm_latency.record(timer(), attrs_err)
                    self.telemetry.metrics.llm_requests.add(1, attrs_err)
                    raise RuntimeError(f"Codex CLI failed: {error_msg}")

                response_text = self._extract_agent_message(stdout.decode())
                logger.info(f"Codex CLI response: {response_text}")

                attrs = {**base_attrs, "outcome": "success"}
                self.telemetry.metrics.llm_latency.record(timer(), attrs)
                self.telemetry.metrics.llm_requests.add(1, attrs)

                if response_schema:
                    try:
                        response_data = json.loads(response_text)
                        parsed_result = response_schema.model_validate(response_data)
                        return parsed_result
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.error(f"Failed to parse structured response: {e}")
                        self.telemetry.metrics.structured_output_failures.add(
                            1, {"service": self.service, "model": self.model_name}
                        )
                        raise ValueError(
                            f"Failed to parse response with schema {response_schema.__name__}: {response_text}"
                        )

                return response_text

            finally:
                if schema_file:
                    try:
                        os.unlink(schema_file.name)
                    except OSError:
                        pass

                if image_file:
                    try:
                        os.unlink(image_file.name)
                    except OSError:
                        pass

    def _extract_agent_message(self, jsonl_output: str) -> str:
        """Extract the agent message from JSONL output."""
        for line in jsonl_output.strip().splitlines():
            try:
                event = json.loads(line)
                if event.get("type") == "item.completed":
                    item = event["item"]
                    if item["type"] == "agent_message":
                        text = item["text"]
                        if not text:
                            raise RuntimeError("Empty agent_message text in Codex output")
                        return text
            except json.JSONDecodeError:
                continue

        raise RuntimeError("No agent_message found in Codex output")
