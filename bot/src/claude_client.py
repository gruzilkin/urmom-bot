"""
Claude Client using the Claude CLI tool.

This client uses the Claude command-line interface to process requests.
Note: Some features like temperature control and grounding are not supported by the CLI.
"""

import asyncio
import json
import logging
from typing import TypeVar

from ai_client import AIClient
from open_telemetry import Telemetry
from opentelemetry.trace import SpanKind
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ClaudeClient(AIClient):
    def __init__(self, telemetry: Telemetry, model_name: str = "claude_code"):
        self.model_name = model_name
        self.telemetry = telemetry
        self.service = "CLAUDE"

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
        if image_data:
            raise ValueError("ClaudeClient does not support image data.")
        base_attrs = {"service": self.service, "model": self.model_name}

        async with self.telemetry.async_create_span(
            "generate_content",
            kind=SpanKind.CLIENT,
            attributes=base_attrs,
        ):
            samples = samples or []
            
            # Build conversation context
            conversation_parts = []
            
            # Add system prompt if provided
            if prompt:
                conversation_parts.append(f"System: {prompt}")
            
            # Add samples as conversation history
            for user_msg, assistant_msg in samples:
                conversation_parts.append(f"Human: {user_msg}")
                conversation_parts.append(f"Assistant: {assistant_msg}")

            # Add the main message
            conversation_parts.append(f"Human: {message}")
            
            # If structured output is requested, modify the message
            if response_schema:
                schema_instruction = f"\n\nPlease respond with a valid JSON object that matches this schema: {response_schema.model_json_schema()}"
                conversation_parts[-1] += schema_instruction
            
            full_conversation = "\n\n".join(conversation_parts)
            
            logger.info(f"Claude CLI input: {full_conversation}")
            
            # Log unsupported features
            if enable_grounding:
                logger.warning("Grounding not supported by Claude CLI")
            
            # Build Claude CLI command
            claude_cmd = [
                "claude",
                "--print",
                "--output-format", "text",
                "--allowedTools", "WebSearch", "WebFetch",
                "--disallowedTools", "Bash", "Edit", "Write", "Create", "Read"
            ]
            
            logger.info(f"Running Claude CLI command: {' '.join(claude_cmd)}")
            
            # Run the Claude CLI command with input via stdin
            timer = self.telemetry.metrics.timer()
            process = await asyncio.create_subprocess_exec(
                *claude_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Send the conversation through stdin
            stdout, stderr = await process.communicate(input=full_conversation.encode('utf-8'))
            
            if process.returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                logger.error(f"Claude CLI command failed with return code {process.returncode}: {error_msg}")
                attrs_err = {**base_attrs, "outcome": "error", "error_type": "CLIError"}
                self.telemetry.metrics.llm_latency.record(timer(), attrs_err)
                self.telemetry.metrics.llm_requests.add(1, attrs_err)
                raise RuntimeError(f"Claude CLI failed: {error_msg}")
            
            response_text = stdout.decode().strip()
            logger.info(f"Claude CLI response: {response_text}")

            attrs = {**base_attrs, "outcome": "success"}
            self.telemetry.metrics.llm_latency.record(timer(), attrs)
            self.telemetry.metrics.llm_requests.add(1, attrs)
            
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
                    logger.error(f"Failed to parse structured response: {e}")
                    self.telemetry.metrics.structured_output_failures.add(1, {"service": self.service, "model": self.model_name})
                    raise ValueError(f"Failed to parse response with schema {response_schema.__name__}: {response_text}")
            
            return response_text
