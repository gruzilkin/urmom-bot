"""
Claude Client using the Claude CLI tool.

This client uses the Claude command-line interface to process requests.
Note: Some features like temperature control and grounding are not supported by the CLI.
"""

import asyncio
import json
import logging
from typing import List, Tuple, Type, TypeVar

from ai_client import AIClient
from open_telemetry import Telemetry
from opentelemetry.trace import SpanKind
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ClaudeClient(AIClient):
    def __init__(self, model_name: str = "claude", temperature: float = 0.7, telemetry: Telemetry = None):
        self.model_name = model_name  # Kept for compatibility but not used
        self.temperature = temperature  # Not used by CLI but kept for interface compatibility
        self.telemetry = telemetry

    async def generate_content(self, message: str, prompt: str = None, samples: List[Tuple[str, str]] = None, enable_grounding: bool = False, response_schema: Type[T] | None = None, temperature: float | None = None) -> str | T:
        async with self.telemetry.async_create_span("generate_content", kind=SpanKind.CLIENT) as span:
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
            
            try:
                # Build Claude CLI command
                claude_cmd = [
                    "claude",
                    "--print",
                    "--output-format", "text",
                    "--allowedTools", "WebSearch",
                    "--disallowedTools", "Bash", "Edit", "Write", "Create", "Read"
                ]
                
                logger.info(f"Running Claude CLI command: {' '.join(claude_cmd)}")
                
                # Run the Claude CLI command with input via stdin
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
                    raise RuntimeError(f"Claude CLI failed: {error_msg}")
                
                response_text = stdout.decode().strip()
                logger.info(f"Claude CLI response: {response_text}")
                
                # Parse structured response if schema was provided
                if response_schema:
                    try:
                        # Try to extract JSON from response
                        json_str = None
                        
                        # Look for JSON block or direct JSON
                        if "```json" in response_text:
                            start = response_text.find("```json") + 7
                            end = response_text.find("```", start)
                            json_str = response_text[start:end].strip()
                        elif response_text.strip().startswith("{") and response_text.strip().endswith("}"):
                            json_str = response_text.strip()
                        else:
                            # Try to parse the entire response as JSON
                            json_str = response_text.strip()
                        
                        response_data = json.loads(json_str)
                        parsed_result = response_schema.model_validate(response_data)
                        return parsed_result
                        
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.error(f"Failed to parse structured response: {e}")
                        raise ValueError(f"Failed to parse response with schema {response_schema.__name__}: {response_text}")
                
                return response_text
                
            except Exception as e:
                logger.error(f"Error in Claude CLI generate_content: {e}", exc_info=True)
                raise