import base64
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from null_telemetry import NullTelemetry
from ollama_client import OllamaClient
from schemas import YesNo


class TestOllamaClientInitialization(unittest.TestCase):
    def test_init_missing_api_key_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Ollama API key not provided"):
            OllamaClient(api_key="", model_name="model", telemetry=NullTelemetry())

    def test_init_missing_model_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Ollama model name not provided"):
            OllamaClient(api_key="key", model_name="", telemetry=NullTelemetry())


class TestOllamaClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        patcher = patch("ollama_client.AsyncClient", autospec=True)
        self.addCleanup(patcher.stop)
        self.mock_async_client_cls = patcher.start()
        self.mock_async_client = self.mock_async_client_cls.return_value
        self.mock_async_client.chat = AsyncMock()
        self.mock_async_client.web_search = AsyncMock()
        self.mock_async_client.web_fetch = AsyncMock()
        self.telemetry = NullTelemetry()
        self.client = OllamaClient(
            api_key="key",
            model_name="model",
            telemetry=self.telemetry,
            base_url="https://example.com",
        )

    def test_default_timeout_passed_to_async_client(self) -> None:
        timeout_arg = self.mock_async_client_cls.call_args.kwargs["timeout"]
        self.assertEqual(timeout_arg, 20.0)

    def test_custom_timeout_passed_to_async_client(self) -> None:
        self.mock_async_client_cls.reset_mock()
        custom_timeout = httpx.Timeout(5.0)

        OllamaClient(
            api_key="key",
            model_name="model",
            telemetry=self.telemetry,
            base_url="https://example.com",
            timeout=custom_timeout,
        )

        timeout_arg = self.mock_async_client_cls.call_args.kwargs["timeout"]
        self.assertIs(timeout_arg, custom_timeout)

    async def test_generate_content_builds_messages(self) -> None:
        self.mock_async_client.chat.return_value = {
            "message": {"content": "ACCESS DENIED"},
            "usage": {},
        }

        result = await self.client.generate_content(message="Secret challenge: gamma?")

        self.assertEqual(result, "ACCESS DENIED")
        self.mock_async_client.chat.assert_awaited_once()

    async def test_generate_content_with_image_encodes_base64(self) -> None:
        self.mock_async_client.chat.return_value = {
            "message": {"content": "Image described"},
            "usage": {},
        }
        image_bytes = b"\x89PNG"

        await self.client.generate_content(
            message="Describe the image",
            image_data=image_bytes,
            image_mime_type="image/png",
        )

        called_kwargs = self.mock_async_client.chat.await_args.kwargs
        messages = called_kwargs["messages"]
        image_messages = [m for m in messages if "images" in m]
        self.assertEqual(len(image_messages), 1)
        encoded = image_messages[0]["images"][0]
        self.assertEqual(encoded, base64.b64encode(image_bytes).decode("utf-8"))

    async def test_structured_output_adds_format_and_instruction(self) -> None:
        self.mock_async_client.chat.return_value = {
            "message": {"content": '{"answer": "YES"}'},
            "usage": {},
        }

        result = await self.client.generate_content(
            message="Is Paris the capital of France?",
            prompt="You are helpful.",
            response_schema=YesNo,
        )

        self.assertEqual(result.answer, "YES")
        called_kwargs = self.mock_async_client.chat.await_args.kwargs
        self.assertIn("format", called_kwargs)
        self.assertEqual(called_kwargs["format"], YesNo.model_json_schema())

    async def test_validation_retry_appends_error_message(self) -> None:
        invalid_response = {
            "message": {"role": "assistant", "content": "not json"},
            "usage": {},
        }
        valid_response = {
            "message": {"role": "assistant", "content": '{"answer": "YES"}'},
            "usage": {},
        }
        self.mock_async_client.chat.side_effect = [
            invalid_response,
            valid_response,
        ]

        result = await self.client.generate_content(
            message="Question?",
            response_schema=YesNo,
            temperature=0.0,
        )

        self.assertEqual(result.answer, "YES")
        self.assertEqual(self.mock_async_client.chat.await_count, 2)
        retry_messages = self.mock_async_client.chat.await_args_list[1].kwargs[
            "messages"
        ]
        self.assertEqual(retry_messages[-3], invalid_response["message"])
        self.assertEqual(retry_messages[-2]["role"], "user")
        self.assertIn("Field 'answer' must be EXACTLY", retry_messages[-2]["content"])

    def test_strip_markdown_code_fence(self) -> None:
        content = """```json
{"answer": "YES"}
```"""
        result = self.client._strip_markdown_code_fence(content)
        self.assertEqual(result, '{"answer": "YES"}')

    def test_parse_structured_response_failure(self) -> None:
        response = {"message": {"content": "not json"}}
        with self.assertRaisesRegex(
            ValueError, "Failed to parse response with schema YesNo"
        ):
            self.client._parse_structured_response(response, YesNo)
