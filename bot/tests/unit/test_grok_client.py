"""Unit tests for GrokClient."""

import unittest
from unittest.mock import Mock, patch
from openai import PermissionDeniedError

from ai_client import BlockedException
from grok_client import GrokClient
from null_telemetry import NullTelemetry
from schemas import YesNo


class TestGrokClientExceptionHandling(unittest.IsolatedAsyncioTestCase):
    """Test exception handling in GrokClient."""

    def setUp(self) -> None:
        """Set up test dependencies."""
        patcher = patch("grok_client.OpenAI")
        self.addCleanup(patcher.stop)
        self.mock_openai_cls = patcher.start()
        self.mock_openai = self.mock_openai_cls.return_value

        self.telemetry = NullTelemetry()

        self.client = GrokClient(
            api_key="test-api-key",
            model_name="test-model",
            telemetry=self.telemetry,
            temperature=0.1,
        )

    async def test_permission_denied_error_structured_output(self) -> None:
        """Test that PermissionDeniedError is converted to BlockedException in structured output path."""
        error_response = Mock()
        error_response.status_code = 403
        error_response.json.return_value = {
            "error": "Content violates usage guidelines",
            "code": "The caller does not have permission",
        }

        permission_error = PermissionDeniedError(
            "Error code: 403 - Content violates usage guidelines",
            response=error_response,
            body={"error": "Content violates usage guidelines"},
        )

        self.mock_openai.beta.chat.completions.parse = Mock(side_effect=permission_error)

        with self.assertRaises(BlockedException) as context:
            await self.client.generate_content(message="Test message", response_schema=YesNo)

        self.assertIn("Content violates safety guidelines", str(context.exception.reason))

    async def test_permission_denied_error_non_structured_output(self) -> None:
        """Test that PermissionDeniedError is converted to BlockedException in non-structured path."""
        error_response = Mock()
        error_response.status_code = 403
        error_response.json.return_value = {
            "error": "Content violates usage guidelines",
            "code": "The caller does not have permission",
        }

        permission_error = PermissionDeniedError(
            "Error code: 403 - Content violates usage guidelines",
            response=error_response,
            body={"error": "Content violates usage guidelines"},
        )

        self.mock_openai.chat.completions.create = Mock(side_effect=permission_error)

        with self.assertRaises(BlockedException) as context:
            await self.client.generate_content(message="Test message")

        self.assertIn("Content violates safety guidelines", str(context.exception.reason))

    async def test_generic_exception_not_converted(self) -> None:
        """Test that generic exceptions are not converted to BlockedException."""
        generic_error = ValueError("Some other error")
        self.mock_openai.chat.completions.create = Mock(side_effect=generic_error)

        with self.assertRaises(ValueError) as context:
            await self.client.generate_content(message="Test message")

        self.assertEqual(str(context.exception), "Some other error")


if __name__ == "__main__":
    unittest.main()
