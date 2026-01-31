import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from null_telemetry import NullTelemetry
from tinyurl_client import TinyURLClient


class AsyncContextManager:
    """Helper to mock async context managers."""

    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, *args):
        pass


class TestTinyURLClient(unittest.IsolatedAsyncioTestCase):
    @patch("tinyurl_client.aiohttp.ClientSession")
    async def test_shorten_retries_on_network_error_then_succeeds(self, mock_session_cls: MagicMock):
        success_response = {
            "code": 0,
            "data": {"tiny_url": "https://tinyurl.com/abc123"},
            "errors": [],
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=success_response)

        call_count = 0

        def post_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise aiohttp.ClientError("Network error")
            return AsyncContextManager(mock_response)

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=post_side_effect)
        mock_session_cls.return_value = AsyncContextManager(mock_session)

        client = TinyURLClient(api_token="test", telemetry=NullTelemetry(), max_tries=3)
        result = await client.shorten("https://example.com/long")

        self.assertEqual(result, "https://tinyurl.com/abc123")
        self.assertEqual(call_count, 2)


if __name__ == "__main__":
    unittest.main()
