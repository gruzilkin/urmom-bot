import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from cobalt_client import CobaltClient, CobaltContentError
from null_telemetry import NullTelemetry


class AsyncContextManager:
    """Helper to mock async context managers."""

    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, *args):
        pass


class TestCobaltClient(unittest.IsolatedAsyncioTestCase):
    @patch("cobalt_client.aiohttp.ClientSession")
    async def test_extract_video_retries_on_network_error_then_succeeds(self, mock_session_cls: MagicMock):
        tunnel_response = {
            "status": "tunnel",
            "url": "https://cobalt.example.com/tunnel/abc123",
            "filename": "twitter_video.mp4",
        }

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value=tunnel_response)

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

        client = CobaltClient(base_url="http://cobalt:9000", telemetry=NullTelemetry(), max_tries=3)
        result = await client.extract_video("https://x.com/user/status/123")

        self.assertEqual(result.url, "https://cobalt.example.com/tunnel/abc123")
        self.assertEqual(result.filename, "twitter_video.mp4")
        self.assertEqual(call_count, 2)

    @patch("cobalt_client.aiohttp.ClientSession")
    async def test_extract_video_skips_photos(self, mock_session_cls: MagicMock):
        photo_response = {
            "status": "tunnel",
            "url": "https://cobalt.example.com/tunnel/abc123",
            "filename": "photo.jpg",
        }

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value=photo_response)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncContextManager(mock_response))
        mock_session_cls.return_value = AsyncContextManager(mock_session)

        client = CobaltClient(base_url="http://cobalt:9000", telemetry=NullTelemetry())

        with self.assertRaises(CobaltContentError) as ctx:
            await client.extract_video("https://x.com/user/status/123")

        self.assertEqual(ctx.exception.code, "content.not_video")


if __name__ == "__main__":
    unittest.main()
