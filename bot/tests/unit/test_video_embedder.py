import unittest
from unittest.mock import AsyncMock, Mock, patch

from cobalt_client import CobaltClient, VideoResult
from null_telemetry import NullTelemetry
from tinyurl_client import TinyURLClient
from video_embedder import VideoEmbedder


class TestFindVideoUrls(unittest.TestCase):
    def setUp(self):
        self.embedder = VideoEmbedder(Mock(), Mock(), NullTelemetry())

    def test_find_twitter_url(self):
        urls = self.embedder.find_video_urls("check this https://twitter.com/user/status/123")
        self.assertEqual(urls, ["https://twitter.com/user/status/123"])

    def test_find_x_url(self):
        urls = self.embedder.find_video_urls("look at https://x.com/user/status/456")
        self.assertEqual(urls, ["https://x.com/user/status/456"])

    def test_find_instagram_reel(self):
        urls = self.embedder.find_video_urls("https://www.instagram.com/reel/ABC123/")
        self.assertEqual(urls, ["https://www.instagram.com/reel/ABC123"])

    def test_find_no_urls(self):
        urls = self.embedder.find_video_urls("no links here")
        self.assertEqual(urls, [])

    def test_find_ignores_non_video_twitter(self):
        urls = self.embedder.find_video_urls("https://twitter.com/user")
        self.assertEqual(urls, [])


class TestProcessUrl(unittest.IsolatedAsyncioTestCase):
    @patch.object(VideoEmbedder, "_download_video")
    async def test_download_returns_bytes_creates_attachment(self, mock_download: AsyncMock):
        mock_download.return_value = b"video data"

        cobalt = Mock(spec=CobaltClient)
        cobalt.extract_video = AsyncMock(
            return_value=VideoResult(
                url="https://cdn.example.com/video.mp4",
                filename="video.mp4",
            )
        )
        tinyurl = Mock(spec=TinyURLClient)

        embedder = VideoEmbedder(cobalt, tinyurl, NullTelemetry())
        result = await embedder.process_url("https://x.com/user/status/123")

        self.assertEqual(result.file_data, b"video data")
        self.assertEqual(result.filename, "video.mp4")
        self.assertIsNone(result.short_url)
        tinyurl.shorten.assert_not_called()

    @patch.object(VideoEmbedder, "_download_video")
    async def test_download_returns_none_creates_short_url(self, mock_download: AsyncMock):
        mock_download.return_value = None

        cobalt = Mock(spec=CobaltClient)
        cobalt.extract_video = AsyncMock(
            return_value=VideoResult(
                url="https://cdn.example.com/video.mp4",
                filename="video.mp4",
            )
        )
        tinyurl = Mock(spec=TinyURLClient)
        tinyurl.shorten = AsyncMock(return_value="https://tinyurl.com/abc123")

        embedder = VideoEmbedder(cobalt, tinyurl, NullTelemetry())
        result = await embedder.process_url("https://x.com/user/status/123")

        self.assertIsNone(result.file_data)
        self.assertEqual(result.short_url, "https://tinyurl.com/abc123")
        tinyurl.shorten.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
