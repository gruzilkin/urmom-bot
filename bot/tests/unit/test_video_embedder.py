import unittest
from unittest.mock import AsyncMock, Mock, patch

from cobalt_client import CobaltClient, VideoResult
from null_telemetry import NullTelemetry
from tinyurl_client import TinyURLClient
from video_compressor import CropBox, VideoCompressor
from video_embedder import VideoEmbedder


class TestFindVideoUrls(unittest.TestCase):
    def setUp(self):
        self.embedder = VideoEmbedder(Mock(), Mock(), Mock(), NullTelemetry())

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

    def test_find_reddit_url(self):
        urls = self.embedder.find_video_urls(
            "https://www.reddit.com/r/PublicFreakout/comments/1qwl6es/loud_fck_ice_chants/"
        )
        self.assertEqual(urls, ["https://www.reddit.com/r/PublicFreakout/comments/1qwl6es/loud_fck_ice_chants"])

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
        compressor = Mock(spec=VideoCompressor)
        compressor.analyze_crop = AsyncMock(return_value=None)
        tinyurl = Mock(spec=TinyURLClient)

        embedder = VideoEmbedder(cobalt, compressor, tinyurl, NullTelemetry())
        result = await embedder.process_url("https://x.com/user/status/123")

        self.assertEqual(result.file_data, b"video data")
        self.assertEqual(result.filename, "video.mp4")
        compressor.compress.assert_not_called()
        tinyurl.shorten.assert_not_called()

    @patch.object(VideoEmbedder, "_download_video")
    async def test_oversized_video_compresses_successfully(self, mock_download: AsyncMock):
        large_data = b"x" * (11 * 1024 * 1024)  # 10MB
        mock_download.return_value = large_data

        cobalt = Mock(spec=CobaltClient)
        cobalt.extract_video = AsyncMock(
            return_value=VideoResult(
                url="https://cdn.example.com/video.webm",
                filename="video.webm",
            )
        )
        compressed_data = b"small" * 100
        compressor = Mock(spec=VideoCompressor)
        compressor.compress = AsyncMock(return_value=compressed_data)
        tinyurl = Mock(spec=TinyURLClient)

        embedder = VideoEmbedder(cobalt, compressor, tinyurl, NullTelemetry())
        result = await embedder.process_url("https://x.com/user/status/123")

        self.assertEqual(result.file_data, compressed_data)
        self.assertEqual(result.filename, "video.mp4")
        compressor.compress.assert_awaited_once_with(large_data, "video.webm")
        tinyurl.shorten.assert_not_called()

    @patch.object(VideoEmbedder, "_download_video")
    async def test_compression_fails_redirect_falls_back_to_tinyurl(self, mock_download: AsyncMock):
        large_data = b"x" * (11 * 1024 * 1024)
        mock_download.return_value = large_data

        cobalt = Mock(spec=CobaltClient)
        cobalt.extract_video = AsyncMock(
            return_value=VideoResult(
                url="https://cdn.example.com/video.mp4",
                filename="video.mp4",
                is_tunnel=False,
            )
        )
        compressor = Mock(spec=VideoCompressor)
        compressor.compress = AsyncMock(return_value=None)
        tinyurl = Mock(spec=TinyURLClient)
        tinyurl.shorten = AsyncMock(return_value="https://tinyurl.com/abc123")

        embedder = VideoEmbedder(cobalt, compressor, tinyurl, NullTelemetry())
        result = await embedder.process_url("https://x.com/user/status/123")

        self.assertIsNone(result.file_data)
        self.assertEqual(result.short_url, "https://tinyurl.com/abc123")
        compressor.compress.assert_awaited_once()
        tinyurl.shorten.assert_awaited_once()

    @patch.object(VideoEmbedder, "_download_video")
    async def test_compression_fails_tunnel_returns_none(self, mock_download: AsyncMock):
        large_data = b"x" * (11 * 1024 * 1024)
        mock_download.return_value = large_data

        cobalt = Mock(spec=CobaltClient)
        cobalt.extract_video = AsyncMock(
            return_value=VideoResult(
                url="https://cobalt.internal/tunnel/abc",
                filename="video.mp4",
                is_tunnel=True,
            )
        )
        compressor = Mock(spec=VideoCompressor)
        compressor.compress = AsyncMock(return_value=None)
        tinyurl = Mock(spec=TinyURLClient)

        embedder = VideoEmbedder(cobalt, compressor, tinyurl, NullTelemetry())
        result = await embedder.process_url("https://x.com/user/status/123")

        self.assertIsNone(result)
        compressor.compress.assert_awaited_once()
        tinyurl.shorten.assert_not_called()

    @patch.object(VideoEmbedder, "_download_video")
    async def test_download_returns_none_redirect_creates_short_url(self, mock_download: AsyncMock):
        mock_download.return_value = None

        cobalt = Mock(spec=CobaltClient)
        cobalt.extract_video = AsyncMock(
            return_value=VideoResult(
                url="https://cdn.example.com/video.mp4",
                filename="video.mp4",
                is_tunnel=False,
            )
        )
        compressor = Mock(spec=VideoCompressor)
        tinyurl = Mock(spec=TinyURLClient)
        tinyurl.shorten = AsyncMock(return_value="https://tinyurl.com/abc123")

        embedder = VideoEmbedder(cobalt, compressor, tinyurl, NullTelemetry())
        result = await embedder.process_url("https://x.com/user/status/123")

        self.assertIsNone(result.file_data)
        self.assertEqual(result.short_url, "https://tinyurl.com/abc123")
        tinyurl.shorten.assert_awaited_once()
        compressor.compress.assert_not_called()

    @patch.object(VideoEmbedder, "_download_video")
    async def test_download_returns_none_tunnel_returns_none(self, mock_download: AsyncMock):
        mock_download.return_value = None

        cobalt = Mock(spec=CobaltClient)
        cobalt.extract_video = AsyncMock(
            return_value=VideoResult(
                url="https://cobalt.internal/tunnel/abc",
                filename="video.mp4",
                is_tunnel=True,
            )
        )
        compressor = Mock(spec=VideoCompressor)
        tinyurl = Mock(spec=TinyURLClient)

        embedder = VideoEmbedder(cobalt, compressor, tinyurl, NullTelemetry())
        result = await embedder.process_url("https://x.com/user/status/123")

        self.assertIsNone(result)
        tinyurl.shorten.assert_not_called()
        compressor.compress.assert_not_called()


class TestWithinLimitsCrop(unittest.IsolatedAsyncioTestCase):
    @patch.object(VideoEmbedder, "_download_video")
    async def test_significant_bars_triggers_compression(self, mock_download: AsyncMock):
        mock_download.return_value = b"video data"
        crop = CropBox(w=1920, h=800, x=0, y=140, pixel_reduction=0.26)
        compressed = b"cropped video"

        cobalt = Mock(spec=CobaltClient)
        cobalt.extract_video = AsyncMock(
            return_value=VideoResult(url="https://cdn.example.com/video.mp4", filename="video.mp4")
        )
        compressor = Mock(spec=VideoCompressor)
        compressor.analyze_crop = AsyncMock(return_value=crop)
        compressor.compress = AsyncMock(return_value=compressed)
        tinyurl = Mock(spec=TinyURLClient)

        embedder = VideoEmbedder(cobalt, compressor, tinyurl, NullTelemetry())
        result = await embedder.process_url("https://x.com/user/status/123")

        self.assertEqual(result.file_data, compressed)
        self.assertEqual(result.filename, "video.mp4")
        compressor.compress.assert_awaited_once_with(b"video data", "video.mp4", crop=crop)

    @patch.object(VideoEmbedder, "_download_video")
    async def test_minor_bars_returned_as_is(self, mock_download: AsyncMock):
        mock_download.return_value = b"video data"
        crop = CropBox(w=1920, h=1026, x=0, y=27, pixel_reduction=0.05)

        cobalt = Mock(spec=CobaltClient)
        cobalt.extract_video = AsyncMock(
            return_value=VideoResult(url="https://cdn.example.com/video.mp4", filename="video.mp4")
        )
        compressor = Mock(spec=VideoCompressor)
        compressor.analyze_crop = AsyncMock(return_value=crop)
        tinyurl = Mock(spec=TinyURLClient)

        embedder = VideoEmbedder(cobalt, compressor, tinyurl, NullTelemetry())
        result = await embedder.process_url("https://x.com/user/status/123")

        self.assertEqual(result.file_data, b"video data")
        self.assertEqual(result.filename, "video.mp4")
        compressor.compress.assert_not_called()

    @patch.object(VideoEmbedder, "_download_video")
    async def test_no_crop_detected_returned_as_is(self, mock_download: AsyncMock):
        mock_download.return_value = b"video data"

        cobalt = Mock(spec=CobaltClient)
        cobalt.extract_video = AsyncMock(
            return_value=VideoResult(url="https://cdn.example.com/video.mp4", filename="video.mp4")
        )
        compressor = Mock(spec=VideoCompressor)
        compressor.analyze_crop = AsyncMock(return_value=None)
        tinyurl = Mock(spec=TinyURLClient)

        embedder = VideoEmbedder(cobalt, compressor, tinyurl, NullTelemetry())
        result = await embedder.process_url("https://x.com/user/status/123")

        self.assertEqual(result.file_data, b"video data")
        self.assertEqual(result.filename, "video.mp4")
        compressor.compress.assert_not_called()

    @patch.object(VideoEmbedder, "_download_video")
    async def test_crop_compression_fails_returned_as_is(self, mock_download: AsyncMock):
        mock_download.return_value = b"video data"
        crop = CropBox(w=1920, h=800, x=0, y=140, pixel_reduction=0.26)

        cobalt = Mock(spec=CobaltClient)
        cobalt.extract_video = AsyncMock(
            return_value=VideoResult(url="https://cdn.example.com/video.mp4", filename="video.mp4")
        )
        compressor = Mock(spec=VideoCompressor)
        compressor.analyze_crop = AsyncMock(return_value=crop)
        compressor.compress = AsyncMock(return_value=None)
        tinyurl = Mock(spec=TinyURLClient)

        embedder = VideoEmbedder(cobalt, compressor, tinyurl, NullTelemetry())
        result = await embedder.process_url("https://x.com/user/status/123")

        self.assertEqual(result.file_data, b"video data")
        self.assertEqual(result.filename, "video.mp4")


if __name__ == "__main__":
    unittest.main()
