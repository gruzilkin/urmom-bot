import logging
import re
from dataclasses import dataclass

import aiohttp

from cobalt_client import (
    CobaltClient,
    CobaltContentError,
    CobaltError,
)
from open_telemetry import Telemetry
from tinyurl_client import TinyURLClient, TinyURLError
from video_compressor import VideoCompressionError, VideoCompressor

logger = logging.getLogger(__name__)

# Patterns for supported video URLs
URL_PATTERNS = [
    # X/Twitter: https://x.com/user/status/123 or https://twitter.com/user/status/123
    re.compile(r"https?://(?:www\.)?(?:x|twitter)\.com/\w+/status/\d+"),
    # Instagram Reels: https://www.instagram.com/reel/ABC123/
    re.compile(r"https?://(?:www\.)?instagram\.com/reel/[\w-]+"),
    # Reddit: https://www.reddit.com/r/subreddit/comments/id/title/
    re.compile(r"https?://(?:www\.)?reddit\.com/r/\w+/comments/\w+/\w+"),
]

# 10MB limit for Discord attachments (non-Nitro)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

# Hard ceiling for downloads before attempting compression
MAX_DOWNLOAD_SIZE_BYTES = 200 * 1024 * 1024


@dataclass
class VideoEmbed:
    """Result of video embedding attempt."""

    # If file_data is set, attach this file
    file_data: bytes | None = None
    filename: str | None = None

    # If short_url is set, reply with this URL (fallback for huge videos)
    short_url: str | None = None

    # Original URL that was processed
    source_url: str | None = None


class VideoEmbedder:
    """
    Service to extract and embed videos from social media links.

    Detects X/Twitter and Instagram links, extracts direct video URLs,
    and either downloads the video (if < 8MB) or compresses it with ffmpeg.
    """

    def __init__(
        self,
        cobalt_client: CobaltClient,
        video_compressor: VideoCompressor,
        tinyurl_client: TinyURLClient,
        telemetry: Telemetry,
        max_file_size: int = MAX_FILE_SIZE_BYTES,
    ):
        self.cobalt_client = cobalt_client
        self.video_compressor = video_compressor
        self.tinyurl_client = tinyurl_client
        self.telemetry = telemetry
        self.max_file_size = max_file_size

    def find_video_urls(self, text: str) -> list[str]:
        """
        Find all supported video URLs in text.

        Args:
            text: Message text to search

        Returns:
            List of matching URLs
        """
        urls = []
        for pattern in URL_PATTERNS:
            urls.extend(pattern.findall(text))
        return urls

    async def process_url(self, url: str) -> VideoEmbed | None:
        """
        Process a single video URL.

        Args:
            url: The social media URL to process

        Returns:
            VideoEmbed with file data, or None if failed
        """
        async with self.telemetry.async_create_span("video_embedder.process_url") as span:
            span.set_attribute("source_url", url)

            try:
                # Extract direct video URL via Cobalt
                result = await self.cobalt_client.extract_video(url)
                video_url = result.url
                filename = result.filename

                # Download the video
                file_data = await self._download_video(video_url)

                if file_data is None:
                    if result.is_tunnel:
                        span.set_attribute("outcome", "tunnel_too_large")
                        return None
                    # Direct CDN link too large to download — fall back to TinyURL
                    span.set_attribute("method", "url")
                    short_url = await self.tinyurl_client.shorten(video_url)
                    return VideoEmbed(short_url=short_url, source_url=url)

                if len(file_data) <= self.max_file_size:
                    span.set_attribute("method", "attachment")
                    span.set_attribute("file_size", len(file_data))
                    return VideoEmbed(
                        file_data=file_data,
                        filename=filename,
                        source_url=url,
                    )

                span.set_attribute("method", "compress")
                compressed = await self.video_compressor.compress(file_data, filename)
                if compressed is None:
                    if not result.is_tunnel:
                        span.set_attribute("method", "url")
                        short_url = await self.tinyurl_client.shorten(video_url)
                        return VideoEmbed(short_url=short_url, source_url=url)
                    span.set_attribute("outcome", "compression_failed")
                    return None

                span.set_attribute("file_size", len(compressed))
                return VideoEmbed(
                    file_data=compressed,
                    filename=_to_mp4_filename(filename),
                    source_url=url,
                )

            except CobaltContentError as e:
                logger.warning(f"Skipping video embed for {url}: {e.code}", exc_info=True)
                span.set_attribute("outcome", "skipped")
                return None

            except (CobaltError, TinyURLError, VideoCompressionError) as e:
                logger.warning(f"Video embed failed for {url}: {e}", exc_info=True)
                span.set_attribute("outcome", "error")
                return None

            except Exception:
                logger.error("Unexpected error in video embedder", exc_info=True)
                span.set_attribute("outcome", "error")
                return None

    async def _download_video(self, url: str) -> bytes | None:
        """
        Download video from URL.

        Args:
            url: Direct video URL

        Returns:
            Video bytes or None if download fails
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Video download returned status {response.status}")
                        return None

                    # Check Content-Length header to avoid downloading absurdly large files
                    content_length = response.headers.get("Content-Length")
                    if content_length and int(content_length) > MAX_DOWNLOAD_SIZE_BYTES:
                        return None

                    # Download with hard ceiling
                    chunks = []
                    total_size = 0
                    async for chunk in response.content.iter_chunked(64 * 1024):
                        total_size += len(chunk)
                        if total_size > MAX_DOWNLOAD_SIZE_BYTES:
                            return None
                        chunks.append(chunk)

                    return b"".join(chunks)

        except aiohttp.ClientError:
            return None
        except Exception:
            return None

    async def process_message(self, text: str) -> list[VideoEmbed]:
        """
        Process all video URLs in a message.

        Args:
            text: Message text

        Returns:
            List of VideoEmbed results (may be empty)
        """
        urls = self.find_video_urls(text)
        if not urls:
            return []

        results = []
        for url in urls:
            embed = await self.process_url(url)
            if embed:
                results.append(embed)

        return results


def _to_mp4_filename(filename: str) -> str:
    base, _ = filename.rsplit(".", 1) if "." in filename else (filename, "")
    return f"{base}.mp4"
