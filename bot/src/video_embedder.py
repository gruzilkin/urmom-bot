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

logger = logging.getLogger(__name__)

# Patterns for supported video URLs
URL_PATTERNS = [
    # X/Twitter: https://x.com/user/status/123 or https://twitter.com/user/status/123
    re.compile(r"https?://(?:www\.)?(?:x|twitter)\.com/\w+/status/\d+"),
    # Instagram Reels: https://www.instagram.com/reel/ABC123/
    re.compile(r"https?://(?:www\.)?instagram\.com/reel/[\w-]+"),
]

# 8MB limit for Discord attachments (non-Nitro)
MAX_FILE_SIZE_BYTES = 8 * 1024 * 1024


@dataclass
class VideoEmbed:
    """Result of video embedding attempt."""

    # If file_data is set, attach this file
    file_data: bytes | None = None
    filename: str | None = None

    # If short_url is set, reply with this URL
    short_url: str | None = None

    # Original URL that was processed
    source_url: str | None = None


class VideoEmbedder:
    """
    Service to extract and embed videos from social media links.

    Detects X/Twitter and Instagram links, extracts direct video URLs,
    and either downloads the video (if < 8MB) or creates a TinyURL.
    """

    def __init__(
        self,
        cobalt_client: CobaltClient,
        tinyurl_client: TinyURLClient,
        telemetry: Telemetry,
        max_file_size: int = MAX_FILE_SIZE_BYTES,
    ):
        self.cobalt_client = cobalt_client
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
            VideoEmbed with either file data or short URL, or None if failed
        """
        async with self.telemetry.async_create_span("video_embedder.process_url") as span:
            span.set_attribute("source_url", url)

            try:
                # Extract direct video URL via Cobalt
                result = await self.cobalt_client.extract_video(url)
                video_url = result.url
                filename = result.filename

                # Try to download and check size
                file_data = await self._download_video(video_url)

                if file_data and len(file_data) <= self.max_file_size:
                    span.set_attribute("method", "attachment")
                    span.set_attribute("file_size", len(file_data))
                    return VideoEmbed(
                        file_data=file_data,
                        filename=filename,
                        source_url=url,
                    )

                span.set_attribute("method", "url")
                short_url = await self.tinyurl_client.shorten(video_url)
                return VideoEmbed(
                    short_url=short_url,
                    source_url=url,
                )

            except CobaltContentError as e:
                span.set_attribute("outcome", "skipped")
                return None

            except (CobaltError, TinyURLError):
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
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Video download returned status {response.status}")
                        return None

                    # Check Content-Length header first to avoid downloading huge files
                    content_length = response.headers.get("Content-Length")
                    if content_length and int(content_length) > self.max_file_size:
                        return None

                    # Download with size limit
                    chunks = []
                    total_size = 0
                    async for chunk in response.content.iter_chunked(64 * 1024):
                        total_size += len(chunk)
                        if total_size > self.max_file_size:
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
