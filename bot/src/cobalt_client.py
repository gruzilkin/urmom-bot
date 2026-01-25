import logging
from dataclasses import dataclass
from typing import Literal

import aiohttp
import backoff
from opentelemetry.trace import SpanKind
from pydantic import BaseModel

from open_telemetry import Telemetry

logger = logging.getLogger(__name__)


# === Response Models ===


class CobaltErrorDetail(BaseModel):
    """Error details from Cobalt API."""

    code: str
    context: dict | None = None


class CobaltPickerItem(BaseModel):
    """Individual item in a picker response."""

    type: Literal["photo", "video", "gif"]
    url: str
    thumb: str | None = None


class CobaltTunnelResponse(BaseModel):
    """Response for tunnel/redirect status."""

    status: Literal["tunnel", "redirect"]
    url: str
    filename: str = "video.mp4"


class CobaltPickerResponse(BaseModel):
    """Response for picker status (multiple items)."""

    status: Literal["picker"]
    picker: list[CobaltPickerItem]
    audio: str | None = None
    audioFilename: str | None = None


class CobaltErrorResponse(BaseModel):
    """Response for error status."""

    status: Literal["error"]
    error: CobaltErrorDetail


# === Exceptions ===


class CobaltError(Exception):
    """Base exception for Cobalt API errors."""

    def __init__(self, code: str, context: dict | None = None):
        self.code = code
        self.context = context
        super().__init__(f"Cobalt error: {code}")


class CobaltConnectionError(CobaltError):
    """Connection failed. Retry may succeed."""

    def __init__(self, code: str = "connection_error", context: dict | None = None):
        super().__init__(code, context)


class CobaltContentError(CobaltError):
    """Content-related error (private, unavailable, etc.). Do not retry."""

    pass


# === Result ===


@dataclass
class VideoResult:
    """Result from Cobalt video extraction."""

    url: str
    filename: str


class CobaltClient:
    """Client for Cobalt media extraction API."""

    def __init__(self, base_url: str, telemetry: Telemetry, max_tries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.telemetry = telemetry
        self.max_tries = max_tries

    async def extract_video(self, url: str) -> VideoResult:
        """
        Extract video URL from a social media link.

        Args:
            url: The social media URL (X/Twitter, Instagram, etc.)

        Returns:
            VideoResult with download URL and filename

        Raises:
            CobaltConnectionError: Connection failed after retries
            CobaltContentError: Content issue (private, not found, etc.) - don't retry
            CobaltError: Other API errors
        """
        async with self.telemetry.async_create_span("cobalt.extract_video", kind=SpanKind.CLIENT) as span:
            span.set_attribute("source_url", url)

            @backoff.on_exception(
                backoff.expo,
                CobaltConnectionError,
                max_tries=self.max_tries,
                jitter=backoff.full_jitter,
            )
            async def _do_request() -> VideoResult:
                return await self._extract_video_impl(url)

            return await _do_request()

    async def _extract_video_impl(self, url: str) -> VideoResult:
        """Internal implementation of video extraction."""
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        payload = {"url": url, "videoQuality": "max", "filenameStyle": "basic"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/",
                    json=payload,
                    headers=headers,
                ) as response:
                    data = await response.json()
                    return self._parse_response(data)

        except aiohttp.ClientError as e:
            logger.warning("Cobalt API connection error (will retry)", exc_info=True)
            raise CobaltConnectionError() from e
        except CobaltError:
            raise
        except Exception as e:
            logger.error("Unexpected Cobalt API error", exc_info=True)
            raise CobaltError("unknown_error") from e

    def _parse_response(self, data: dict) -> VideoResult:
        """Parse Cobalt API response into VideoResult."""
        status = data.get("status")

        if status == "error":
            error_response = CobaltErrorResponse.model_validate(data)
            code = error_response.error.code
            context = error_response.error.context

            if "content." in code or "link." in code:
                raise CobaltContentError(code, context)
            raise CobaltError(code, context)

        if status in ("tunnel", "redirect"):
            tunnel_response = CobaltTunnelResponse.model_validate(data)
            return VideoResult(
                url=tunnel_response.url,
                filename=tunnel_response.filename,
            )

        if status == "picker":
            picker_response = CobaltPickerResponse.model_validate(data)

            # Return first video from picker
            for item in picker_response.picker:
                if item.type == "video":
                    return VideoResult(
                        url=item.url,
                        filename=picker_response.audioFilename or "video.mp4",
                    )

            # Fallback to first item with URL
            if picker_response.picker:
                first = picker_response.picker[0]
                return VideoResult(
                    url=first.url,
                    filename="media",
                )
            raise CobaltContentError("fetch.empty")

        raise CobaltError(f"unexpected_status:{status}")
