import logging

import aiohttp
import backoff
from opentelemetry.trace import SpanKind
from pydantic import BaseModel

from open_telemetry import Telemetry

logger = logging.getLogger(__name__)


# === Response Models ===


class TinyURLData(BaseModel):
    """Data object from successful TinyURL response."""

    tiny_url: str
    domain: str | None = None
    alias: str | None = None


class TinyURLSuccessResponse(BaseModel):
    """Successful TinyURL API response."""

    code: int
    data: TinyURLData
    errors: list[str] = []


class TinyURLErrorResponse(BaseModel):
    """Error TinyURL API response."""

    code: int
    data: list | dict | None = None
    errors: list[str] = []


# === Exceptions ===


class TinyURLError(Exception):
    """Error creating TinyURL."""

    pass


class TinyURLConnectionError(TinyURLError):
    """Connection failed. Retry may succeed."""

    pass


class TinyURLClient:
    """Client for TinyURL URL shortening API."""

    API_URL = "https://api.tinyurl.com/create"

    def __init__(self, api_token: str, telemetry: Telemetry, max_tries: int = 3):
        """
        Initialize TinyURL client.

        Args:
            api_token: TinyURL API token from https://tinyurl.com/app/dev
            telemetry: Telemetry instance for tracing
            max_tries: Maximum retry attempts for connection errors
        """
        self.api_token = api_token
        self.telemetry = telemetry
        self.max_tries = max_tries

    async def shorten(self, url: str) -> str:
        """
        Shorten a URL using TinyURL API.

        Args:
            url: The long URL to shorten

        Returns:
            Shortened URL string

        Raises:
            TinyURLConnectionError: Connection failed after retries
            TinyURLError: If shortening fails
        """
        async with self.telemetry.async_create_span("tinyurl.shorten", kind=SpanKind.CLIENT) as span:

            @backoff.on_exception(
                backoff.expo,
                TinyURLConnectionError,
                max_tries=self.max_tries,
                jitter=backoff.full_jitter,
            )
            async def _do_request() -> str:
                return await self._shorten_impl(url)

            result = await _do_request()
            span.set_attribute("tiny_url", result)
            return result

    async def _shorten_impl(self, url: str) -> str:
        """Internal implementation of URL shortening."""
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        payload = {
            "url": url,
            "domain": "tinyurl.com",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.API_URL,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    data = await response.json()
                    return self._parse_response(data, response.status)

        except aiohttp.ClientError as e:
            logger.warning("TinyURL connection error (will retry)", exc_info=True)
            raise TinyURLConnectionError(f"Connection error: {e}") from e
        except TinyURLError:
            raise
        except Exception as e:
            logger.error("Unexpected TinyURL error", exc_info=True)
            raise TinyURLError(f"Unexpected error: {e}") from e

    def _parse_response(self, data: dict, http_status: int) -> str:
        """Parse TinyURL API response."""
        code = data.get("code", 0)

        if http_status != 200 or code != 0:
            error_response = TinyURLErrorResponse.model_validate(data)
            error_msg = error_response.errors[0] if error_response.errors else f"code {code}"
            raise TinyURLError(f"TinyURL API error: {error_msg}")

        success_response = TinyURLSuccessResponse.model_validate(data)
        return success_response.data.tiny_url
