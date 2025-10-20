"""Wrappers that add retry and fallback behaviour to ``AIClient`` implementations."""

from __future__ import annotations

import logging
import random
from typing import Iterable, Callable, Any

import backoff

from ai_client import AIClient, BlockedException
from open_telemetry import Telemetry

logger = logging.getLogger(__name__)


class RetryAIClient(AIClient):
    """Apply configurable retry policy to an underlying async AI client.

    Can retry based on either max_time (seconds) or max_tries (attempts).
    Useful for rate-limited APIs (max_time) or flaky services (max_tries).
    """

    def __init__(
        self,
        delegate: AIClient,
        telemetry: Telemetry,
        max_time: int | None = None,
        max_tries: int | None = None,
        jitter: bool = False,
    ) -> None:
        if max_time is not None and max_tries is not None:
            raise ValueError("Cannot specify both max_time and max_tries")
        if max_time is None and max_tries is None:
            max_tries = 3

        self._delegate = delegate
        self._max_time = max_time
        self._max_tries = max_tries
        self._jitter = backoff.full_jitter if jitter else None
        self._telemetry = telemetry

    async def generate_content(
        self,
        message: str,
        prompt: str | None = None,
        samples: list[tuple[str, str]] | None = None,
        enable_grounding: bool = False,
        response_schema=None,
        temperature: float | None = None,
        image_data: bytes | None = None,
        image_mime_type: str | None = None,
    ):
        async with self._telemetry.async_create_span("retry_generate_content"):

            async def _do_call():
                return await self._delegate.generate_content(
                    message=message,
                    prompt=prompt,
                    samples=samples,
                    enable_grounding=enable_grounding,
                    response_schema=response_schema,
                    temperature=temperature,
                    image_data=image_data,
                    image_mime_type=image_mime_type,
                )

            backoff_kwargs = {
                "wait_gen": backoff.expo,
                "exception": Exception,
                "jitter": self._jitter,
                "giveup": lambda e: isinstance(e, BlockedException),
            }

            if self._max_time is not None:
                backoff_kwargs["max_time"] = self._max_time
            else:
                backoff_kwargs["max_tries"] = self._max_tries

            wrapped = backoff.on_exception(**backoff_kwargs)(_do_call)
            return await wrapped()


class CompositeAIClient(AIClient):
    """Try a sequence of AI clients until one succeeds."""

    def __init__(
        self,
        clients: Iterable[AIClient],
        telemetry: Telemetry,
        is_bad_response: Callable[[Any], bool] | None = None,
        shuffle: bool = False,
    ) -> None:
        clients_list = list(clients)
        if not clients_list:
            raise ValueError("CompositeAIClient requires at least one underlying client")

        self._clients = tuple(clients_list)
        self._is_bad_response = is_bad_response or (lambda _: False)
        self._telemetry = telemetry
        self._shuffle = shuffle

    async def generate_content(
        self,
        message: str,
        prompt: str | None = None,
        samples: list[tuple[str, str]] | None = None,
        enable_grounding: bool = False,
        response_schema=None,
        temperature: float | None = None,
        image_data: bytes | None = None,
        image_mime_type: str | None = None,
    ):
        last_error: Exception | None = None

        clients_to_try = list(self._clients)
        if self._shuffle:
            random.shuffle(clients_to_try)

        async with self._telemetry.async_create_span("composite_generate_content") as span:
            client_order = [client.__class__.__name__ for client in clients_to_try]
            span.set_attribute("client_order", ",".join(client_order))

            for client in clients_to_try:
                client_label = client.__class__.__name__
                try:
                    response = await client.generate_content(
                        message=message,
                        prompt=prompt,
                        samples=samples,
                        enable_grounding=enable_grounding,
                        response_schema=response_schema,
                        temperature=temperature,
                        image_data=image_data,
                        image_mime_type=image_mime_type,
                    )
                    if not self._is_bad_response(response):
                        return response
                    logger.warning(
                        "LLM client %s produced a response that triggered fallback.",
                        client_label,
                    )

                except Exception as exc:
                    last_error = exc
                    logger.warning("LLM client %s failed: %s", client_label, exc, exc_info=True)

            raise RuntimeError("All fallback clients failed") from last_error


__all__ = ["CompositeAIClient", "RetryAIClient"]
