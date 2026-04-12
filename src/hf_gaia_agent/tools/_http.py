"""Centralized HTTP client, headers, retry policy, and shared text helpers."""

from __future__ import annotations

import time

import httpx

HTTP_HEADERS = {
    "User-Agent": "GAIABot/1.0 (contact@example.com) python-httpx/0.27.0",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

SEARCH_HEADERS = {
    **HTTP_HEADERS,
}

DEFAULT_TIMEOUT = 30.0
SEARCH_TIMEOUT = 20.0
AUDIO_TIMEOUT = 120.0
DEFAULT_RETRIES = 3
RETRY_STATUS_CODES = frozenset({429, 502, 503, 504})


class _RetryTransport(httpx.BaseTransport):
    """Small retry wrapper for transient network failures."""

    def __init__(
        self,
        wrapped: httpx.BaseTransport,
        *,
        retries: int = DEFAULT_RETRIES,
    ) -> None:
        self._wrapped = wrapped
        self._retries = retries

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(1, self._retries + 1):
            try:
                response = self._wrapped.handle_request(request)
                if (
                    response.status_code not in RETRY_STATUS_CODES
                    or attempt == self._retries
                ):
                    return response
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                if attempt == self._retries:
                    raise
            time.sleep(min(2 ** (attempt - 1), 4))

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("retry loop exhausted without a response")  # pragma: no cover


def make_client(
    *,
    timeout: float = DEFAULT_TIMEOUT,
    headers: dict[str, str] | None = None,
    follow_redirects: bool = True,
    retries: int = DEFAULT_RETRIES,
) -> httpx.Client:
    """Create an httpx.Client with consistent defaults."""
    transport: httpx.BaseTransport | None = None
    if retries > 0:
        transport = _RetryTransport(httpx.HTTPTransport(), retries=retries)
    return httpx.Client(
        timeout=timeout,
        follow_redirects=follow_redirects,
        headers=headers or HTTP_HEADERS,
        transport=transport,
    )


def truncate(value: str, *, max_chars: int = 80_000) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 15] + "\n...[truncated]"
