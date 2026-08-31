"""Polite, retrying HTTP access to the FPL API.

The API is undocumented, unofficial and free, so this client is deliberately
conservative: it identifies itself, caps how many requests are in flight, and
backs off when the server is unhappy.

Retry policy, and the reasoning behind it:

* Network errors and 5xx are transient, so they are retried.
* 429 is retried, honouring ``Retry-After`` when the server sends one.
* Every other 4xx fails immediately. A 404 will not fix itself; retrying it
  just adds load and delays the error you needed to see.

Backoff uses *full jitter* -- sleep a random duration in ``[0, delay]`` rather
than exactly ``delay``. Without jitter, requests that fail together retry
together, and you get synchronised waves of load instead of a smooth spread.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from types import TracebackType

import httpx

from . import config


class FetchError(RuntimeError):
    """A URL could not be fetched, either definitively or after retrying."""

    def __init__(self, url: str, message: str, status: int | None = None) -> None:
        super().__init__(f"{url}: {message}")
        self.url = url
        self.status = status


@dataclass(frozen=True)
class Response:
    """A successful response, kept as raw bytes.

    Deliberately *not* parsed JSON: the archive stores exactly what the server
    sent, and re-serialising a parsed object would not be that.
    """

    url: str
    status: int
    body: bytes


def is_retryable(status: int) -> bool:
    """Is this status worth trying again?"""
    return status == 429 or 500 <= status < 600


def retry_after_seconds(response: httpx.Response) -> float | None:
    """Read ``Retry-After``, when present as a plain number of seconds.

    The header may also hold an HTTP date; we ignore that form and fall back to
    ordinary backoff rather than adding date parsing for a rare case.
    """
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw.strip()))
    except ValueError:
        return None


class FplClient:
    """An async HTTP client that throttles itself and retries sensibly."""

    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
        max_concurrency: int = config.MAX_CONCURRENCY,
        max_retries: int = config.MAX_RETRIES,
        timeout: float = config.REQUEST_TIMEOUT,
        backoff_base: float = config.BACKOFF_BASE,
        backoff_cap: float = config.BACKOFF_CAP,
    ) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._client = httpx.AsyncClient(
            headers={"User-Agent": config.USER_AGENT, "Accept": "application/json"},
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )

    async def __aenter__(self) -> FplClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _backoff(self, attempt: int) -> float:
        ceiling = min(self._backoff_cap, self._backoff_base * (2**attempt))
        return random.uniform(0.0, ceiling)

    async def fetch(
        self, url: str, *, headers: dict[str, str] | None = None
    ) -> Response:
        """GET ``url``, retrying transient failures. Raises `FetchError`."""
        last_error = "unknown error"
        last_status: int | None = None

        for attempt in range(self._max_retries + 1):
            wait: float | None = None

            # Hold the semaphore only for the request itself, never while
            # sleeping -- otherwise one backing-off request would idle a slot
            # that another request could be using.
            async with self._semaphore:
                try:
                    response = await self._client.get(url, headers=headers)
                except httpx.TransportError as exc:
                    last_status = None
                    last_error = f"transport error: {exc!r}"
                else:
                    if response.status_code < 400:
                        return Response(
                            url=url,
                            status=response.status_code,
                            body=response.content,
                        )

                    last_status = response.status_code
                    last_error = f"HTTP {response.status_code}"
                    if not is_retryable(response.status_code):
                        raise FetchError(url, last_error, last_status)
                    wait = retry_after_seconds(response)

            if attempt == self._max_retries:
                break
            await asyncio.sleep(self._backoff(attempt) if wait is None else wait)

        attempts = self._max_retries + 1
        raise FetchError(
            url,
            f"gave up after {attempts} attempt(s) ({last_error})",
            last_status,
        )
