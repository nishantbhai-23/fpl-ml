"""Retry policy tests.

``backoff_base=0`` throughout, so full-jitter backoff sleeps for zero seconds
and the suite stays fast while still exercising the real retry loop.
"""

from __future__ import annotations

import httpx
import pytest

from fpl_ml.client import FetchError, FplClient

URL = "https://example.test/thing/"


def client_returning(*statuses: int, **kwargs) -> tuple[FplClient, list[int]]:
    """A client whose transport yields ``statuses`` in order, then repeats the last."""
    calls: list[int] = []
    remaining = list(statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        status = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        calls.append(status)
        return httpx.Response(status, content=b'{"ok":true}')

    transport = httpx.MockTransport(handler)
    return FplClient(transport=transport, backoff_base=0.0, **kwargs), calls


async def test_returns_body_on_success():
    client, calls = client_returning(200)
    async with client:
        response = await client.fetch(URL)

    assert response.status == 200
    assert response.body == b'{"ok":true}'
    assert calls == [200]


async def test_retries_transient_failure_then_succeeds():
    client, calls = client_returning(503, 500, 200)
    async with client:
        response = await client.fetch(URL)

    assert response.status == 200
    assert calls == [503, 500, 200]


async def test_retries_429_honouring_retry_after():
    attempts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.path)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, content=b"{}")

    async with FplClient(
        transport=httpx.MockTransport(handler), backoff_base=0.0
    ) as client:
        response = await client.fetch(URL)

    assert response.status == 200
    assert len(attempts) == 2


async def test_does_not_retry_client_error():
    client, calls = client_returning(404)
    async with client:
        with pytest.raises(FetchError) as caught:
            await client.fetch(URL)

    # The point of the test: exactly one attempt. A 404 will not fix itself.
    assert calls == [404]
    assert caught.value.status == 404


async def test_gives_up_after_max_retries():
    client, calls = client_returning(500, max_retries=2)
    async with client:
        with pytest.raises(FetchError) as caught:
            await client.fetch(URL)

    assert len(calls) == 3  # the initial attempt plus two retries
    assert caught.value.status == 500


async def test_retries_transport_errors():
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 3:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, content=b"{}")

    async with FplClient(
        transport=httpx.MockTransport(handler), backoff_base=0.0
    ) as client:
        response = await client.fetch(URL)

    assert response.status == 200
    assert len(attempts) == 3
