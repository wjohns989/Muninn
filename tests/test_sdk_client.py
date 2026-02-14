"""Tests for Muninn Python SDK clients."""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

import httpx
import pytest
import requests

from muninn import AsyncMemory, Memory
from muninn.sdk import AsyncMuninnClient, MuninnClient
from muninn.sdk.errors import MuninnAPIError, MuninnConnectionError


def _requests_response(status_code: int, payload: Any, url: str = "http://localhost:42069") -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    if isinstance(payload, (dict, list)):
        raw = json.dumps(payload).encode("utf-8")
    else:
        raw = str(payload).encode("utf-8")
    response._content = raw
    response.url = url
    response.headers["Content-Type"] = "application/json"
    return response


class _StubSession:
    def __init__(self, mapping: Dict[Tuple[str, str], Any]):
        self.mapping = mapping
        self.calls = []
        self.headers: Dict[str, str] = {}
        self.closed = False

    def request(self, *, method: str, url: str, json: Any, params: Any, timeout: float):
        path = urlparse(url).path
        key = (method.upper(), path)
        self.calls.append(
            {
                "method": method.upper(),
                "path": path,
                "json": json,
                "params": params,
                "timeout": timeout,
            }
        )
        result = self.mapping[key]
        if isinstance(result, Exception):
            raise result
        return result

    def close(self) -> None:
        self.closed = True


def test_sync_add_success_and_payload_shape():
    stub = _StubSession(
        {
            ("POST", "/add"): _requests_response(
                200,
                {"success": True, "data": {"id": "mem-1", "content": "remember this"}},
            )
        }
    )
    client = MuninnClient(base_url="http://localhost:42069", session=stub, timeout=3.0)

    result = client.add(content="remember this", namespace="global", user_id="u1")

    assert result["id"] == "mem-1"
    assert stub.calls[0]["json"]["content"] == "remember this"
    assert stub.calls[0]["json"]["user_id"] == "u1"
    assert stub.calls[0]["timeout"] == 3.0


def test_sync_health_unwrapped_payload():
    stub = _StubSession(
        {
            ("GET", "/health"): _requests_response(
                200,
                {"status": "healthy", "memory_count": 3},
            )
        }
    )
    client = MuninnClient(base_url="http://localhost:42069", session=stub)

    result = client.health()

    assert result["status"] == "healthy"
    assert result["memory_count"] == 3


def test_sync_api_error_on_http_failure():
    stub = _StubSession(
        {
            ("POST", "/search"): _requests_response(
                500,
                {"detail": "internal failure"},
            )
        }
    )
    client = MuninnClient(base_url="http://localhost:42069", session=stub)

    with pytest.raises(MuninnAPIError, match="internal failure"):
        client.search("hello")


def test_sync_api_error_on_success_false():
    stub = _StubSession(
        {
            ("POST", "/search"): _requests_response(
                200,
                {"success": False, "detail": "request rejected"},
            )
        }
    )
    client = MuninnClient(base_url="http://localhost:42069", session=stub)

    with pytest.raises(MuninnAPIError, match="request rejected"):
        client.search("hello")


def test_sync_connection_error_wrapped():
    stub = _StubSession(
        {
            ("GET", "/health"): requests.ConnectionError("down"),
        }
    )
    client = MuninnClient(base_url="http://localhost:42069", session=stub)

    with pytest.raises(MuninnConnectionError, match="Failed to connect"):
        client.health()


@pytest.mark.asyncio
async def test_async_search_and_connection_error():
    async def ok_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search"
        body = json.loads(request.content.decode("utf-8"))
        assert body["query"] == "goal status"
        assert body["explain"] is True
        return httpx.Response(200, json={"success": True, "data": [{"id": "mem-a"}]})

    transport = httpx.MockTransport(ok_handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = AsyncMuninnClient(base_url="http://localhost:42069", http_client=http_client)
        result = await client.search("goal status", explain=True)
        assert result[0]["id"] == "mem-a"

    async def err_handler(request: httpx.Request):
        raise httpx.ConnectError("refused", request=request)

    err_transport = httpx.MockTransport(err_handler)
    async with httpx.AsyncClient(transport=err_transport) as http_client:
        client = AsyncMuninnClient(base_url="http://localhost:42069", http_client=http_client)
        with pytest.raises(MuninnConnectionError, match="Failed to connect"):
            await client.health()


def test_mem0_style_alias_exports():
    assert issubclass(Memory, MuninnClient)
    assert issubclass(AsyncMemory, AsyncMuninnClient)
