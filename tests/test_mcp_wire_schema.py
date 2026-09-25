"""Validate Muninn's MCP responses against the official wire schemas (mcp-types).

The official SDK client rejects results that miss required fields, for example
2026-07-28 list results without cacheScope/ttlMs. These checks catch that
without a network or a running backend.
"""

import copy
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from muninn.mcp import handlers
from muninn.mcp.http import _ACTIVE_HTTP_SESSIONS, _ACTIVE_HTTP_SESSIONS_LOCK, close_all_http_sessions
from muninn.mcp.http import streamable_http_router
from muninn.mcp.state import _SESSION_CONTEXTS, _SESSION_CONTEXTS_LOCK, _thread_local

methods = pytest.importorskip("mcp_types.methods")

HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}
LEGACY_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")
MODERN = "2026-07-28"


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setenv("MUNINN_NO_AUTH", "1")
    monkeypatch.setenv("MUNINN_MCP_AUTOSTART_ON_LAUNCH", "0")
    monkeypatch.setenv("MUNINN_MCP_AUTOSTART_SERVER", "0")
    monkeypatch.setattr(handlers, "get_git_info", lambda: {"project": "p", "branch": "b"})
    backend = MagicMock()
    backend.json.return_value = {"success": True, "data": [{"id": "m1", "memory": "a fact"}]}
    monkeypatch.setattr(handlers, "make_request_with_retry", lambda *a, **k: backend)
    with _ACTIVE_HTTP_SESSIONS_LOCK:
        _ACTIVE_HTTP_SESSIONS.clear()
    with _SESSION_CONTEXTS_LOCK:
        previous = copy.deepcopy(_SESSION_CONTEXTS)
        _SESSION_CONTEXTS.clear()
    if hasattr(_thread_local, "mcp_session_id"):
        delattr(_thread_local, "mcp_session_id")
    yield backend
    close_all_http_sessions()
    with _SESSION_CONTEXTS_LOCK:
        _SESSION_CONTEXTS.clear()
        _SESSION_CONTEXTS.update(previous)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(streamable_http_router)
    return TestClient(app)


def _checked(method, version, response):
    body = response.json()
    assert "result" in body, body
    methods.validate_server_result(method, version, body["result"])
    return body["result"]


LEGACY_CALLS = [
    ("tools/list", {}),
    ("tools/call", {"name": "search_memory", "arguments": {"query": "q"}}),
    ("tools/call", {"name": "delete_all_memories", "arguments": {}}),  # isError result
    ("resources/list", {}),
    ("resources/templates/list", {}),
    ("prompts/list", {}),
    ("ping", {}),
]


@pytest.mark.parametrize("version", LEGACY_VERSIONS)
@pytest.mark.parametrize("toolset", ["full", "chatgpt"])
def test_handshake_versions_match_the_official_schema(client, version, toolset):
    init = client.post(f"/mcp?toolset={toolset}", headers=HEADERS, json={
        "jsonrpc": "2.0", "id": 0, "method": "initialize",
        "params": {"protocolVersion": version, "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}},
    })
    _checked("initialize", version, init)
    session = {**HEADERS, "Mcp-Session-Id": init.headers["mcp-session-id"], "MCP-Protocol-Version": version}
    calls = LEGACY_CALLS if toolset == "full" else [
        ("tools/list", {}), ("tools/call", {"name": "search", "arguments": {"query": "q"}})]
    for msg_id, (method, params) in enumerate(calls, start=1):
        response = client.post("/mcp", headers=session,
                               json={"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})
        _checked(method, version, response)


MODERN_CALLS = [
    ("server/discover", {}),
    ("tools/list", {}),
    ("tools/call", {"name": "search_memory", "arguments": {"query": "q"}}),
    ("tools/call", {"name": "delete_all_memories", "arguments": {}}),
    ("resources/list", {}),
    ("resources/templates/list", {}),
    ("prompts/list", {}),
]


@pytest.mark.parametrize("method, params", MODERN_CALLS)
def test_stateless_version_matches_the_official_schema(client, method, params):
    params = {**params, "_meta": {
        "io.modelcontextprotocol/protocolVersion": MODERN,
        "io.modelcontextprotocol/clientInfo": {"name": "t", "version": "1"},
        "io.modelcontextprotocol/clientCapabilities": {},
    }}
    headers = {**HEADERS, "MCP-Protocol-Version": MODERN, "Mcp-Method": method}
    if method == "tools/call":
        headers["Mcp-Name"] = params["name"]
    response = client.post("/mcp", headers=headers,
                           json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    result = _checked(method, MODERN, response)
    if method.endswith("/list") or method == "server/discover":
        assert result["cacheScope"] == "private" and result["ttlMs"] >= 0


def test_chatgpt_structured_results_match_the_official_schema(client, isolated):
    isolated.json.return_value = {"success": True, "data": {"id": "m1", "memory": "a fact", "project": "p"}}
    init = client.post("/mcp?toolset=chatgpt", headers=HEADERS, json={
        "jsonrpc": "2.0", "id": 0, "method": "initialize",
        "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}},
    })
    session = {**HEADERS, "Mcp-Session-Id": init.headers["mcp-session-id"]}
    fetched = client.post("/mcp", headers=session, json={
        "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "fetch", "arguments": {"id": "m1"}}})
    result = _checked("tools/call", "2025-11-25", fetched)
    assert result["structuredContent"]["text"] == "a fact"
