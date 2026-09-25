import base64
import copy
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from muninn.mcp.http import (
    _ACTIVE_HTTP_SESSIONS,
    _ACTIVE_HTTP_SESSIONS_LOCK,
    MCP_PROTOCOL_VERSION_HEADER,
    MCP_SESSION_ID_HEADER,
    _begin_dispatch,
    _finish_dispatch,
    close_all_http_sessions,
    http_transport_status,
    streamable_http_router,
)
from muninn.mcp.state import _SESSION_CONTEXTS, _SESSION_CONTEXTS_LOCK, _thread_local


@pytest.fixture(autouse=True)
def reset_http_transport_state(monkeypatch):
    monkeypatch.setenv("MUNINN_NO_AUTH", "1")
    monkeypatch.setenv("MUNINN_MCP_AUTOSTART_ON_LAUNCH", "0")
    with _ACTIVE_HTTP_SESSIONS_LOCK:
        _ACTIVE_HTTP_SESSIONS.clear()
    with _SESSION_CONTEXTS_LOCK:
        previous_contexts = copy.deepcopy(_SESSION_CONTEXTS)
        _SESSION_CONTEXTS.clear()
    if hasattr(_thread_local, "mcp_session_id"):
        delattr(_thread_local, "mcp_session_id")
    yield
    close_all_http_sessions()
    with _SESSION_CONTEXTS_LOCK:
        _SESSION_CONTEXTS.clear()
        _SESSION_CONTEXTS.update(previous_contexts)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(streamable_http_router)
    return TestClient(app, raise_server_exceptions=True)


def _json_headers(**extra):
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    headers.update(extra)
    return headers


def _initialize(client: TestClient, version: str = "2025-06-18"):
    response = client.post(
        "/mcp",
        headers=_json_headers(),
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": version,
                "capabilities": {},
                "clientInfo": {"name": "http-test", "version": "1.0.0"},
            },
        },
    )
    return response, response.headers.get("mcp-session-id")


def test_initialize_and_tools_list_use_one_bounded_session(client):
    response, session_id = _initialize(client)
    assert response.status_code == 200
    assert session_id
    assert http_transport_status()["active_sessions"] == 1

    tools = client.post(
        "/mcp",
        headers=_json_headers(
            **{
                MCP_SESSION_ID_HEADER: session_id,
                MCP_PROTOCOL_VERSION_HEADER: "2025-06-18",
            }
        ),
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    )
    assert tools.status_code == 200
    assert tools.json()["result"]["tools"]


def test_session_capacity_is_fail_closed(client, monkeypatch):
    monkeypatch.setenv("MUNINN_MCP_HTTP_MAX_SESSIONS", "2")
    assert _initialize(client)[0].status_code == 200
    assert _initialize(client)[0].status_code == 200

    rejected, session_id = _initialize(client)

    assert rejected.status_code == 503
    assert session_id is None
    assert http_transport_status()["active_sessions"] == 2


def test_expired_session_is_removed_with_its_context(client, monkeypatch):
    monkeypatch.setenv("MUNINN_MCP_HTTP_SESSION_TTL_SEC", "30")
    _, session_id = _initialize(client)
    with _ACTIVE_HTTP_SESSIONS_LOCK:
        _ACTIVE_HTTP_SESSIONS[session_id].last_seen_at = 1.0
    with _SESSION_CONTEXTS_LOCK:
        assert session_id in _SESSION_CONTEXTS

    monkeypatch.setattr("muninn.mcp.http.time.time", lambda: 100.0)
    assert _initialize(client)[0].status_code == 200

    with _ACTIVE_HTTP_SESSIONS_LOCK:
        assert session_id not in _ACTIVE_HTTP_SESSIONS
    with _SESSION_CONTEXTS_LOCK:
        assert session_id not in _SESSION_CONTEXTS


def test_batch_size_is_bounded(client, monkeypatch):
    monkeypatch.setenv("MUNINN_MCP_HTTP_MAX_BATCH_SIZE", "2")
    response = client.post(
        "/mcp",
        headers=_json_headers(),
        json=[
            {"jsonrpc": "2.0", "id": index, "method": "ping", "params": {}}
            for index in range(3)
        ],
    )
    assert response.status_code == 400
    assert "oversized" in response.json()["error"]["message"]


def test_request_body_size_is_bounded_before_json_decode(client, monkeypatch):
    monkeypatch.setenv("MUNINN_MCP_HTTP_MAX_REQUEST_BYTES", "1024")
    oversized = (
        b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"blob":"'
        + (b"x" * 2048)
        + b'"}}'
    )

    response = client.post("/mcp", headers=_json_headers(), content=oversized)

    assert response.status_code == 413
    assert "too large" in response.json()["error"]["message"].lower()


def test_dispatch_capacity_is_bounded_per_session_and_globally(client, monkeypatch):
    monkeypatch.setenv("MUNINN_MCP_HTTP_MAX_INFLIGHT_PER_SESSION", "1")
    monkeypatch.setenv("MUNINN_MCP_HTTP_MAX_INFLIGHT", "1")
    _, session_id = _initialize(client)

    lease = _begin_dispatch(session_id)
    assert lease is not None
    assert _begin_dispatch(session_id) is None
    assert http_transport_status()["active_dispatches"] == 1

    _finish_dispatch(lease)
    next_lease = _begin_dispatch(session_id)
    assert next_lease is not None
    _finish_dispatch(next_lease)
    assert http_transport_status()["active_dispatches"] == 0


def test_active_dispatch_is_not_expired_until_lease_finishes(client, monkeypatch):
    monkeypatch.setenv("MUNINN_MCP_HTTP_SESSION_TTL_SEC", "30")
    _, session_id = _initialize(client)
    lease = _begin_dispatch(session_id)
    with _ACTIVE_HTTP_SESSIONS_LOCK:
        _ACTIVE_HTTP_SESSIONS[session_id].last_seen_at = 1.0

    monkeypatch.setattr("muninn.mcp.http.time.time", lambda: 100.0)
    assert _initialize(client)[0].status_code == 200
    with _ACTIVE_HTTP_SESSIONS_LOCK:
        assert session_id in _ACTIVE_HTTP_SESSIONS

    _finish_dispatch(lease)
    assert _initialize(client)[0].status_code == 200
    with _ACTIVE_HTTP_SESSIONS_LOCK:
        assert session_id not in _ACTIVE_HTTP_SESSIONS


def test_delete_terminates_session_and_state(client):
    _, session_id = _initialize(client)
    response = client.delete("/mcp", headers={MCP_SESSION_ID_HEADER: session_id})
    assert response.status_code == 200
    assert http_transport_status()["active_sessions"] == 0


def test_configured_bearer_token_is_required(client, monkeypatch):
    monkeypatch.delenv("MUNINN_NO_AUTH", raising=False)
    monkeypatch.setenv("MUNINN_AUTH_TOKEN", "synthetic-http-test-token")

    assert _initialize(client)[0].status_code == 401
    authorized = client.post(
        "/mcp",
        headers=_json_headers(Authorization="Bearer synthetic-http-test-token"),
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}},
        },
    )
    assert authorized.status_code == 200


def test_shared_fastapi_server_exposes_streamable_http_route():
    import server

    # Probe behaviour rather than app.routes, whose shape varies across FastAPI
    # releases (0.141 no longer flattens included routers there).
    shared = TestClient(server.app, raise_server_exceptions=False)
    for method in ("GET", "POST", "DELETE"):
        response = shared.request(method, "/mcp", json={} if method == "POST" else None)
        assert response.status_code != 404, method
        # The handler itself answers GET with a JSON-RPC 405 (no SSE stream is
        # offered); a framework-level 405 would carry no JSON-RPC body.
        if response.status_code == 405:
            assert response.json().get("jsonrpc") == "2.0", method


@pytest.mark.parametrize(
    "version",
    ["2025-11-25", "2025-11-05", "2025-06-18", "2025-03-26", "2024-11-05"],
)
def test_all_supported_protocol_versions_initialize(client, version):
    response, session_id = _initialize(client, version)
    assert response.status_code == 200
    assert session_id
    assert response.json()["result"]["protocolVersion"] == version


def test_capabilities_are_version_conservative(client):
    older, _ = _initialize(client, "2025-06-18")
    latest, _ = _initialize(client, "2025-11-25")

    assert older.json()["result"]["capabilities"] == {"tools": {"listChanged": False}}
    assert "tasks" in latest.json()["result"]["capabilities"]


# --- MCP 2026-07-28 (stateless) -------------------------------------------

MODERN = "2026-07-28"


def _modern(client, method, params=None, *, msg_id=1, version=MODERN, headers=None):
    params = dict(params or {})
    params["_meta"] = {
        "io.modelcontextprotocol/protocolVersion": version,
        "io.modelcontextprotocol/clientInfo": {"name": "test-client", "version": "1.0"},
        "io.modelcontextprotocol/clientCapabilities": {},
    }
    request_headers = _json_headers(**{"MCP-Protocol-Version": version, "Mcp-Method": method})
    if method == "tools/call":
        request_headers["Mcp-Name"] = params.get("name", "")
    request_headers.update(headers or {})
    body = {"jsonrpc": "2.0", "method": method, "params": params}
    if msg_id is not None:
        body["id"] = msg_id
    return client.post("/mcp", json=body, headers=request_headers)


def test_modern_discover_needs_no_session(client):
    response = _modern(client, "server/discover")

    assert response.status_code == 200
    assert "Mcp-Session-Id" not in response.headers
    result = response.json()["result"]
    assert result["resultType"] == "complete"
    assert result["supportedVersions"][0] == MODERN and "2025-11-25" in result["supportedVersions"]
    assert result["_meta"]["io.modelcontextprotocol/serverInfo"]["name"] == "muninn-mcp"


def test_modern_tools_list_and_call_without_initialize(client, monkeypatch):
    from muninn.mcp import handlers

    listed = _modern(client, "tools/list")
    assert listed.status_code == 200
    assert any(tool["name"] == "search_memory" for tool in listed.json()["result"]["tools"])

    captured = {}
    backend = MagicMock()
    backend.json.return_value = {"success": True, "data": [{"id": "m1", "memory": "fact"}]}

    def fake_request(method, url, **kwargs):
        captured.update(kwargs["json"])
        return backend

    monkeypatch.setattr(handlers, "make_request_with_retry", fake_request)
    monkeypatch.setattr(handlers, "get_git_info", lambda: {"project": "p", "branch": "b"})

    called = _modern(client, "tools/call", {"name": "search_memory", "arguments": {"query": "q"}}, msg_id=2)
    assert called.status_code == 200 and called.json()["id"] == 2
    assert called.json()["result"]["resultType"] == "complete"
    assert captured["session_id"] is None  # no protocol session to key inhibition on

    _modern(client, "tools/call",
            {"name": "search_memory", "arguments": {"query": "q", "session_id": "conv-7"}}, msg_id=3)
    assert captured["session_id"] == "conv-7"
    with _SESSION_CONTEXTS_LOCK:
        assert not _SESSION_CONTEXTS  # per-request contexts are discarded


@pytest.mark.parametrize("override", [
    {"MCP-Protocol-Version": "2025-11-25"},
    {"Mcp-Method": "tools/call"},
])
def test_modern_header_mismatch_is_rejected(client, override):
    response = _modern(client, "tools/list", headers=override)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32020


def test_modern_tool_name_header_must_match_body(client):
    response = _modern(client, "tools/call", {"name": "search_memory", "arguments": {"query": "q"}},
                       headers={"Mcp-Name": "delete_all_memories"})
    assert response.status_code == 400 and response.json()["error"]["code"] == -32020

    encoded = "=?base64?" + base64.b64encode(b"search_memory").decode() + "?="
    ok = _modern(client, "tools/call", {"name": "search_memory", "arguments": {}},
                 headers={"Mcp-Name": encoded})
    assert ok.status_code == 200


def test_modern_unsupported_version_lists_supported(client):
    response = _modern(client, "tools/list", version="2099-01-01")
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == -32022
    assert error["data"]["requested"] == "2099-01-01" and MODERN in error["data"]["supported"]


def test_modern_unknown_method_is_404_and_notification_is_202(client):
    unknown = _modern(client, "roots/list")
    assert unknown.status_code == 404 and unknown.json()["error"]["code"] == -32601
    notification = _modern(client, "tools/list", msg_id=None)
    assert notification.status_code == 202


def test_legacy_initialize_still_works_alongside_modern(client):
    response, session_id = _initialize(client, "2025-11-25")
    assert response.status_code == 200 and session_id
    assert _modern(client, "server/discover").status_code == 200
