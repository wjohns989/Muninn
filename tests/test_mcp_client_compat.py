"""Cross-client MCP compatibility: tool coverage, hints, toolsets, ChatGPT search/fetch, errors, Origin."""

import copy
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from muninn.mcp import handlers
from muninn.mcp.definitions import TOOLS_SCHEMAS, TOOLSETS
from muninn.mcp.http import (
    _ACTIVE_HTTP_SESSIONS,
    _ACTIVE_HTTP_SESSIONS_LOCK,
    close_all_http_sessions,
    streamable_http_router,
)
from muninn.mcp.state import _SESSION_CONTEXTS, _SESSION_CONTEXTS_LOCK, _SESSION_STATE, _thread_local


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch):
    monkeypatch.setenv("MUNINN_NO_AUTH", "1")
    monkeypatch.setenv("MUNINN_MCP_AUTOSTART_ON_LAUNCH", "0")
    monkeypatch.setenv("MUNINN_MCP_AUTOSTART_SERVER", "0")
    monkeypatch.delenv("MUNINN_MCP_TOOLSET", raising=False)
    monkeypatch.setattr(handlers, "get_git_info", lambda: {"project": "p", "branch": "b"})
    with _ACTIVE_HTTP_SESSIONS_LOCK:
        _ACTIVE_HTTP_SESSIONS.clear()
    with _SESSION_CONTEXTS_LOCK:
        previous = copy.deepcopy(_SESSION_CONTEXTS)
        _SESSION_CONTEXTS.clear()
    if hasattr(_thread_local, "mcp_session_id"):
        delattr(_thread_local, "mcp_session_id")
    yield
    close_all_http_sessions()
    with _SESSION_CONTEXTS_LOCK:
        _SESSION_CONTEXTS.clear()
        _SESSION_CONTEXTS.update(previous)


def _backend(monkeypatch, body, calls=None):
    response = MagicMock()
    response.json.return_value = body

    def fake_request(method, url, **kwargs):
        if calls is not None:
            calls.append((method, url, kwargs))
        return response

    monkeypatch.setattr(handlers, "make_request_with_retry", fake_request)


def _listed(toolset=None):
    sent = []
    if toolset:
        _SESSION_STATE["toolset"] = toolset
    try:
        handlers.handle_list_tools(1, lambda _id, result: sent.append(result))
    finally:
        _SESSION_STATE.pop("toolset", None)
    return {tool["name"]: tool for tool in sent[0]["tools"]}


def _call(name, arguments):
    sent = []
    handlers.handle_call_tool(
        7,
        {"name": name, "arguments": arguments},
        lambda _id, code, message: sent.append({"error": {"code": code, "message": message}}),
        lambda _id, result: sent.append({"result": result}),
    )
    return sent[0]


# --- Every advertised tool must be callable ----------------------------------

@pytest.mark.parametrize("toolset", sorted(TOOLSETS))
def test_every_listed_tool_has_a_dispatcher(monkeypatch, toolset):
    calls = []
    _backend(monkeypatch, {"success": True, "data": {"id": "m1", "memory": "fact"}}, calls)
    _SESSION_STATE["toolset"] = toolset
    try:
        for name in TOOLSETS[toolset]:
            args = {"query": "q", "memory_id": "m1", "id": "m1", "correction": "c", "content": "c",
                    "instruction": "i", "goal_statement": "g", "confirm": True}
            try:
                assert handlers._do_call_tool_logic(name, args, None) is not None, name
            except (ValueError, KeyError, TypeError):
                pass  # reached the tool's own validation, so it is dispatched
    finally:
        _SESSION_STATE.pop("toolset", None)


@pytest.mark.parametrize("name, path", [
    ("set_project_goal", "/goal/set"),
    ("detect_information_gaps", "/reasoning/detect-gaps"),
    ("trigger_distillation", "/optimization/distill"),
    ("correct_fact", "/optimization/correct"),
    ("forage_knowledge", "/optimization/forage"),
])
def test_previously_unrouted_tools_reach_their_endpoint(monkeypatch, name, path):
    calls = []
    _backend(monkeypatch, {"success": True, "data": {"ok": True}}, calls)
    reply = _call(name, {"query": "q", "memory_id": "m1", "correction": "c", "goal_statement": "g"})
    assert "result" in reply and not reply["result"].get("isError")
    assert calls[0][1].endswith(path)


# --- Tool hints ----------------------------------------------------------------

def test_hints_describe_a_local_store():
    tools = _listed()
    assert all(tool["annotations"]["openWorldHint"] is False for name, tool in tools.items() if name != "mimir_relay")
    assert tools["mimir_relay"]["annotations"]["openWorldHint"] is True
    assert tools["add_memory"]["annotations"]["destructiveHint"] is False
    for name in ("delete_memory", "delete_all_memories", "update_memory", "correct_fact"):
        assert tools[name]["annotations"]["destructiveHint"] is True
    assert tools["search_memory"]["annotations"]["readOnlyHint"] is True
    assert tools["add_memory"]["title"] == "Add Memory"


def test_listing_does_not_mutate_shared_schemas():
    _listed()
    assert all("$schema" not in schema["inputSchema"] for schema in TOOLS_SCHEMAS)


# --- Toolsets ----------------------------------------------------------------

def test_default_toolset_is_full_and_backward_compatible():
    assert set(_listed()) == {schema["name"] for schema in TOOLS_SCHEMAS}


def test_core_toolset_fits_cursor_and_readonly_has_no_writes():
    core = _listed("core")
    assert "add_memory" in core and "search_memory" in core and len(core) <= 20
    readonly = _listed("readonly")
    assert readonly and all(tool["annotations"]["readOnlyHint"] for tool in readonly.values())


def test_env_selects_toolset_and_unknown_falls_back(monkeypatch):
    monkeypatch.setenv("MUNINN_MCP_TOOLSET", "chatgpt")
    assert set(_listed()) == {"search", "fetch"}
    monkeypatch.setenv("MUNINN_MCP_TOOLSET", "bogus")
    assert len(_listed()) == len(TOOLS_SCHEMAS)


def test_tools_outside_the_toolset_are_not_callable(monkeypatch):
    monkeypatch.setenv("MUNINN_MCP_TOOLSET", "readonly")
    _backend(monkeypatch, {"success": True, "data": {}})
    reply = _call("delete_memory", {"memory_id": "m1"})
    assert reply["error"]["code"] == -32601


# --- ChatGPT search/fetch ------------------------------------------------------

def test_chatgpt_search_returns_results_shape(monkeypatch):
    monkeypatch.setenv("MUNINN_MCP_TOOLSET", "chatgpt")
    long_text = "Deploy notes " + "x" * 200
    _backend(monkeypatch, {"success": True, "data": [
        {"id": "m1", "memory": long_text, "metadata": {"nested": {"deep": 1}}},
        {"id": "m2", "memory": "second\nline"},
    ]})
    reply = _call("search", {"query": "deploy"})["result"]
    results = reply["structuredContent"]["results"]
    assert [r["id"] for r in results] == ["m1", "m2"]
    assert results[0]["url"] == "muninn://memory/m1"
    assert len(results[0]["title"]) <= 80 and results[1]["title"] == "second line"
    assert handlers.json.loads(reply["content"][0]["text"]) == reply["structuredContent"]


def test_chatgpt_fetch_returns_full_text(monkeypatch):
    monkeypatch.setenv("MUNINN_MCP_TOOLSET", "chatgpt")
    text = "y" * 5000  # longer than the preview compactor keeps
    calls = []
    _backend(monkeypatch, {"success": True, "data": {"id": "a/b", "memory": text, "project": "p"}}, calls)
    reply = _call("fetch", {"id": "a/b"})["result"]
    assert calls[0][1].endswith("/memory/a%2Fb")
    assert reply["structuredContent"]["text"] == text
    assert reply["structuredContent"]["metadata"]["project"] == "p"


def test_chatgpt_tools_declare_output_schemas():
    tools = _listed("chatgpt")
    assert tools["search"]["outputSchema"]["required"] == ["results"]
    assert tools["fetch"]["annotations"]["readOnlyHint"] is True


# --- Errors the model can read ---------------------------------------------------

def test_backend_http_error_is_an_is_error_result(monkeypatch):
    monkeypatch.setenv("MUNINN_MCP_TOOLSET", "chatgpt")
    _backend(monkeypatch, {"detail": "Memory nope not found"})
    reply = _call("fetch", {"id": "nope"})["result"]
    assert reply["isError"] is True
    assert reply["content"][0]["text"] == "Error: Memory nope not found"


def test_backend_outage_is_an_is_error_result(monkeypatch):
    def boom(*_args, **_kwargs):
        raise ConnectionError("Muninn server unreachable")

    monkeypatch.setattr(handlers, "make_request_with_retry", boom)
    reply = _call("search_memory", {"query": "q"})["result"]
    assert reply["isError"] is True and "unreachable" in reply["content"][0]["text"]


def test_delete_all_without_confirm_is_an_is_error_result():
    reply = _call("delete_all_memories", {})["result"]
    assert reply["isError"] is True and "confirm" in reply["content"][0]["text"]


# --- Toolset per URL over HTTP ----------------------------------------------------

@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(streamable_http_router)
    return TestClient(app)


HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}


def test_http_session_toolset_comes_from_the_url(client):
    init = client.post("/mcp?toolset=chatgpt", headers=HEADERS, json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}},
    })
    session = init.headers["mcp-session-id"]
    listed = client.post("/mcp", headers={**HEADERS, "Mcp-Session-Id": session},
                         json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert {tool["name"] for tool in listed.json()["result"]["tools"]} == {"search", "fetch"}

    other = client.post("/mcp", headers=HEADERS, json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}},
    })
    other_listed = client.post("/mcp", headers={**HEADERS, "Mcp-Session-Id": other.headers["mcp-session-id"]},
                               json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert len(other_listed.json()["result"]["tools"]) == len(TOOLS_SCHEMAS)


def test_stateless_request_toolset_comes_from_the_url(client):
    version = "2026-07-28"
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"_meta": {
        "io.modelcontextprotocol/protocolVersion": version,
        "io.modelcontextprotocol/clientInfo": {"name": "t", "version": "1"},
        "io.modelcontextprotocol/clientCapabilities": {},
    }}}
    headers = {**HEADERS, "MCP-Protocol-Version": version, "Mcp-Method": "tools/list"}
    listed = client.post("/mcp?toolset=core", headers=headers, json=body)
    assert set(tool["name"] for tool in listed.json()["result"]["tools"]) == set(TOOLSETS["core"])


# --- Origin guard ------------------------------------------------------------------

def test_origin_rules(monkeypatch):
    from muninn.core.origin import configured_origins, origin_allowed

    assert origin_allowed(None)
    for origin in ("http://localhost:42069", "http://127.0.0.1:3000", "https://localhost", "http://[::1]:42069"):
        assert origin_allowed(origin), origin
    for origin in ("https://evil.example", "null", "http://localhost.evil.example", "http://127.0.0.1.nip.io"):
        assert not origin_allowed(origin), origin
    monkeypatch.setenv("MUNINN_ALLOWED_ORIGINS", "https://chatgpt.com, null")
    assert origin_allowed("https://chatgpt.com/", configured_origins())
    assert origin_allowed("null", configured_origins())
    monkeypatch.setenv("MUNINN_ALLOWED_ORIGINS", "*")
    assert origin_allowed("https://anything.example", configured_origins())


def test_server_rejects_foreign_browser_origins():
    import server

    client = TestClient(server.app)
    blocked = client.post("/search", json={"query": "q"}, headers={"Origin": "https://evil.example"})
    assert blocked.status_code == 403
    preflight = client.options("/mcp", headers={
        "Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert preflight.status_code == 403
    local = client.options("/mcp", headers={
        "Origin": "http://localhost:42069", "Access-Control-Request-Method": "POST"})
    assert local.status_code == 200
    assert local.headers["access-control-allow-origin"] == "http://localhost:42069"
    assert client.get("/health").status_code != 403
