"""Authenticated API/MCP contract for agent-visible redacted transcript spans."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

import server
from muninn.history.blind_index import SecureHistoryBlindIndex
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.service import HistoryService
from muninn.mcp.definitions import TOOLSETS


@pytest.mark.asyncio
async def test_history_search_fetch_requires_auth_and_never_returns_secret(tmp_path, monkeypatch):
    main_token = "test-main-auth-token-aaaaaaaaaaaaaaaaaaaaaaaa"
    monkeypatch.setenv("MUNINN_AUTH_TOKEN", main_token)
    monkeypatch.setenv("MUNINN_API_KEY", "different-api-key-bbbbbbbbbbbbbbbbbbbbbbbb")
    monkeypatch.setenv("MUNINN_NO_AUTH", "0")
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(tmp_path / "archive"))
    monkeypatch.setattr(server, "is_security_enabled", lambda: True)
    source = tmp_path / "chat.jsonl"
    source.write_text(
        "discussion of parser regression\nOPENROUTER_API_KEY=CANARY_ONLY\n",
        encoding="utf-8",
    )
    archive = SecureHistoryArchive.create(tmp_path / "archive", "synthetic archive recovery passphrase")
    archive.archive_file(source, "codex")
    SecureHistoryBlindIndex(archive).build()
    service = HistoryService(None, tmp_path / "unused-vault", home=tmp_path,
                             archive_passphrase="synthetic archive recovery passphrase")
    monkeypatch.setattr(server, "_require_history", lambda: service)
    transport = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 1234))
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        route = "/history/secure/search"
        assert (await client.post(route, json={"query": "parser"})).status_code == 401
        headers = {"Authorization": f"Bearer {main_token}"}
        api_headers = {"Authorization": "Bearer different-api-key-bbbbbbbbbbbbbbbbbbbbbbbb"}
        assert (await client.post(route, json={"query": "parser"}, headers=api_headers)).status_code == 401
        found = await client.post(route, json={"query": "parser"}, headers=headers)
        assert found.status_code == 200
        assert found.headers["cache-control"] == "no-store"
        capability = found.json()["data"]["matches"][0]["fetch_capability"]
        assert "CANARY_ONLY" not in found.text
        fetch = await client.post(
            "/history/secure/fetch", json={"capability": capability}, headers=headers,
        )
        assert fetch.status_code == 200
        assert "parser regression" in fetch.json()["data"]["redacted_text"]
        assert "CANARY_ONLY" not in fetch.text
        assert fetch.headers["cache-control"] == "no-store"
        assert (await client.post("/history/secure/fetch", json={"capability": capability})).status_code == 401
        assert (await client.post("/history/secure/fetch", json={"capability": "bad"},
                                  headers=headers)).status_code == 400
    remote_transport = httpx.ASGITransport(app=server.app, client=("192.168.1.2", 1234))
    async with httpx.AsyncClient(transport=remote_transport, base_url="http://localhost") as remote:
        assert (await remote.post(route, json={"query": "parser"}, headers=headers)).status_code == 404
    monkeypatch.setattr(server, "is_security_enabled", lambda: False)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as disabled:
        assert (await disabled.post(route, json={"query": "parser"}, headers=headers)).status_code == 404


def test_core_mcp_toolset_exposes_search_and_fetch_without_reveal():
    assert "search_secure_history" in TOOLSETS["core"]
    assert "start_secure_history_search" in TOOLSETS["core"]
    assert "poll_secure_history_search" in TOOLSETS["core"]
    assert "poll_secure_history_analysis" in TOOLSETS["core"]
    assert "fetch_secure_history" in TOOLSETS["core"]
    assert "start_secure_history_transcript" in TOOLSETS["core"]
    assert "poll_secure_history_transcript" in TOOLSETS["full"]
    assert "read_secure_history_transcript_page" in TOOLSETS["core"]
    assert "analyze_secure_history" in TOOLSETS["core"]
    assert not any("reveal" in name for name in TOOLSETS["core"])


@pytest.mark.parametrize(("name", "arguments", "data", "marker"), [
    ("search_secure_history", {"query": "parser"},
     {"matches": [{"fetch_capability": "SAFE_CAPABILITY_MARKER"}]}, "SAFE_CAPABILITY_MARKER"),
    ("start_secure_history_search", {"query": "parser"},
     {"job_id": "SAFE_JOB_MARKER", "state": "pending"}, "SAFE_JOB_MARKER"),
    ("poll_secure_history_search", {"job_id": "SAFE_JOB_MARKER"},
     {"state": "succeeded", "result": {"matches": [{"fetch_capability": "SAFE_CAPABILITY_MARKER"}]}},
     "SAFE_CAPABILITY_MARKER"),
    ("cancel_secure_history_search", {"job_id": "SAFE_JOB_MARKER"},
     {"state": "cancelled", "job_id": "SAFE_JOB_MARKER"}, "SAFE_JOB_MARKER"),
    ("poll_secure_history_analysis", {"job_id": "SAFE_ANALYSIS_JOB_MARKER"},
     {"state": "succeeded", "provisional": True,
      "result": {"analysis": {"summary": "SAFE_ANALYSIS_MARKER"}}}, "SAFE_ANALYSIS_MARKER"),
    ("cancel_secure_history_analysis", {"job_id": "SAFE_ANALYSIS_JOB_MARKER"},
     {"state": "cancelled", "job_id": "SAFE_ANALYSIS_JOB_MARKER"}, "SAFE_ANALYSIS_JOB_MARKER"),
    ("fetch_secure_history", {"capability": "SAFE_CAPABILITY_MARKER"},
     {"redacted_text": "SAFE_TRANSCRIPT_MARKER"}, "SAFE_TRANSCRIPT_MARKER"),
    ("start_secure_history_transcript", {"capability": "SAFE_CAPABILITY_MARKER"},
     {"state": "pending"}, "pending"),
    ("poll_secure_history_transcript", {"capability": "SAFE_CAPABILITY_MARKER"},
     {"state": "ready", "cursor": "SAFE_CURSOR_MARKER"}, "SAFE_CURSOR_MARKER"),
    ("read_secure_history_transcript_page", {"cursor": "SAFE_CURSOR_MARKER"},
     {"redacted_text": "SAFE_PAGE_MARKER"}, "SAFE_PAGE_MARKER"),
    ("analyze_secure_history", {"capability": "SAFE_CAPABILITY_MARKER"},
     {"status": "ok", "analysis": {"summary": "SAFE_SUMMARY_MARKER"}}, "SAFE_SUMMARY_MARKER"),
    ("search_credential_metadata", {"query": "openrouter"},
     [{"source_hint": "config/.env.local"}], "config/.env.local"),
])
def test_private_mcp_result_preserves_required_fields(monkeypatch, name, arguments, data, marker):
    from muninn.mcp import handlers

    monkeypatch.setenv("MUNINN_MCP_AUTOSTART_SERVER", "0")
    monkeypatch.setattr(handlers, "active_toolset",
                        lambda: "full" if name in {
                            "poll_secure_history_transcript", "cancel_secure_history_search",
                            "cancel_secure_history_analysis"} else "core")
    monkeypatch.setattr(
        handlers, "make_request_with_retry",
        lambda *_args, **_kwargs: SimpleNamespace(json=lambda: {"success": True, "data": data}),
    )
    received = []
    handlers.handle_call_tool(
        1, {"name": name, "arguments": arguments},
        lambda *_args: pytest.fail("Private tool unexpectedly returned an RPC error"),
        lambda _mid, result: received.append(result),
    )
    assert len(received) == 1
    assert received[0].get("isError") is not True
    assert marker in received[0]["content"][0]["text"]


@pytest.mark.asyncio
async def test_secure_analysis_endpoint_is_local_and_auth_only(monkeypatch):
    from muninn.history import secure_analysis

    token = "test-main-auth-token-aaaaaaaaaaaaaaaaaaaaaaaa"
    monkeypatch.setenv("MUNINN_AUTH_TOKEN", token)
    monkeypatch.setenv("MUNINN_NO_AUTH", "0")
    monkeypatch.setattr(server, "is_security_enabled", lambda: True)
    monkeypatch.setattr(server, "_require_history", lambda: object())
    server._secure_history_analyze_times.clear()
    seen = {}

    async def fake_analyze(_history, capability, *, allow_remote, prefer_remote):
        seen.update({"capability": capability, "allow_remote": allow_remote,
                     "prefer_remote": prefer_remote})
        return {"status": "ok", "provider": "ollama", "analysis": {"summary": "safe"}}

    monkeypatch.setattr(secure_analysis, "analyze_secure_hit", fake_analyze)
    path = "/history/secure/analyze"
    local = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 1234))
    async with httpx.AsyncClient(transport=local, base_url="http://localhost") as client:
        assert (await client.post(path, json={"capability": "opaque"})).status_code == 401
        response = await client.post(path, json={"capability": "opaque"},
                                     headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert seen == {"capability": "opaque", "allow_remote": False,
                        "prefer_remote": False}
    remote = httpx.ASGITransport(app=server.app, client=("192.168.1.2", 1234))
    async with httpx.AsyncClient(transport=remote, base_url="http://localhost") as client:
        assert (await client.post(path, json={"capability": "opaque"},
                                  headers={"Authorization": f"Bearer {token}"})).status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", [
    "search_credential_metadata", "search_secure_history", "start_secure_history_search",
    "poll_secure_history_search", "cancel_secure_history_search",
    "poll_secure_history_analysis", "cancel_secure_history_analysis",
    "fetch_secure_history", "analyze_secure_history",
    "start_secure_history_transcript", "poll_secure_history_transcript",
    "read_secure_history_transcript_page",
])
async def test_mcp_private_tools_reject_generic_api_key(tmp_path, monkeypatch, tool_name):
    monkeypatch.setenv("MUNINN_AUTH_TOKEN", "main-token-aaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    monkeypatch.setenv("MUNINN_API_KEY", "api-key-bbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    monkeypatch.setenv("MUNINN_NO_AUTH", "0")
    headers = {
        "Authorization": "Bearer api-key-bbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "Accept": "application/json", "Content-Type": "application/json",
        "MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "tools/call",
        "Mcp-Name": tool_name,
    }
    body = {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {
            "name": tool_name, "arguments": (
                {"query": "openrouter"} if tool_name == "search_credential_metadata" else {"job_id": "a" * 32}
            ),
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                "io.modelcontextprotocol/clientInfo": {"name": "test", "version": "1"},
                "io.modelcontextprotocol/clientCapabilities": {},
            },
        },
    }
    transport = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 1234))
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        denied = await client.post("/mcp", json=body, headers=headers)
        assert denied.status_code == 401

    from muninn.mcp import sse as mcp_sse

    session_id = mcp_sse._create_session()
    assert session_id
    try:
        sse_transport = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 1234))
        async with httpx.AsyncClient(transport=sse_transport, base_url="http://localhost") as client:
            denied_sse = await client.post(
                "/mcp/messages", params={"session_id": session_id}, json=body,
                headers={"Authorization": headers["Authorization"]},
            )
            assert denied_sse.status_code == 401
    finally:
        await mcp_sse._close_session(session_id)


@pytest.mark.asyncio
async def test_transcript_page_routes_require_main_local_token(monkeypatch):
    token = "test-main-auth-token-aaaaaaaaaaaaaaaaaaaaaaaa"
    monkeypatch.setenv("MUNINN_AUTH_TOKEN", token)
    monkeypatch.setenv("MUNINN_API_KEY", "test-generic-api-key-bbbbbbbbbbbbbbbbbbbb")
    monkeypatch.setenv("MUNINN_NO_AUTH", "0")
    monkeypatch.setattr(server, "is_security_enabled", lambda: True)
    fake = SimpleNamespace(
        secure_projection_start=lambda cap: {"state": "pending"},
        secure_projection_poll=lambda cap: {"state": "ready", "cursor": "opaque-cursor"},
        secure_projection_page=lambda cursor: {"redacted_text": "safe continuation", "next_cursor": None},
    )
    monkeypatch.setattr(server, "_require_history", lambda: fake)
    server._secure_history_page_times.clear()
    local = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 1234))
    async with httpx.AsyncClient(transport=local, base_url="http://localhost") as client:
        start = "/history/secure/transcript/start"
        assert (await client.post(start, json={"capability": "opaque"})).status_code == 401
        generic = {"Authorization": "Bearer test-generic-api-key-bbbbbbbbbbbbbbbbbbbb"}
        assert (await client.post(start, json={"capability": "opaque"}, headers=generic)).status_code == 401
        main = {"Authorization": f"Bearer {token}"}
        queued = await client.post(start, json={"capability": "opaque"}, headers=main)
        assert queued.status_code == 202 and queued.headers["cache-control"] == "no-store"
        polled = await client.post("/history/secure/transcript/poll",
                                   json={"capability": "opaque"}, headers=main)
        assert polled.json()["data"]["cursor"] == "opaque-cursor"
        page = await client.post("/history/secure/transcript/page",
                                 json={"cursor": "opaque-cursor"}, headers=main)
        assert page.json()["data"]["redacted_text"] == "safe continuation"
        assert page.headers["cache-control"] == "no-store"
    remote = httpx.ASGITransport(app=server.app, client=("192.168.1.2", 1234))
    async with httpx.AsyncClient(transport=remote, base_url="http://localhost") as client:
        assert (await client.post(start, json={"capability": "opaque"}, headers=main)).status_code == 404


@pytest.mark.asyncio
async def test_real_archive_search_to_http_transcript_continuation(tmp_path, monkeypatch):
    import json

    token = "test-main-auth-token-aaaaaaaaaaaaaaaaaaaaaaaa"
    monkeypatch.setenv("MUNINN_AUTH_TOKEN", token)
    monkeypatch.setenv("MUNINN_NO_AUTH", "0")
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(tmp_path / "archive"))
    monkeypatch.setattr(server, "is_security_enabled", lambda: True)
    source = tmp_path / "conversation.jsonl"
    rows = [
        {"type": "event_msg", "payload": {"type": "user_message",
                                           "message": "orbital-widget " * 700 + " api_key=secretvalue"}},
        {"type": "response_item", "payload": {"type": "function_call_output",
                                               "output": "TOOL_CANARY"}},
        {"type": "event_msg", "payload": {"type": "agent_message",
                                           "message": "Final answer."}},
    ]
    source.write_bytes(b"\n".join(json.dumps(row).encode() for row in rows) + b"\n")
    archive = SecureHistoryArchive.create(tmp_path / "archive", "synthetic archive recovery passphrase")
    archive.archive_file(source, "codex")
    SecureHistoryBlindIndex(archive).build()
    service = HistoryService(None, tmp_path / "unused-vault", home=tmp_path,
                             archive_passphrase="synthetic archive recovery passphrase")
    monkeypatch.setattr(server, "_require_history", lambda: service)
    server._secure_history_page_times.clear()
    transport = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 1234))
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
            found = await client.post("/history/secure/search", json={"query": "orbital-widget"},
                                      headers=headers)
            assert found.status_code == 200
            capability = found.json()["data"]["matches"][0]["fetch_capability"]
            status = await client.post("/history/secure/transcript/start",
                                       json={"capability": capability}, headers=headers)
            for _ in range(100):
                if status.json()["data"]["state"] != "pending":
                    break
                await asyncio.sleep(.01)
                status = await client.post("/history/secure/transcript/poll",
                                           json={"capability": capability}, headers=headers)
            assert status.json()["data"]["state"] == "ready"
            assert status.json()["data"]["coverage"]["omitted_units"] == 1
            cursor = status.json()["data"]["cursor"]
            pages = []
            while cursor:
                response = await client.post("/history/secure/transcript/page",
                                             json={"cursor": cursor}, headers=headers)
                assert response.status_code == 200
                assert response.headers["cache-control"] == "no-store"
                page = response.json()["data"]
                assert len(page["redacted_text"]) <= 4000
                pages.append(page["redacted_text"])
                cursor = page["next_cursor"]
            transcript = "".join(pages)
            assert transcript.count("orbital-widget") == 700
            assert "Final answer." in transcript
            assert "secretvalue" not in transcript and "TOOL_CANARY" not in transcript
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_opt_in_cpu_worker_indexes_new_encrypted_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    monkeypatch.setenv("MUNINN_HISTORY_INDEX_AUTO", "1")
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(tmp_path / "archive"))
    source = tmp_path / "chat.jsonl"
    source.write_text("new memory about an orbital-widget parser", encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", "synthetic archive recovery passphrase")
    archive.archive_file(source, "codex")
    service = HistoryService(None, tmp_path / "unused-vault", home=tmp_path,
                             archive_passphrase="synthetic archive recovery passphrase")
    await service.start()
    try:
        async def wait_for_index():
            while not service.last_secure_index:
                await asyncio.sleep(0.01)
        await asyncio.wait_for(wait_for_index(), timeout=5)
        assert service.last_secure_index["ready"] == 1
        assert service.secure_search("orbital-widget")["matches"]
    finally:
        await service.stop()
