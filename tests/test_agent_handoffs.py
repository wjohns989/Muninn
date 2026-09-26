"""Shared-store handoffs between agents, session briefings, agent identity and built-in prompting."""

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from muninn.core import handoffs
from muninn.core.types import MemoryRecord
from muninn.mcp import handlers
from muninn.mcp.prompts import get_prompt, list_prompts, protocol_for
from muninn.mcp.state import _SESSION_CONTEXTS, _SESSION_CONTEXTS_LOCK, _SESSION_STATE, _thread_local
from muninn.store.sqlite_metadata import SQLiteMetadataStore


@pytest.fixture
def store(tmp_path):
    return SQLiteMetadataStore(tmp_path / "metadata.db")


@pytest.fixture
def memory(store):
    return SimpleNamespace(_metadata=store)


def _remember(store, content, *, project="Muninn", agent="codex", scope="project", category=None, created_at=None):
    metadata = {"user_id": "global_user"}
    if category:
        metadata["category"] = category
    record = MemoryRecord(content=content, project=project, source_agent=agent, scope=scope, metadata=metadata)
    if created_at is not None:
        record.created_at = created_at
    store.add(record)
    return record


# --- Handoff lifecycle ----------------------------------------------------------

@pytest.mark.asyncio
async def test_handoff_moves_from_one_agent_to_another(memory):
    created = await handoffs.create_handoff(
        memory, project="Muninn", from_agent="claude-desktop",
        summary="Designed the handoff API.\nTests pending.",
        details={"next_steps": ["write tests", " "], "files": "muninn/core/handoffs.py", "branch": "feat/x"},
    )
    assert created["status"] == "open" and created["title"] == "Designed the handoff API. Tests pending."
    assert created["details"] == {
        "next_steps": ["write tests"], "files": ["muninn/core/handoffs.py"], "branch": "feat/x"}

    resumed = (await handoffs.resume_handoff(memory, agent="codex", project="Muninn"))["handoff"]
    assert resumed["id"] == created["id"]
    assert resumed["status"] == "claimed" and resumed["claimed_by"] == "codex"

    assert (await handoffs.resume_handoff(memory, agent="claude-code", project="Muninn"))["handoff"] is None

    done = await handoffs.finish_handoff(memory, handoff_id=created["id"], agent="codex", note="shipped")
    assert done["status"] == "done" and done["note"] == "shipped"


@pytest.mark.asyncio
async def test_resume_respects_recipient_and_reports_takeover(memory):
    for_codex = await handoffs.create_handoff(
        memory, project="p", from_agent="claude-code", summary="for codex", to_agent="codex")
    await handoffs.create_handoff(memory, project="other", from_agent="claude-code", summary="elsewhere")

    general = await handoffs.create_handoff(memory, project="p", from_agent="claude-code", summary="for anyone")

    # Addressed handoffs go to their recipient first; others take unaddressed work first.
    first = (await handoffs.resume_handoff(memory, agent="codex", project="p"))["handoff"]
    assert first["id"] == for_codex["id"] and "note_to_agent" not in first
    await handoffs.finish_handoff(memory, handoff_id=first["id"], agent="codex", status="open")
    other = (await handoffs.resume_handoff(memory, agent="gemini-cli", project="p"))["handoff"]
    assert other["id"] == general["id"]
    # With only someone else's handoff left, it is still reachable and says who it was for.
    leftover = (await handoffs.resume_handoff(memory, agent="gemini-cli", project="p"))["handoff"]
    assert leftover["id"] == for_codex["id"] and leftover["note_to_agent"] == "This handoff was addressed to codex."
    await handoffs.finish_handoff(memory, handoff_id=first["id"], agent="gemini-cli", status="open")
    first = (await handoffs.resume_handoff(memory, agent="codex", project="p"))["handoff"]

    # A crashed agent's claim can be taken over explicitly, and the takeover is visible.
    taken = (await handoffs.resume_handoff(memory, agent="claude-code", handoff_id=first["id"]))["handoff"]
    assert taken["claimed_by"] == "claude-code" and taken["previously_claimed_by"] == "codex"
    assert taken["note_to_agent"] == "This handoff was addressed to codex."


@pytest.mark.asyncio
async def test_peek_does_not_claim_and_validation_errors(memory):
    created = await handoffs.create_handoff(memory, project="p", from_agent="a", summary="s")
    peeked = (await handoffs.resume_handoff(memory, agent="b", project="p", claim=False))["handoff"]
    assert peeked["status"] == "open"
    with pytest.raises(ValueError):
        await handoffs.create_handoff(memory, project="p", from_agent="a", summary="  ")
    with pytest.raises(ValueError):
        await handoffs.create_handoff(memory, project="", from_agent="a", summary="s")
    with pytest.raises(ValueError):
        await handoffs.finish_handoff(memory, handoff_id=created["id"], agent="a", status="bogus")
    assert await handoffs.finish_handoff(memory, handoff_id="missing", agent="a") is None
    released = await handoffs.finish_handoff(memory, handoff_id=created["id"], agent="b", status="open")
    assert released["status"] == "open"


# --- Session briefing ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_project_context_gathers_what_a_new_session_needs(memory, store):
    store.set_project_goal(user_id="global_user", namespace="global", project="Muninn",
                           goal_statement="Ship v3.25", constraints=["no data loss"])
    store.set_user_profile(user_id="global_user", profile={"editor": "vim"})
    _remember(store, "Always run the full suite before pushing", category="project_instruction", agent="claude-code")
    _remember(store, "Qdrant payload keeps archived flag", agent="codex", created_at=1.0)
    _remember(store, "BM25 rebuild keeps scope", agent="claude-desktop", created_at=2.0)
    _remember(store, "Prefers concise answers", project="global", scope="global", agent="claude-desktop")
    _remember(store, "Unrelated project fact", project="Other")
    await handoffs.create_handoff(memory, project="Muninn", from_agent="gemini-cli", summary="half done")

    context = await handoffs.project_context(memory, project="Muninn")

    assert context["goal"] == {"goal_statement": "Ship v3.25", "constraints": ["no data loss"]}
    assert [h["from_agent"] for h in context["active_handoffs"]] == ["gemini-cli"]
    assert [m["memory"] for m in context["instructions"]] == ["Always run the full suite before pushing"]
    assert [m["memory"] for m in context["recent_memories"]] == [
        "BM25 rebuild keeps scope", "Qdrant payload keeps archived flag"]
    assert [m["memory"] for m in context["global_preferences"]] == ["Prefers concise answers"]
    assert context["user_profile"] == {"editor": "vim"}
    assert context["agents"] == ["claude-code", "claude-desktop", "codex", "gemini-cli"]
    assert "resume_handoff" in context["hint"]


@pytest.mark.asyncio
async def test_project_context_without_project_lists_where_work_is_waiting(memory):
    await handoffs.create_handoff(memory, project="alpha", from_agent="a", summary="s")
    context = await handoffs.project_context(memory, project=None)
    assert context["projects_with_open_handoffs"] == ["alpha"]
    assert "Pass project" in context["hint"]


# --- REST -----------------------------------------------------------------------

def test_rest_handoff_flow(monkeypatch, memory):
    import server

    monkeypatch.setattr(server, "memory", memory)
    client = TestClient(server.app)

    created = client.post("/handoffs", json={"project": "p", "summary": "s", "from_agent": "codex"}).json()["data"]
    assert client.get("/handoffs", params={"project": "p", "status": "open"}).json()["data"][0]["id"] == created["id"]
    resumed = client.post("/handoffs/resume", json={"agent": "claude-code", "project": "p"}).json()["data"]
    assert resumed["handoff"]["claimed_by"] == "claude-code"
    finished = client.post(f"/handoffs/{created['id']}/finish", json={"agent": "claude-code"})
    assert finished.json()["data"]["status"] == "done"
    assert client.post("/handoffs/nope/finish", json={"agent": "x"}).status_code == 404
    assert client.post("/handoffs", json={"project": "p", "summary": " "}).status_code == 400
    assert client.get("/context", params={"project": "p"}).json()["data"]["project"] == "p"


# --- Agent identity and project over MCP ----------------------------------------

@pytest.fixture
def mcp_state(monkeypatch):
    monkeypatch.delenv("MUNINN_AGENT_NAME", raising=False)
    monkeypatch.delenv("MUNINN_PROJECT", raising=False)
    monkeypatch.delenv("MUNINN_MCP_TOOLSET", raising=False)
    with _SESSION_CONTEXTS_LOCK:
        previous = copy.deepcopy(_SESSION_CONTEXTS)
        _SESSION_CONTEXTS.clear()
    yield
    if hasattr(_thread_local, "mcp_session_id"):
        delattr(_thread_local, "mcp_session_id")
    with _SESSION_CONTEXTS_LOCK:
        _SESSION_CONTEXTS.clear()
        _SESSION_CONTEXTS.update(previous)


@pytest.mark.parametrize("client_name, expected", [
    ("claude-code", "claude-code"),
    ("claude-ai", "claude-desktop"),
    ("codex-mcp-client", "codex"),
    ("Visual Studio Code", "vscode"),
    ("", "unknown"),
])
def test_agent_comes_from_client_info(mcp_state, client_name, expected):
    _SESSION_STATE["client_info"] = {"name": client_name}
    assert handlers.client_agent() == expected


def test_agent_override_order(mcp_state, monkeypatch):
    _SESSION_STATE["client_info"] = {"name": "claude-code"}
    monkeypatch.setenv("MUNINN_AGENT_NAME", "Claude Desktop")
    assert handlers.client_agent() == "claude-desktop"
    _SESSION_STATE["agent_name"] = "codex"
    assert handlers.client_agent() == "codex"


def test_network_sessions_never_use_the_server_directory(mcp_state, monkeypatch):
    monkeypatch.setattr(handlers, "get_git_info", lambda: {"project": "server-cwd", "branch": "main"})
    monkeypatch.setenv("MUNINN_AGENT_NAME", "from-server-env")
    _thread_local.mcp_session_id = "http-session"
    _SESSION_STATE["client_info"] = {"name": "codex-mcp-client"}
    assert handlers.client_project({}) == {"project": None, "branch": None}
    assert handlers.client_project({"project": "Muninn"})["project"] == "Muninn"
    assert handlers.client_agent() == "codex"  # server env is not the client's


def test_stdio_uses_git_repo_or_pinned_project(mcp_state, monkeypatch):
    monkeypatch.setattr(handlers, "get_git_info", lambda: {"project": "Muninn", "branch": "main"})
    assert handlers.client_project({}) == {"project": "Muninn", "branch": "main"}
    monkeypatch.setenv("MUNINN_PROJECT", "pinned")
    assert handlers.client_project({})["project"] == "pinned"
    monkeypatch.delenv("MUNINN_PROJECT")
    monkeypatch.setattr(handlers, "get_git_info", lambda: {"project": "Downloads", "branch": "unknown"})
    assert handlers.client_project({})["project"] is None  # not a repository: no guessed project


def test_add_memory_records_agent_and_explicit_project(mcp_state, monkeypatch):
    calls = []
    response = MagicMock()
    response.json.return_value = {"success": True, "data": {"id": "m1"}}
    monkeypatch.setattr(handlers, "make_request_with_retry", lambda *a, **k: calls.append(k["json"]) or response)
    _thread_local.mcp_session_id = "http-session"
    _SESSION_STATE["client_info"] = {"name": "claude-ai"}

    handlers._do_add_memory({"content": "fact", "project": "Muninn"}, None)

    assert calls[0]["agent_id"] == "claude-desktop"
    assert calls[0]["metadata"]["project"] == "Muninn" and calls[0]["metadata"]["agent"] == "claude-desktop"


def test_create_handoff_tool_needs_a_project_over_http(mcp_state):
    _thread_local.mcp_session_id = "http-session"
    with pytest.raises(ValueError, match="needs project"):
        handlers._do_create_handoff({"summary": "s"}, None)


def test_briefing_tools_return_whole_nested_json(mcp_state, monkeypatch):
    monkeypatch.setenv("MUNINN_MCP_AUTOSTART_SERVER", "0")
    response = MagicMock()
    response.json.return_value = {"success": True, "data": {
        "project": "p", "active_handoffs": [{"id": "h1", "details": {"next_steps": ["deep step"]}}]}}
    monkeypatch.setattr(handlers, "make_request_with_retry", lambda *a, **k: response)
    sent = []
    handlers.handle_call_tool(1, {"name": "get_project_context", "arguments": {"project": "p"}},
                              lambda *a: sent.append(a), lambda _id, result: sent.append(result))
    assert "deep step" in sent[0]["content"][0]["text"]


# --- Built-in prompting ---------------------------------------------------------------

def test_instructions_carry_the_shared_memory_protocol():
    full = protocol_for("full")
    for phrase in ("get_project_context", "resume_handoff", "create_handoff", "project", "secrets"):
        assert phrase in full
    assert "create_handoff" not in protocol_for("readonly")
    assert protocol_for("chatgpt").startswith("Muninn is the user's shared memory")


def test_prompts_follow_the_toolset():
    assert [p["name"] for p in list_prompts("full")] == ["start", "resume", "handoff", "remember"]
    assert [p["name"] for p in list_prompts("readonly")] == ["start"]
    assert list_prompts("chatgpt") == []


def test_prompt_rendering_and_validation():
    text = get_prompt("handoff", {"project": "Muninn", "to_agent": "codex", "notes": "flaky test"}, "full")
    message = text["messages"][0]["content"]["text"]
    assert 'project "Muninn"' in message and "to codex" in message and "flaky test" in message
    unnamed = get_prompt("start", {}, "full")["messages"][0]["content"]["text"]
    assert "repository or folder name" in unnamed
    with pytest.raises(ValueError, match="content"):
        get_prompt("remember", {}, "full")
    with pytest.raises(ValueError, match="Unknown prompt"):
        get_prompt("handoff", {}, "readonly")
