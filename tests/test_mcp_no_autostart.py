"""MCP clients must not create a second backend by default."""

from contextlib import nullcontext

from muninn.mcp import lifecycle


def test_unavailable_backend_does_not_spawn_without_explicit_opt_in(monkeypatch):
    calls = []
    monkeypatch.delenv("MUNINN_MCP_AUTO_START", raising=False)
    monkeypatch.setenv("MUNINN_MCP_AUTOSTART_SERVER", "1")
    monkeypatch.setattr(lifecycle, "is_server_running", lambda: False)
    monkeypatch.setattr(lifecycle, "start_server", lambda: calls.append("spawn") or True)
    monkeypatch.setattr(lifecycle, "_startup_spawn_lock", nullcontext)

    assert lifecycle.ensure_server_running() is False
    assert calls == []


def test_explicit_opt_in_allows_existing_spawn_path(monkeypatch):
    state = {"spawned": False}
    monkeypatch.setenv("MUNINN_MCP_AUTO_START", "true")
    monkeypatch.setattr(lifecycle, "is_server_running", lambda: state["spawned"])
    monkeypatch.setattr(lifecycle, "start_server", lambda: state.update(spawned=True) or True)
    monkeypatch.setattr(lifecycle, "_startup_spawn_lock", nullcontext)

    assert lifecycle.ensure_server_running() is True
    assert state["spawned"] is True
