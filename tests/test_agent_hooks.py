"""Session hooks for Claude Code and Codex: briefing at start, capture before compaction and at exit."""

import asyncio
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from muninn.history import hook_install
from muninn.history.hooks import handle_hook
from muninn.store.sqlite_metadata import SQLiteMetadataStore

sys.path.insert(0, str(Path(__file__).parent))
from test_history_import import FakeMemory, claude_rows, jsonl  # noqa: E402


class RecordingService:
    def __init__(self):
        self.calls = []

    def capture_later(self, path, provider, *, force=False):
        self.calls.append((path, provider, force))


@pytest.fixture
def memory(tmp_path):
    return FakeMemory(SQLiteMetadataStore(tmp_path / "metadata.db"))


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "Storefront"
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return path


def test_session_start_injects_the_project_briefing(memory, repo):
    asyncio.run(__import__("muninn.core.handoffs", fromlist=["x"]).create_handoff(
        memory, project="Storefront", from_agent="codex", summary="Cart totals half done",
        details={"next_steps": ["round tax per line"]}))
    out = asyncio.run(handle_hook("claude-code", {
        "hook_event_name": "SessionStart", "source": "startup", "cwd": str(repo)}, memory, None))
    text = out["hookSpecificOutput"]["additionalContext"]
    assert out["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert 'project "Storefront"' in text and "Cart totals half done" in text and "round tax per line" in text


def test_session_start_outside_a_project_says_so(memory, tmp_path):
    out = asyncio.run(handle_hook("codex", {"hook_event_name": "SessionStart", "cwd": str(Path.home())}, memory, None))
    assert "no project detected" in out["hookSpecificOutput"]["additionalContext"]


@pytest.mark.parametrize("event, extra, expected", [
    ("PreCompact", {"trigger": "auto"}, [("/t.jsonl", "claude_code", True)]),
    ("SessionEnd", {"reason": "other"}, [("/t.jsonl", "claude_code", True)]),
    ("Stop", {}, [("/t.jsonl", "claude_code", False)]),
    ("SessionStart", {"source": "compact"}, [("/t.jsonl", "claude_code", True)]),
    ("SessionStart", {"source": "startup"}, []),
    ("UserPromptSubmit", {}, []),
])
def test_capture_follows_the_event(memory, event, extra, expected):
    service = RecordingService()
    out = asyncio.run(handle_hook("claude-code", {"hook_event_name": event, "transcript_path": "/t.jsonl",
                                                  "cwd": "/nowhere", **extra}, memory, service))
    assert service.calls == expected
    assert (out == {}) == (event != "SessionStart")


def test_unknown_agents_and_missing_transcripts_are_ignored(memory):
    service = RecordingService()
    asyncio.run(handle_hook("someone", {"hook_event_name": "PreCompact", "transcript_path": "/t"}, memory, service))
    asyncio.run(handle_hook("codex", {"hook_event_name": "PreCompact", "transcript_path": None}, memory, service))
    assert service.calls == []


def test_capture_imports_one_thread_and_debounces(tmp_path, monkeypatch):
    from muninn.history.service import HistoryService

    for var in ("CLAUDE_CONFIG_DIR", "CODEX_HOME", "MUNINN_HISTORY_HOMES"):
        monkeypatch.delenv(var, raising=False)
    home = tmp_path / "home"
    transcript = jsonl(home / ".claude" / "projects" / "-x" / "live.jsonl", claude_rows("/x", session="live"))
    memory = FakeMemory(SQLiteMetadataStore(tmp_path / "metadata.db"))
    service = HistoryService(memory, tmp_path / "vault", home=home)

    async def scenario():
        first = await service.capture(str(transcript), "claude_code")
        assert first["captured"] and first["turn_memories"] >= 3
        assert memory._metadata.get_history_thread("claude_code:live")["turns_imported"] == 3
        assert (await service.capture(str(transcript), "claude_code"))["skipped"] == "debounced"
        forced = await service.capture(str(transcript), "claude_code", force=True)
        assert forced["turn_memories"] == 0  # nothing new: incremental
        await service.stop()

    asyncio.run(scenario())


# --- installer -------------------------------------------------------------------

def test_install_keeps_user_hooks_and_uninstall_restores_them(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir()
    original = {"model": "opus", "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "notify.sh"}]}]}}
    settings.write_text(json.dumps(original))

    plan = hook_install.claude_plan("http://127.0.0.1:42069", home=tmp_path)
    assert plan.changed and hook_install.installed(plan) == []
    stop = plan.after["hooks"]["Stop"]
    assert stop[0]["hooks"][0]["command"] == "notify.sh" and stop[1]["hooks"][0]["type"] == "http"
    assert plan.after["hooks"]["SessionStart"][0]["matcher"] == "startup|resume|clear|compact"
    backup = hook_install.apply_plan(plan)
    assert backup and json.loads(backup.read_text()) == original

    again = hook_install.claude_plan("http://127.0.0.1:42069", home=tmp_path)
    assert not again.changed and hook_install.installed(again) == ["PreCompact", "SessionEnd", "SessionStart", "Stop"]
    hook_install.apply_plan(hook_install.claude_plan("http://127.0.0.1:42069", install=False, home=tmp_path))
    assert json.loads(settings.read_text()) == original


def test_codex_plan_uses_the_stdlib_client_and_honours_codex_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-moved"))
    plan = hook_install.codex_plan(home=tmp_path, python="/usr/bin/python3")
    assert plan.path == tmp_path / "codex-moved" / "hooks.json"
    handler = plan.after["hooks"]["SessionEnd"][0]["hooks"][0]
    assert handler["command"].startswith('"/usr/bin/python3" "')
    assert handler["command"].endswith('hook_client.py" codex')
    assert handler["timeout"] == 1 and set(plan.after["hooks"]) == {"SessionStart", "PreCompact", "Stop", "SessionEnd"}


# --- the command-hook client -----------------------------------------------------------

CLIENT = Path(__file__).resolve().parent.parent / "muninn" / "hook_client.py"


def _run_client(payload, env):
    return subprocess.run([sys.executable, "-I", str(CLIENT), "codex"], input=json.dumps(payload),
                          capture_output=True, text=True, env=env, timeout=20)


def test_client_forwards_and_prints_the_briefing():
    received = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received["path"] = self.path
            received["auth"] = self.headers.get("Authorization")
            received["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.dumps({"hookSpecificOutput": {"additionalContext": "brief"}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.handle_request, daemon=True).start()
    env = {"MUNINN_SERVER_URL": f"http://127.0.0.1:{server.server_port}", "MUNINN_AUTH_TOKEN": "tok"}
    result = _run_client({"hook_event_name": "SessionStart", "cwd": "/x"}, env)
    server.server_close()
    assert result.returncode == 0 and json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"] == "brief"
    assert received == {"path": "/hooks/codex", "auth": "Bearer tok",
                        "body": {"hook_event_name": "SessionStart", "cwd": "/x"}}


def test_client_never_blocks_the_agent_when_the_server_is_down():
    result = _run_client({"hook_event_name": "SessionEnd"}, {"MUNINN_SERVER_URL": "http://127.0.0.1:9"})
    assert result.returncode == 0 and result.stdout == "" and "not reachable" in result.stderr
