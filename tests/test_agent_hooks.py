"""Session hooks for Claude Code and Codex: briefing at start, capture before compaction and at exit."""

import asyncio
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _legacy_history_test_mode(monkeypatch):
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "legacy")

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


@pytest.mark.parametrize("event,force", [
    ("PreCompress", True), ("AfterAgent", False), ("SessionEnd", True),
])
def test_gemini_events_capture_without_blocking(memory, event, force):
    service = RecordingService()
    out = asyncio.run(handle_hook("gemini-cli", {
        "hook_event_name": event, "transcript_path": "/chat.json",
        "cwd": "/nowhere"}, memory, service))
    assert out == {}
    assert service.calls == [("/chat.json", "gemini_cli", force)]


def test_gemini_session_start_returns_context(memory, repo):
    out = asyncio.run(handle_hook("gemini-cli", {
        "hook_event_name": "SessionStart", "source": "startup",
        "cwd": str(repo)}, memory, RecordingService()))
    assert out["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert out["hookSpecificOutput"]["additionalContext"]


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
    assert stop[0]["hooks"][0]["command"] == "notify.sh" and stop[1]["hooks"][0]["type"] == "command"
    assert plan.after["hooks"]["SessionStart"][0]["matcher"] == "startup|resume|clear|compact"
    start_handler = plan.after["hooks"]["SessionStart"][0]["hooks"][0]
    assert start_handler["type"] == "command"
    assert start_handler["command"].endswith('hook_client.py" claude-code "http://127.0.0.1:42069"')
    assert start_handler["timeout"] == 10
    for event in ("SessionStart", "PreCompact", "Stop", "SessionEnd"):
        handler = plan.after["hooks"][event][-1]["hooks"][0]
        assert handler["type"] == "command"
        assert 'hook_client.py" claude-code "http://127.0.0.1:42069"' in handler["command"]
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


def test_gemini_plan_preserves_unrelated_hooks_and_uses_millisecond_timeouts(tmp_path):
    assert hook_install.GEMINI_EVENTS_MS == {
        "SessionStart": 30000, "PreCompress": 30000,
        "AfterAgent": 10000, "SessionEnd": 30000,
    }
    settings = tmp_path / ".gemini" / "settings.json"
    settings.parent.mkdir()
    original = {"model": {"name": "keep"}, "hooks": {
        "SessionEnd": [{"hooks": [{"name": "other", "type": "command", "command": "notify.exe"}]}]}}
    settings.write_text(json.dumps(original))
    plan = hook_install.gemini_plan("http://127.0.0.1:42069", home=tmp_path,
                                    python="C:/Python/python.exe")
    assert plan.changed
    assert plan.after["model"] == original["model"]
    assert plan.after["hooks"]["SessionEnd"][0] == original["hooks"]["SessionEnd"][0]
    assert set(plan.after["hooks"]) == {"SessionStart", "PreCompress", "AfterAgent", "SessionEnd"}
    for event, timeout in hook_install.GEMINI_EVENTS_MS.items():
        handler = plan.after["hooks"][event][-1]["hooks"][0]
        assert handler["name"] == "muninn-local-memory"
        assert handler["timeout"] == timeout
        if os.name == "nt":
            assert handler["command"].startswith("& 'C:/Python/python.exe' ")
            assert handler["command"].endswith("hook_client.py' gemini-cli 'http://127.0.0.1:42069'")
        else:
            assert handler["command"].startswith('"C:/Python/python.exe" ')
            assert handler["command"].endswith('hook_client.py" gemini-cli "http://127.0.0.1:42069"')
    backup = hook_install.apply_plan(plan)
    assert backup and json.loads(backup.read_text()) == original
    assert not hook_install.gemini_plan("http://127.0.0.1:42069", home=tmp_path,
                                         python="C:/Python/python.exe").changed
    hook_install.apply_plan(hook_install.gemini_plan("http://127.0.0.1:42069",
                                                       install=False, home=tmp_path))
    assert json.loads(settings.read_text()) == original


def test_gemini_command_quotes_windows_and_unix_independently():
    python = "C:/Program Files/O'Brien & Co/python.exe"
    client = Path("C:/User Files/O'Brien & Co/hook_client.py")
    url = "http://127.0.0.1:42069/a&b's"
    windows = hook_install._gemini_command(python, client, url, windows=True)
    unix = hook_install._gemini_command(python, client, url, windows=False)
    assert windows.startswith("& '")
    assert "O''Brien" in windows and "b''s" in windows
    assert unix.startswith(f'"{python}" ')


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell hook execution proof")
def test_gemini_windows_hook_command_runs_from_powershell(tmp_path):
    folder = tmp_path / "O'Brien & Co"
    folder.mkdir()
    client = folder / "probe client.py"
    client.write_text(
        "import sys\nprint('|'.join(sys.argv[1:]))\nsys.exit(7 if '--fail' in sys.argv else 0)\n",
        encoding="utf-8",
    )
    url = "http://127.0.0.1:42069/a&b's"
    for fail in (False, True):
        suffix = "--fail" if fail else url
        command = hook_install._gemini_command(sys.executable, client, suffix, windows=True)
        # Gemini CLI's Windows hook runner appends this exact exit-code guard.
        command += "; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }"
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, timeout=60, check=False,
        )
        assert result.returncode == (7 if fail else 0), result.stderr.decode(errors="replace")
        assert result.stdout.decode().strip() == f"gemini-cli|{suffix}"


# --- the command-hook client -----------------------------------------------------------

CLIENT = Path(__file__).resolve().parent.parent / "muninn" / "hook_client.py"


def _run_client(payload, env):
    # Windows sockets need SystemRoot even in a deliberately sparse test env.
    base = {key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ}
    return subprocess.run([sys.executable, "-I", str(CLIENT), "codex"], input=json.dumps(payload),
                          capture_output=True, text=True, env={**base, **env}, timeout=20)


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
    assert result.returncode == 0 and result.stdout, result.stderr
    assert json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"] == "brief"
    assert received == {"path": "/hooks/codex", "auth": "Bearer tok",
                        "body": {"hook_event_name": "SessionStart", "cwd": "/x"}}


def test_client_never_blocks_the_agent_when_the_server_is_down():
    result = _run_client({"hook_event_name": "SessionEnd"}, {"MUNINN_SERVER_URL": "http://127.0.0.1:9"})
    assert result.returncode == 0 and result.stdout == "" and "not reachable" in result.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="Windows User environment fallback")
def test_client_reads_windows_user_token_when_process_token_is_missing(monkeypatch):
    from contextlib import nullcontext

    import winreg

    from muninn import hook_client

    monkeypatch.delenv("MUNINN_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(winreg, "OpenKey", lambda *_: nullcontext(None))
    monkeypatch.setattr(winreg, "QueryValueEx", lambda *_: ("test-only-token", winreg.REG_SZ))
    assert hook_client._auth_token() == "test-only-token"


def test_hook_token_selection_is_endpoint_aware(monkeypatch):
    from muninn import hook_client

    monkeypatch.setattr(hook_client.sys, "platform", "win32")
    monkeypatch.setattr(hook_client, "_windows_user_token", lambda: "current-user-token", raising=False)
    monkeypatch.setenv("MUNINN_AUTH_TOKEN", "stale-process-token")
    assert hook_client._auth_token("http://127.0.0.1:42069") == "current-user-token"
    assert hook_client._auth_token("http://localhost:42069") == "current-user-token"
    assert hook_client._auth_token("http://[::1]:42069") == "current-user-token"
    assert hook_client._auth_token("http://127.0.0.1:42070") == "stale-process-token"

    monkeypatch.delenv("MUNINN_AUTH_TOKEN")
    assert hook_client._auth_token("http://127.0.0.1:42070") == "current-user-token"
    assert hook_client._auth_token("https://remote.example") == ""
    assert hook_client._auth_token("http://127.0.0.1.evil:42069") == ""
    assert hook_client._auth_token("http://user@127.0.0.1:42069") == ""
    assert hook_client._auth_token("ftp://127.0.0.1:42069") == ""


def test_hook_client_passes_target_url_into_token_selection(monkeypatch):
    import io

    from muninn import hook_client

    observed = []
    monkeypatch.setattr(hook_client.sys, "stdin", io.StringIO('{"hook_event_name":"AuthProbe"}'))
    monkeypatch.setattr(hook_client, "_auth_token", lambda url: observed.append(url) or "")
    monkeypatch.setattr(hook_client.urllib.request, "urlopen",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline")))
    assert hook_client.main(["hook_client", "codex", "https://remote.example"]) == 0
    assert observed == ["https://remote.example"]


def test_gemini_after_agent_client_timeout_is_shorter_than_host_deadline():
    from muninn import hook_client

    assert "AfterAgent" in hook_client.FAST_EVENTS
    assert hook_install.GEMINI_EVENTS_MS["AfterAgent"] > 800
