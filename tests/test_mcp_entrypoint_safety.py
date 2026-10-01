"""Isolated bridge compatibility and privacy; never contact local services."""
import logging
import json
import os
from pathlib import Path
import subprocess
import site
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

import mcp_wrapper
from muninn.mcp import main as entrypoint


def test_package_main_delegates_to_the_canonical_wrapper(monkeypatch):
    calls = []
    monkeypatch.setattr(mcp_wrapper, "main", lambda: calls.append("canonical"))
    # Keep the preimage safe too: it must not read stdin, initialize credentials
    # or dispatch its background dependency checks during this red-first test.
    monkeypatch.setattr(entrypoint, "sys", SimpleNamespace(platform="fixture", stderr=sys.stderr))
    monkeypatch.setattr(entrypoint, "initialize_security", lambda: None)
    monkeypatch.setattr(entrypoint, "_server", SimpleNamespace(
        read_message=lambda *args: None, stop=lambda: None))
    monkeypatch.setattr(entrypoint.threading, "Thread", lambda **kwargs: SimpleNamespace(start=lambda: None))
    entrypoint._TRANSPORT_CLOSED.set()
    try:
        entrypoint.main()
    finally:
        entrypoint._TRANSPORT_CLOSED.clear()
    assert calls == ["canonical"]


def test_default_startup_warnings_only_check_readiness(monkeypatch):
    def forbidden():
        pytest.fail("readiness inspection attempted lifecycle action")
    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", forbidden)
    monkeypatch.setattr(mcp_wrapper, "check_and_start_ollama", forbidden)
    monkeypatch.setattr(mcp_wrapper, "is_server_running", lambda: False, raising=False)
    monkeypatch.setattr(mcp_wrapper, "is_ollama_running", lambda: False, raising=False)
    warnings = mcp_wrapper._collect_startup_warnings()
    assert any("Muninn" in warning for warning in warnings)
    assert any("Ollama" in warning for warning in warnings)


@pytest.mark.parametrize("name", ["_append_initialize_trace", "_append_tool_call_trace"])
def test_diagnostics_never_persist_or_log_supplied_payload(tmp_path, monkeypatch, caplog, name):
    monkeypatch.setattr(entrypoint, "_INITIALIZE_TRACE_PATH", str(tmp_path / "initialize.jsonl"))
    monkeypatch.setattr(entrypoint, "_TOOL_CALL_TRACE_PATH", str(tmp_path / "tools.jsonl"))
    canary = "fixture-secret-must-not-reach-diagnostics"
    with caplog.at_level(logging.DEBUG):
        getattr(entrypoint, name)("initialize_request", {"params": canary, "id": canary})
    assert list(tmp_path.iterdir()) == []
    assert canary not in caplog.text


@pytest.mark.parametrize("module", [mcp_wrapper, entrypoint])
def test_guarded_dispatch_does_not_expose_exception_values(monkeypatch, caplog, capsys, module):
    sent = []
    canary = "fixture-credential-in-exception"
    def fail(message):
        raise ValueError(canary)
    monkeypatch.setattr(module, "_dispatch_rpc_message", fail)
    monkeypatch.setattr(module, "send_json_rpc", sent.append)
    module._TRANSPORT_CLOSED.clear()
    with caplog.at_level(logging.ERROR):
        module._dispatch_rpc_message_guarded({"id": "fixture", "method": "tools/call"})
    assert len(sent) == 1 and sent[0]["error"]["code"] == -32603
    assert canary not in str(sent)
    assert canary not in caplog.text
    assert canary not in capsys.readouterr().err


def test_executable_package_bridge_uses_fixture_only_readiness_and_no_plaintext_trace(tmp_path):
    requests_seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests_seen.append(("GET", self.path))
            self.send_response(503)
            self.end_headers()
        def do_POST(self):
            requests_seen.append(("POST", self.path))
            self.send_response(503)
            self.end_headers()
        def log_message(self, *args):
            pass
    fixture = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=fixture.serve_forever, daemon=True)
    thread.start()
    # Only OS essentials are inherited. No user environment, vault, key, client
    # profile, transcript or provider configuration is copied into this fixture.
    environment = {name: os.environ[name] for name in ("SYSTEMROOT", "WINDIR", "COMSPEC")
                   if name in os.environ}
    base = f"http://127.0.0.1:{fixture.server_port}"
    environment.update({"MUNINN_AUTH_TOKEN": "fixture-only-not-a-real-token-0000000000",
        "PYTHONPATH": os.pathsep.join([*site.getsitepackages(), site.getusersitepackages()]),
        "MUNINN_DATA_DIR": str(tmp_path), "TEMP": str(tmp_path), "TMP": str(tmp_path),
        "MUNINN_SERVER_URL": base, "MUNINN_OLLAMA_URL": base,
        "MUNINN_MCP_AUTO_START": "0", "MUNINN_MCP_AUTOSTART_ON_LAUNCH": "0",
        "MUNINN_MCP_AUTOSTART_SERVER": "0", "MUNINN_MCP_AUTOSTART_OLLAMA": "0"})
    canary = "fixture-private-client-metadata"
    messages = [
        {"jsonrpc": "2.0", "id": "initialize", "method": "initialize", "params": {
            "protocolVersion": "2025-11-25", "capabilities": {},
            "clientInfo": {"name": "fixture", "version": "1", "private": canary}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": "tools", "method": "tools/list"},
        {"jsonrpc": "2.0", "id": "ping", "method": "ping"},
    ]
    try:
        result = subprocess.run([sys.executable, "-B", "-m", "muninn.mcp"],
            cwd=Path(__file__).resolve().parents[1], env=environment,
            input="\n".join(json.dumps(message) for message in messages) + "\n",
            capture_output=True, text=True, timeout=15)
    finally:
        fixture.shutdown()
        fixture.server_close()
        thread.join(timeout=2)
    assert result.returncode == 0, result.stderr
    responses = {message["id"]: message for message in map(json.loads, result.stdout.splitlines())}
    assert responses["initialize"]["result"]["protocolVersion"] == "2025-11-25"
    assert responses["tools"]["result"]["tools"]
    assert responses["ping"]["result"] == {}
    assert canary not in result.stdout + result.stderr
    assert set(requests_seen) == {("GET", "/health"), ("GET", "/")}
    assert not list(tmp_path.rglob("*.jsonl"))
