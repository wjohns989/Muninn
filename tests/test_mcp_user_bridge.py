"""The Windows MCP bridge must never leak a user token or launch services."""

import os
import types

import pytest

import muninn_mcp_bridge as bridge


@pytest.fixture
def valid_token(monkeypatch):
    token = "A" * 40
    monkeypatch.setattr(bridge, "_read_windows_user_token", lambda: token)
    monkeypatch.delenv("MUNINN_SERVER_URL", raising=False)
    return token


def test_bridge_uses_registry_token_even_without_process_env(monkeypatch, valid_token):
    monkeypatch.delenv("MUNINN_AUTH_TOKEN", raising=False)
    bridge.prepare_environment()
    assert os.environ["MUNINN_AUTH_TOKEN"] == valid_token
    assert os.environ["MUNINN_SERVER_URL"] == "http://127.0.0.1:42069"
    assert os.environ["MUNINN_NO_AUTH"] == "0"
    for name in ("MUNINN_MCP_AUTO_START", "MUNINN_MCP_AUTOSTART_ON_LAUNCH",
                 "MUNINN_MCP_AUTOSTART_SERVER", "MUNINN_MCP_AUTOSTART_OLLAMA"):
        assert os.environ[name] == "0"
    assert os.environ["NO_PROXY"] == "*"
    assert os.environ["no_proxy"] == "*"


def test_bridge_replaces_stale_process_token(monkeypatch, valid_token):
    monkeypatch.setenv("MUNINN_AUTH_TOKEN", "stale-host-token")
    bridge.prepare_environment()
    assert os.environ["MUNINN_AUTH_TOKEN"] == valid_token


@pytest.mark.parametrize("value", [None, "", " ", "short", "A" * 31, "A" * 32 + "\n"])
def test_bridge_rejects_invalid_user_token_without_importing_wrapper(monkeypatch, capsys, value):
    monkeypatch.setattr(bridge, "_read_windows_user_token", lambda: value)
    imported = []
    monkeypatch.setattr(bridge.importlib, "import_module", lambda name: imported.append(name))
    assert bridge.main() == 2
    assert imported == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Muninn MCP bridge: authenticated local connection unavailable\n"


@pytest.mark.parametrize("url", [
    "https://example.com/mcp", "http://localhost:42069", "http://127.0.0.1:42070",
    "http://127.0.0.1:42069/mcp", "http://127.0.0.1:42069@evil.example",
])
def test_bridge_rejects_url_override(monkeypatch, valid_token, url):
    monkeypatch.setenv("MUNINN_SERVER_URL", url)
    with pytest.raises(bridge.BridgeError):
        bridge.prepare_environment()
    assert os.environ["MUNINN_SERVER_URL"] == url


def test_bridge_delegates_only_after_secure_setup(monkeypatch, valid_token, capsys):
    called = []

    def import_module(name):
        called.append((name, os.environ["MUNINN_AUTH_TOKEN"],
                       os.environ["MUNINN_MCP_AUTO_START"]))
        return types.SimpleNamespace(main=lambda: called.append("main"))

    monkeypatch.setattr(bridge.importlib, "import_module", import_module)
    assert bridge.main() == 0
    assert called == [("mcp_wrapper", valid_token, "0"), "main"]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert valid_token not in captured.err


def test_bridge_registry_failure_is_safe(monkeypatch, capsys):
    def failure():
        raise OSError("private registry detail")

    monkeypatch.setattr(bridge, "_read_windows_user_token", failure)
    assert bridge.main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "private registry detail" not in captured.err
