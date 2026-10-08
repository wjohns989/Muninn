"""The service must not accidentally start with authentication disabled."""

import pytest

import server


def test_no_auth_requires_explicit_loopback_opt_in(monkeypatch):
    monkeypatch.setenv("MUNINN_NO_AUTH", "1")
    monkeypatch.delenv("MUNINN_DEV_MODE", raising=False)

    with pytest.raises(RuntimeError, match="Authentication is disabled"):
        server._assert_startup_auth("127.0.0.1")
    with pytest.raises(RuntimeError, match="Authentication is disabled"):
        server._assert_startup_auth("0.0.0.0", allow_no_auth=True)
    server._assert_startup_auth("127.0.0.1", allow_no_auth=True)
    server._assert_startup_auth("::1", allow_no_auth=True)


def test_dev_mode_without_token_requires_opt_in(monkeypatch):
    monkeypatch.delenv("MUNINN_NO_AUTH", raising=False)
    monkeypatch.delenv("MUNINN_API_KEY", raising=False)
    monkeypatch.delenv("MUNINN_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("MUNINN_SERVER_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("MUNINN_DEV_MODE", "true")

    with pytest.raises(RuntimeError, match="Authentication is disabled"):
        server._assert_startup_auth("localhost")
    server._assert_startup_auth("localhost", allow_no_auth=True)


def test_explicit_token_enforces_auth_even_in_dev_mode(monkeypatch):
    monkeypatch.delenv("MUNINN_NO_AUTH", raising=False)
    monkeypatch.setenv("MUNINN_DEV_MODE", "true")
    monkeypatch.setenv("MUNINN_AUTH_TOKEN", "x" * 32)

    server._assert_startup_auth("0.0.0.0")


def test_main_checks_security_before_existing_server_probe(monkeypatch):
    monkeypatch.setenv("MUNINN_NO_AUTH", "1")
    monkeypatch.setattr(server.sys, "argv", ["server.py"])
    monkeypatch.setattr(server, "_existing_server_healthy", lambda *_: pytest.fail("probe ran before auth guard"))

    with pytest.raises(RuntimeError, match="Authentication is disabled"):
        server.main()
