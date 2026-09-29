import json
from pathlib import Path
from unittest.mock import patch

import pytest

import muninn.cli as cli


@pytest.fixture(autouse=True)
def _isolate_codex_config(tmp_path: Path, monkeypatch) -> None:
    """Never let CLI tests inspect or modify the user's real Codex config."""
    monkeypatch.setattr(cli, "_CODEX_CONFIG_PATH", tmp_path / "codex-config.toml")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_doctor_detects_drift_and_repairs_config(tmp_path, monkeypatch):
    token_file = tmp_path / ".muninn_token"
    token_file.write_text("expected-token", encoding="utf-8")
    cfg_path = tmp_path / "mcp.json"
    _write_json(
        cfg_path,
        {
            "mcpServers": {
                "muninn": {
                    "command": "python",
                    "env": {
                        "MUNINN_AUTH_TOKEN": "old-token",
                        "MUNINN_SERVER_URL": "http://127.0.0.1:9999",
                    },
                }
            }
        },
    )

    monkeypatch.setattr(cli, "_MCP_CONFIG_PATHS", [cfg_path])
    monkeypatch.setattr(cli, "_check_server_health", lambda url, token, timeout: (True, "ok"))

    args = cli.build_parser().parse_args(
        [
            "doctor",
            "--token-file",
            str(token_file),
            "--server-url",
            "http://127.0.0.1:42069",
        ]
    )
    rc = cli.cmd_doctor(args)
    assert rc == 1

    repair_args = cli.build_parser().parse_args(
        [
            "doctor",
            "--token-file",
            str(token_file),
            "--server-url",
            "http://127.0.0.1:42069",
            "--repair",
        ]
    )
    repair_rc = cli.cmd_doctor(repair_args)
    assert repair_rc == 0

    patched = _read_json(cfg_path)
    env = patched["mcpServers"]["muninn"]["env"]
    assert env["MUNINN_AUTH_TOKEN"] == "expected-token"
    assert env["MUNINN_SERVER_URL"] == "http://127.0.0.1:42069"


def test_doctor_returns_critical_when_no_expected_token(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_MCP_CONFIG_PATHS", [])
    monkeypatch.delenv("MUNINN_AUTH_TOKEN", raising=False)
    args = cli.build_parser().parse_args(
        [
            "doctor",
            "--token-file",
            str(tmp_path / "missing.token"),
            "--server-url",
            "http://127.0.0.1:42069",
        ]
    )
    rc = cli.cmd_doctor(args)
    assert rc == 2


def test_admin_token_selection_prefers_explicit_then_environment_over_stale_default(tmp_path, monkeypatch):
    default_file = tmp_path / "default.token"
    explicit_file = tmp_path / "explicit.token"
    configured_file = tmp_path / "configured.token"
    default_file.write_text("stale-default", encoding="utf-8")
    explicit_file.write_text("explicit-token", encoding="utf-8")
    configured_file.write_text("configured-token", encoding="utf-8")
    monkeypatch.setattr(cli, "_DEFAULT_TOKEN_FILE", default_file)
    monkeypatch.setattr(cli, "_read_windows_user_auth_token", lambda: "registry-token")
    monkeypatch.setenv("MUNINN_AUTH_TOKEN", "process-token")
    monkeypatch.delenv("MUNINN_TOKEN_FILE", raising=False)

    assert cli._select_auth_token(explicit_file) == ("explicit-token", "file")
    assert cli._select_auth_token(None) == ("process-token", "env")
    monkeypatch.setenv("MUNINN_TOKEN_FILE", str(configured_file))
    assert cli._select_auth_token(None) == ("configured-token", "file")
    monkeypatch.delenv("MUNINN_TOKEN_FILE")
    monkeypatch.delenv("MUNINN_AUTH_TOKEN")
    assert cli._select_auth_token(None) == ("registry-token", "user-env")
    monkeypatch.setattr(cli, "_read_windows_user_auth_token", lambda: None)
    assert cli._select_auth_token(None) == ("stale-default", "file")


def test_doctor_does_not_repair_configs_when_server_rejects_selected_token(tmp_path, monkeypatch):
    token_file = tmp_path / "candidate.token"
    token_file.write_text("unverified-token", encoding="utf-8")
    cfg_path = tmp_path / "mcp.json"
    original = {"mcpServers": {"muninn": {"command": "python", "env": {
        "MUNINN_AUTH_TOKEN": "current-token", "MUNINN_SERVER_URL": "http://127.0.0.1:42069",
    }}}}
    _write_json(cfg_path, original)
    monkeypatch.setattr(cli, "_MCP_CONFIG_PATHS", [cfg_path])
    monkeypatch.setattr(cli, "_check_server_health", lambda url, token, timeout: (False, "401"))
    args = cli.build_parser().parse_args(["doctor", "--token-file", str(token_file), "--repair"])

    assert cli.cmd_doctor(args) == 2
    assert _read_json(cfg_path) == original


def test_authenticated_check_rejects_disabled_auth(monkeypatch):
    calls = []

    class Response:
        def __init__(self, status_code):
            self.status_code = status_code

    def fake_get(url, **kwargs):
        calls.append((url, kwargs["headers"]["Authorization"]))
        return Response(200 if len(calls) == 1 else 401)

    monkeypatch.setattr(cli.requests, "get", fake_get)
    assert cli._check_server_health("http://127.0.0.1:42069", "selected-token", 2) == (True, "ok")
    assert calls[0] == ("http://127.0.0.1:42069/auth/check", "Bearer selected-token")
    assert calls[1][1] != calls[0][1]

    calls.clear()
    monkeypatch.setattr(cli.requests, "get", lambda url, **kwargs: Response(200))
    assert cli._check_server_health("http://127.0.0.1:42069", "selected-token", 2) == (
        False, "auth_not_enforced",
    )


def test_auth_request_errors_never_echo_bearer(monkeypatch, tmp_path):
    secret = "dummy-private-token"

    def fail(*args, **kwargs):
        raise cli.requests.exceptions.InvalidHeader(f"Authorization: Bearer {secret}")

    monkeypatch.setattr(cli.requests, "get", fail)
    assert cli._check_server_health("http://127.0.0.1:42069", secret, 2) == (
        False, "request_InvalidHeader",
    )

    token_file = tmp_path / "explicit.token"
    token_file.write_text(secret, encoding="utf-8")
    monkeypatch.setattr(cli.requests, "request", fail)
    args = cli.build_parser().parse_args([
        "history", "status", "--token-file", str(token_file), "--server-url", "http://127.0.0.1:42069",
    ])
    with pytest.raises(SystemExit) as exc:
        cli._admin_request(args, "GET", "/history/status")
    assert secret not in str(exc.value)


def test_admin_request_uses_user_token_locally_and_fails_closed_remotely(tmp_path, monkeypatch):
    default_file = tmp_path / "default.token"
    default_file.write_text("stale-default", encoding="utf-8")
    monkeypatch.setattr(cli, "_DEFAULT_TOKEN_FILE", default_file)
    monkeypatch.delenv("MUNINN_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("MUNINN_TOKEN_FILE", raising=False)
    monkeypatch.setattr(cli, "_read_windows_user_auth_token", lambda: "registry-token")
    calls = []

    class Response:
        status_code = 200

        def json(self):
            return {"data": {"ok": True}}

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs["headers"]))
        return Response()

    monkeypatch.setattr(cli.requests, "request", fake_request)
    args = cli.build_parser().parse_args(["history", "status", "--server-url", "http://127.0.0.1:42069"])
    assert cli._admin_request(args, "GET", "/history/status") == {"ok": True}
    assert calls == [("GET", "http://127.0.0.1:42069/history/status",
                      {"Authorization": "Bearer registry-token"})]

    remote_args = cli.build_parser().parse_args(["history", "status", "--server-url", "https://remote.example"])
    with pytest.raises(SystemExit, match="non-loopback"):
        cli._admin_request(remote_args, "GET", "/history/status")
    assert len(calls) == 1

    missing = tmp_path / "missing.token"
    explicit_args = cli.build_parser().parse_args([
        "history", "status", "--token-file", str(missing), "--server-url", "http://127.0.0.1:42069",
    ])
    with pytest.raises(SystemExit, match="missing or empty"):
        cli._admin_request(explicit_args, "GET", "/history/status")
    assert len(calls) == 1


@pytest.mark.parametrize("url_field", ["url", "serverUrl"])
def test_http_host_profiles_are_classified_without_env_injection(tmp_path, monkeypatch, url_field):
    cfg_path = tmp_path / "mcp.json"
    config = {"mcpServers": {"muninn": {
        url_field: "http://127.0.0.1:42069/mcp?agent=gemini" if url_field == "url"
                   else "http://127.0.0.1:42069/mcp",
        "headers": {"Authorization": "Bearer ${MUNINN_AUTH_TOKEN}"},
        "type": "http", "trust": True,
    }}}
    _write_json(cfg_path, config)
    before = cfg_path.read_bytes()
    entry = cli._collect_muninn_server_entries(cfg_path)[0]
    assert entry.server_url == "http://127.0.0.1:42069"
    assert not entry.token_check
    assert entry.note == "runtime token unverified"
    assert not cli._patch_mcp_config_env(cfg_path, new_token="new-token",
                                         new_server_url="http://127.0.0.1:42069")
    assert cfg_path.read_bytes() == before

    monkeypatch.setattr(cli, "_MCP_CONFIG_PATHS", [cfg_path])
    monkeypatch.setenv("MUNINN_SERVER_URL", "http://127.0.0.1:42069")
    with patch("muninn.cli.secrets.token_urlsafe", return_value="rotated-token"):
        args = cli.build_parser().parse_args(["rotate-token", "--token-file", str(tmp_path / "rotated.token")])
        assert cli.cmd_rotate_token(args) == 0
    assert cfg_path.read_bytes() == before


def test_codex_http_env_reference_missing_from_process_is_unverified(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text(
        '[mcp_servers.muninn]\nurl = "http://127.0.0.1:42069/mcp"\n'
        'bearer_token_env_var = "MUNINN_AUTH_TOKEN"\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("MUNINN_AUTH_TOKEN", raising=False)
    entry = cli._collect_codex_muninn_entries(config)[0]
    assert entry.server_url == "http://127.0.0.1:42069"
    assert not entry.token_check
    assert entry.note == "runtime token unverified"
    monkeypatch.setenv("MUNINN_AUTH_TOKEN", "possibly-stale-process-token")
    entry = cli._collect_codex_muninn_entries(config)[0]
    assert not entry.token_check
    assert entry.note == "runtime token unverified"


def test_doctor_never_rewrites_unverified_codex_http_profile(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.toml"
    config.write_text(
        '[mcp_servers.muninn]\nurl = "http://127.0.0.1:9999/mcp"\n'
        'bearer_token_env_var = "CUSTOM_MUNINN_TOKEN"\n\n'
        '[mcp_servers.muninn.env]\nKEEP = "original"\n',
        encoding="utf-8",
    )
    before = config.read_bytes()
    monkeypatch.setattr(cli, "_CODEX_CONFIG_PATH", config)
    monkeypatch.setattr(cli, "_MCP_CONFIG_PATHS", [])
    monkeypatch.setattr(cli, "_check_server_health", lambda url, token, timeout: (True, "ok"))
    token_file = tmp_path / "expected.token"
    token_file.write_text("verified-token", encoding="utf-8")
    args = cli.build_parser().parse_args([
        "doctor", "--token-file", str(token_file), "--server-url", "http://127.0.0.1:42069", "--repair",
    ])
    assert cli.cmd_doctor(args) == 1
    assert config.read_bytes() == before
    assert "unverified" in capsys.readouterr().out


@pytest.mark.parametrize("profile", [
    {"url": "http://127.0.0.1:42069/mcp", "command": "python"},
    {"command": "python", "disabled": True},
    {"command": "python", "headers": {"Authorization": "Bearer placeholder"}},
    {"command": "python", "env": {"MUNINN_NO_AUTH": "1"}},
])
def test_unknown_disabled_and_no_auth_profiles_are_never_generic_repair_targets(tmp_path, profile):
    cfg_path = tmp_path / "mcp.json"
    _write_json(cfg_path, {"mcpServers": {"muninn": profile}})
    before = cfg_path.read_bytes()
    entry = cli._collect_muninn_server_entries(cfg_path)[0]
    assert not entry.token_check
    assert entry.note
    assert not cli._patch_mcp_config_env(cfg_path, new_token="new-token",
                                         new_server_url="http://127.0.0.1:42069")
    assert cfg_path.read_bytes() == before


def test_doctor_preserves_dynamic_http_profiles_and_reports_unknown_not_drift(tmp_path, monkeypatch, capsys):
    token_file = tmp_path / "expected.token"
    token_file.write_text("expected-token", encoding="utf-8")
    gemini = tmp_path / "gemini.json"
    antigravity = tmp_path / "antigravity.json"
    _write_json(gemini, {"mcpServers": {"muninn": {
        "url": "http://127.0.0.1:42069/mcp?agent=gemini",
        "headers": {"Authorization": "Bearer ${MUNINN_AUTH_TOKEN}"},
    }}})
    _write_json(antigravity, {"mcpServers": {"muninn": {
        "serverUrl": "http://127.0.0.1:42069/mcp",
        "headers": {"Authorization": "Bearer expected-token"},
    }}})
    before = {path: path.read_bytes() for path in (gemini, antigravity)}
    monkeypatch.setattr(cli, "_MCP_CONFIG_PATHS", [gemini, antigravity])
    monkeypatch.setattr(cli, "_check_server_health", lambda url, token, timeout: (True, "ok"))
    args = cli.build_parser().parse_args([
        "doctor", "--token-file", str(token_file), "--server-url", "http://127.0.0.1:42069", "--repair",
    ])
    assert cli.cmd_doctor(args) == 1
    output = capsys.readouterr().out
    assert "unverified" in output
    assert "token drift" not in output
    assert "URL drift" not in output
    assert all(path.read_bytes() == contents for path, contents in before.items())


def test_rotate_token_patches_server_url_and_token(tmp_path, monkeypatch):
    cfg_path = tmp_path / "mcp.json"
    token_file = tmp_path / ".muninn_token"
    _write_json(
        cfg_path,
        {
            "mcpServers": {
                "muninn": {
                    "command": "python",
                    "env": {},
                }
            }
        },
    )
    monkeypatch.setattr(cli, "_MCP_CONFIG_PATHS", [cfg_path])
    monkeypatch.setenv("MUNINN_SERVER_URL", "http://127.0.0.1:42069")

    with patch("muninn.cli.secrets.token_urlsafe", return_value="token-fixed-123"):
        args = cli.build_parser().parse_args(
            [
                "rotate-token",
                "--token-file",
                str(token_file),
            ]
        )
        rc = cli.cmd_rotate_token(args)

    assert rc == 0
    cfg = _read_json(cfg_path)
    env = cfg["mcpServers"]["muninn"]["env"]
    assert env["MUNINN_AUTH_TOKEN"] == "token-fixed-123"
    assert env["MUNINN_SERVER_URL"] == "http://127.0.0.1:42069"
