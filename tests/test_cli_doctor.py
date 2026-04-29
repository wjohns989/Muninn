import json
from pathlib import Path
from unittest.mock import patch

import muninn.cli as cli


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


def test_doctor_health_check_failure(tmp_path, monkeypatch):
    token_file = tmp_path / ".muninn_token"
    token_file.write_text("expected-token", encoding="utf-8")
    monkeypatch.setattr(cli, "_MCP_CONFIG_PATHS", [])
    monkeypatch.setattr(cli, "_check_server_health", lambda url, token, timeout: (False, "error"))

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
    assert rc == 2


def test_doctor_handles_missing_and_bad_configs(tmp_path, monkeypatch):
    token_file = tmp_path / ".muninn_token"
    token_file.write_text("expected-token", encoding="utf-8")

    non_existent = tmp_path / "does_not_exist.json"
    bad_config = tmp_path / "bad.json"
    bad_config.write_text("not-json", encoding="utf-8")

    monkeypatch.setattr(cli, "_MCP_CONFIG_PATHS", [non_existent, bad_config])
    monkeypatch.setattr(cli, "_check_server_health", lambda url, token, timeout: (True, "ok"))

    def mock_collect(path):
        if path == bad_config:
            raise RuntimeError("mock error")
        return []

    monkeypatch.setattr(cli, "_collect_muninn_server_entries", mock_collect)

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
    # warnings populated due to bad_config parsing failure
    assert rc == 1


def test_doctor_missing_url_in_entry(tmp_path, monkeypatch):
    token_file = tmp_path / ".muninn_token"
    token_file.write_text("expected-token", encoding="utf-8")
    cfg_path = tmp_path / "mcp.json"
    cfg_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(cli, "_MCP_CONFIG_PATHS", [cfg_path])
    monkeypatch.setattr(cli, "_check_server_health", lambda url, token, timeout: (True, "ok"))

    entry = cli._DoctorServerEntry(
        config_path=cfg_path,
        server_name="muninn",
        token="expected-token",
        server_url=None,
    )
    monkeypatch.setattr(cli, "_collect_muninn_server_entries", lambda x: [entry])

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


def test_doctor_repairs_codex_toml(tmp_path, monkeypatch):
    token_file = tmp_path / ".muninn_token"
    token_file.write_text("expected-token", encoding="utf-8")
    codex_path = tmp_path / "config.toml"

    monkeypatch.setattr(cli, "_MCP_CONFIG_PATHS", [])
    monkeypatch.setattr(cli, "_CODEX_CONFIG_PATH", codex_path)
    monkeypatch.setattr(cli, "_check_server_health", lambda url, token, timeout: (True, "ok"))
    monkeypatch.setattr(cli, "_patch_codex_toml", lambda path, new_token, new_server_url, dry_run: True)

    args = cli.build_parser().parse_args(
        [
            "doctor",
            "--token-file",
            str(token_file),
            "--server-url",
            "http://127.0.0.1:42069",
            "--repair",
        ]
    )
    rc = cli.cmd_doctor(args)
    assert rc == 0


def test_doctor_unrepairable_drift(tmp_path, monkeypatch):
    token_file = tmp_path / ".muninn_token"
    token_file.write_text("expected-token", encoding="utf-8")
    cfg_path = tmp_path / "mcp.json"
    cfg_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(cli, "_MCP_CONFIG_PATHS", [cfg_path])
    monkeypatch.setattr(cli, "_check_server_health", lambda url, token, timeout: (True, "ok"))
    monkeypatch.setattr(cli, "_patch_mcp_config_env", lambda *args, **kwargs: True)

    entry1 = cli._DoctorServerEntry(
        config_path=cfg_path,
        server_name="muninn1",
        token="wrong-token",
        server_url="http://127.0.0.1:42069",
    )
    entry2 = cli._DoctorServerEntry(
        config_path=cfg_path,
        server_name="muninn2",
        token="expected-token",
        server_url="http://wrong-url",
    )
    entry3 = cli._DoctorServerEntry(
        config_path=cfg_path,
        server_name="muninn3",
        token="expected-token",
        server_url=None,
    )

    monkeypatch.setattr(cli, "_collect_muninn_server_entries", lambda x: [entry1, entry2, entry3])

    args = cli.build_parser().parse_args(
        [
            "doctor",
            "--token-file",
            str(token_file),
            "--server-url",
            "http://127.0.0.1:42069",
            "--repair",
        ]
    )
    rc = cli.cmd_doctor(args)
    assert rc == 1


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
