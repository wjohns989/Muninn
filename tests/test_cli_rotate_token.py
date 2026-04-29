import argparse
import json
from pathlib import Path
from unittest.mock import patch

from muninn.cli import _patch_mcp_config, cmd_rotate_token


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_patch_mcp_config_injects_token_for_mcpservers_schema(tmp_path: Path):
    cfg_path = tmp_path / "mcp.json"
    _write_json(
        cfg_path,
        {
            "mcpServers": {
                "muninn": {
                    "command": "python",
                    "args": ["mcp_wrapper.py"],
                    "env": {},
                }
            }
        },
    )

    changed = _patch_mcp_config(cfg_path, "tok-123", dry_run=False)
    assert changed is True
    cfg = _read_json(cfg_path)
    assert cfg["mcpServers"]["muninn"]["env"]["MUNINN_AUTH_TOKEN"] == "tok-123"


def test_patch_mcp_config_updates_token_for_servers_schema(tmp_path: Path):
    cfg_path = tmp_path / "settings.json"
    _write_json(
        cfg_path,
        {
            "servers": {
                "muninn-main": {
                    "command": "python",
                    "args": ["mcp_wrapper.py"],
                    "env": {"MUNINN_AUTH_TOKEN": "old-token"},
                }
            }
        },
    )

    changed = _patch_mcp_config(cfg_path, "new-token", dry_run=False)
    assert changed is True
    cfg = _read_json(cfg_path)
    assert cfg["servers"]["muninn-main"]["env"]["MUNINN_AUTH_TOKEN"] == "new-token"


def test_patch_mcp_config_returns_false_without_muninn_server(tmp_path: Path):
    cfg_path = tmp_path / "mcp.json"
    _write_json(
        cfg_path,
        {
            "mcpServers": {
                "other": {
                    "command": "python",
                    "args": ["other_wrapper.py"],
                    "env": {},
                }
            }
        },
    )

    changed = _patch_mcp_config(cfg_path, "token", dry_run=False)
    assert changed is False


def test_cmd_rotate_token_happy_path(tmp_path: Path):
    token_file = tmp_path / ".muninn_token"
    mcp_config_path = tmp_path / "mcp.json"
    _write_json(
        mcp_config_path,
        {
            "mcpServers": {
                "muninn": {
                    "command": "python",
                    "args": ["mcp_wrapper.py"],
                    "env": {},
                }
            }
        },
    )

    args = argparse.Namespace(
        token_file=token_file,
        dry_run=False,
        token_only=False,
    )

    with (
        patch("muninn.cli._MCP_CONFIG_PATHS", [mcp_config_path]),
        patch("muninn.cli.secrets.token_urlsafe", return_value="mocked-token-123"),
    ):
        exit_code = cmd_rotate_token(args)

    assert exit_code == 0
    assert token_file.exists()
    assert token_file.read_text(encoding="utf-8") == "mocked-token-123"

    # Check config was patched
    cfg = _read_json(mcp_config_path)
    assert cfg["mcpServers"]["muninn"]["env"]["MUNINN_AUTH_TOKEN"] == "mocked-token-123"


def test_cmd_rotate_token_token_only(tmp_path: Path, capsys):
    token_file = tmp_path / ".muninn_token"
    args = argparse.Namespace(
        token_file=token_file,
        dry_run=False,
        token_only=True,
    )

    with patch("muninn.cli.secrets.token_urlsafe", return_value="mocked-token-456"):
        exit_code = cmd_rotate_token(args)

    assert exit_code == 0
    assert token_file.exists()
    assert token_file.read_text(encoding="utf-8") == "mocked-token-456"

    captured = capsys.readouterr()
    assert captured.out.strip() == "mocked-token-456"


def test_cmd_rotate_token_dry_run(tmp_path: Path):
    token_file = tmp_path / ".muninn_token"
    mcp_config_path = tmp_path / "mcp.json"
    _write_json(
        mcp_config_path,
        {
            "mcpServers": {
                "muninn": {
                    "command": "python",
                    "args": ["mcp_wrapper.py"],
                    "env": {},
                }
            }
        },
    )

    args = argparse.Namespace(
        token_file=token_file,
        dry_run=True,
        token_only=False,
    )

    with (
        patch("muninn.cli._MCP_CONFIG_PATHS", [mcp_config_path]),
        patch("muninn.cli.secrets.token_urlsafe", return_value="mocked-token-789"),
    ):
        exit_code = cmd_rotate_token(args)

    assert exit_code == 0
    assert not token_file.exists()

    # Check config was not patched
    cfg = _read_json(mcp_config_path)
    assert "MUNINN_AUTH_TOKEN" not in cfg["mcpServers"]["muninn"]["env"]
