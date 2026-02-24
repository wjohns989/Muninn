import json
from pathlib import Path

from muninn.cli import _patch_mcp_config


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
