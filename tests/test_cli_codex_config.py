from pathlib import Path

from muninn.cli import _collect_codex_muninn_entries, _patch_codex_toml


def test_patch_streamable_http_uses_supported_codex_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[mcp_servers.muninn]",
                'url = "http://127.0.0.1:41000/mcp"',
                "startup_timeout_sec = 60.0",
                "",
                "[mcp_servers.other]",
                'command = "other"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    changed = _patch_codex_toml(
        config_path,
        new_token="private-token-must-not-be-serialized",
        new_server_url="http://127.0.0.1:42069",
    )

    updated = config_path.read_text(encoding="utf-8")
    assert changed is True
    assert 'url = "http://127.0.0.1:42069/mcp"' in updated
    assert 'bearer_token_env_var = "MUNINN_AUTH_TOKEN"' in updated
    assert "[mcp_servers.muninn.env]" not in updated
    assert "private-token-must-not-be-serialized" not in updated
    assert '[mcp_servers.other]\ncommand = "other"' in updated


def test_patch_streamable_http_removes_legacy_env_table(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[mcp_servers.muninn]",
                'url = "http://127.0.0.1:42069/mcp"',
                'bearer_token_env_var = "MUNINN_AUTH_TOKEN"',
                "",
                "[mcp_servers.muninn.env]",
                'MUNINN_AUTH_TOKEN = "legacy-secret"',
                'MUNINN_SERVER_URL = "http://127.0.0.1:42069"',
                "",
                "[mcp_servers.other]",
                'command = "other"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    changed = _patch_codex_toml(
        config_path,
        new_token="replacement-secret",
        new_server_url="http://127.0.0.1:42069",
    )

    updated = config_path.read_text(encoding="utf-8")
    assert changed is True
    assert "[mcp_servers.muninn.env]" not in updated
    assert "legacy-secret" not in updated
    assert "replacement-secret" not in updated
    assert '[mcp_servers.other]\ncommand = "other"' in updated


def test_patch_stdio_preserves_env_table_support(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[mcp_servers.muninn]\n"
        'command = "python"\n'
        'args = ["server.py"]\n',
        encoding="utf-8",
    )

    changed = _patch_codex_toml(
        config_path,
        new_token="stdio-token",
        new_server_url="http://127.0.0.1:42069",
    )

    updated = config_path.read_text(encoding="utf-8")
    assert changed is True
    assert "[mcp_servers.muninn.env]" in updated
    assert 'MUNINN_AUTH_TOKEN = "stdio-token"' in updated
    assert 'MUNINN_SERVER_URL = "http://127.0.0.1:42069"' in updated


def test_collect_streamable_http_resolves_bearer_env_and_server_url(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[mcp_servers.muninn]\n"
        'url = "http://127.0.0.1:42069/mcp"\n'
        'bearer_token_env_var = "MUNINN_AUTH_TOKEN"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("MUNINN_AUTH_TOKEN", "resolved-token")

    entries = _collect_codex_muninn_entries(config_path)

    assert len(entries) == 1
    assert entries[0].token == "resolved-token"
    assert entries[0].server_url == "http://127.0.0.1:42069"
