"""Synthetic local credential CLI behavior; no real secret or data directory."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import pytest

from muninn.cli import build_parser, cmd_credentials
from muninn.history.credential_store import CredentialStore, source_fingerprint

_PASSPHRASE = "synthetic long local passphrase"
_VALUE = "synthetic-cli-secret-12345"


class _TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def _args(action: str, root: Path, **overrides) -> argparse.Namespace:
    values = {"action": action, "root": root, "query": None, "record_id": None,
              "destination": None, "source": None}
    values.update(overrides)
    return argparse.Namespace(**values)


def test_credential_cli_requires_terminal_for_unlock(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="interactive local terminal"):
        cmd_credentials(_args("init", tmp_path / "vault"))
    assert not (tmp_path / "vault").exists()


def test_credential_cli_search_reveal_backup_restore(tmp_path: Path, monkeypatch) -> None:
    output = _TTY()
    monkeypatch.setattr(sys, "stdin", _TTY())
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr("getpass.getpass", lambda _prompt: _PASSPHRASE)
    root = tmp_path / "vault"
    assert cmd_credentials(_args("init", root)) == 0
    store = CredentialStore(root)
    record_id = store.add(passphrase=_PASSPHRASE, value=_VALUE, service="example",
                          project="test", source_hash=source_fingerprint("source"))
    output.seek(0)
    output.truncate(0)
    assert cmd_credentials(_args("search", root, query="example")) == 0
    assert _VALUE not in output.getvalue()
    assert json.loads(output.getvalue())[0]["id"] == record_id
    output.seek(0)
    output.truncate(0)
    assert cmd_credentials(_args("reveal", root, record_id=record_id)) == 0
    assert output.getvalue().strip() == _VALUE
    backup = tmp_path / "backup"
    assert cmd_credentials(_args("backup", root, destination=backup)) == 0
    restored = tmp_path / "restored"
    assert cmd_credentials(_args("restore", restored, source=backup)) == 0
    assert CredentialStore(restored).reveal(record_id, passphrase=_PASSPHRASE) == _VALUE


def test_credential_parser_has_no_passphrase_argv_option() -> None:
    parser = build_parser()
    args = parser.parse_args(["credentials", "reveal", "--record-id", "a" * 32])
    assert args.action == "reveal"
    with pytest.raises(SystemExit):
        parser.parse_args(["credentials", "init", "--passphrase", "do-not-put-secrets-in-argv"])
