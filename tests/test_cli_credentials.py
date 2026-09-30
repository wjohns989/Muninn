"""Synthetic local credential CLI behavior; no real secret or data directory."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import pytest

import muninn.history.credential_discovery as discovery
from muninn.cli import build_parser, cmd_credentials
from muninn.history.credential_store import AmbiguousCandidate, CredentialStore, source_fingerprint
from muninn.history.secure_archive import SecureHistoryArchive

_PASSPHRASE = "synthetic long local passphrase"
_VALUE = "synthetic-cli-secret-12345"


class _TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def _args(action: str, root: Path, **overrides) -> argparse.Namespace:
    values = {"action": action, "root": root, "query": None, "record_id": None,
              "destination": None, "source": None, "project_root": None,
              "archive_root": None, "archive_offset": 0, "archive_generation": None,
              "max_snapshots": None, "backup_before": None,
              "review_state": "pending", "confirm_exact_candidate": False,
              "confirm_not_credential": False}
    values.update(overrides)
    return argparse.Namespace(**values)


def test_credential_cli_requires_terminal_for_unlock(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="interactive local terminal"):
        cmd_credentials(_args("init", tmp_path / "vault"))
    assert not (tmp_path / "vault").exists()


def test_review_status_is_nonsecret_and_does_not_require_terminal(tmp_path: Path, capsys) -> None:
    root = tmp_path / "vault"
    CredentialStore.create(root, _PASSPHRASE)
    assert cmd_credentials(_args("review-status", root)) == 0
    assert json.loads(capsys.readouterr().out) == {}


def test_local_ambiguity_review_requires_explicit_confirmation(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "vault"
    store = CredentialStore.create(root, _PASSPHRASE)
    store.scan_source(
        passphrase=_PASSPHRASE, source_hash=source_fingerprint("review-source"),
        project="test", origin="project",
        findings=[AmbiguousCandidate("SERVICE_API_KEY", "unparsed_value", _VALUE, ".env")],
    )
    ambiguity_id = store.list_ambiguities()[0]["id"]
    output = _TTY()
    monkeypatch.setattr(sys, "stdin", _TTY())
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr("getpass.getpass", lambda _prompt: _PASSPHRASE)
    assert cmd_credentials(_args("review-list", root)) == 0
    assert _VALUE not in output.getvalue()
    assert json.loads(output.getvalue())[0]["id"] == ambiguity_id
    output.seek(0)
    output.truncate(0)
    with pytest.raises(SystemExit, match="confirmation"):
        cmd_credentials(_args("review-accept", root, record_id=ambiguity_id))
    assert store.ambiguity_status() == {"pending": 1}
    backup = tmp_path / "before-accept"
    assert cmd_credentials(_args("review-accept", root, record_id=ambiguity_id,
                                 confirm_exact_candidate=True, backup_before=backup)) == 0
    assert CredentialStore(backup).ambiguity_status() == {"pending": 1}
    assert store.ambiguity_status() == {"accepted": 1}
    assert _VALUE not in output.getvalue()


def test_local_user_can_reject_deferred_ambiguity_with_backup(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "vault"
    store = CredentialStore.create(root, _PASSPHRASE)
    store.scan_source(
        passphrase=_PASSPHRASE, source_hash=source_fingerprint("deferred-source"),
        project="test", origin="transcript",
        findings=[AmbiguousCandidate("SERVICE_API_KEY", "unparsed_value", _VALUE, "")],
    )
    ambiguity_id = store.list_ambiguities()[0]["id"]
    store.decide_ambiguity(ambiguity_id, passphrase=_PASSPHRASE, decision="deferred")
    output = _TTY()
    monkeypatch.setattr(sys, "stdin", _TTY())
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr("getpass.getpass", lambda _prompt: _PASSPHRASE)
    with pytest.raises(SystemExit, match="confirmation"):
        cmd_credentials(_args("review-reject", root, record_id=ambiguity_id))
    backup = tmp_path / "before-reject"
    assert cmd_credentials(_args("review-reject", root, record_id=ambiguity_id,
                                 confirm_not_credential=True, backup_before=backup)) == 0
    assert CredentialStore(backup).ambiguity_status() == {"deferred": 1}
    assert store.ambiguity_status() == {"rejected": 1}
    assert store.search("SERVICE_API_KEY") == []
    assert _VALUE not in output.getvalue()


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
    review = parser.parse_args([
        "credentials", "review-accept", "--record-id", "a" * 32,
        "--confirm-exact-candidate", "--backup-before", "private-backup",
    ])
    assert review.action == "review-accept" and review.confirm_exact_candidate
    with pytest.raises(SystemExit):
        parser.parse_args(["credentials", "init", "--passphrase", "do-not-put-secrets-in-argv"])


def test_credential_cli_scans_selected_project_without_printing_value(tmp_path: Path, monkeypatch) -> None:
    output = _TTY()
    monkeypatch.setattr(sys, "stdin", _TTY())
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr("getpass.getpass", lambda _prompt: _PASSPHRASE)
    root = tmp_path / "vault"
    CredentialStore.create(root, _PASSPHRASE)
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("SERVICE_API_KEY=aaaabbbbcccc11112222\n")
    (project / "config.yaml").write_text("SERVICE_AUTH_TOKEN: yamlVALUE12345678\n")
    backup = tmp_path / "before-scan"
    result_code = cmd_credentials(_args("scan", root, project_root=[project], backup_before=backup))
    assert result_code == 0, output.getvalue().splitlines()[-1]
    assert CredentialStore(backup).search("SERVICE_API_KEY") == []
    report = json.loads(output.getvalue().splitlines()[-1])
    assert report["complete"] is True
    assert report["project"]["inserted"] == 2
    assert "aaaabbbbcccc11112222" not in output.getvalue()
    assert CredentialStore(root).search("SERVICE_API_KEY")[0]["source_hint"] == ".env"
    assert CredentialStore(root).search("SERVICE_AUTH_TOKEN")[0]["source_hint"] == "config.yaml"


def test_credential_cli_reports_binary_coverage_gap_without_source_details(tmp_path: Path, monkeypatch) -> None:
    output = _TTY()
    monkeypatch.setattr(sys, "stdin", _TTY())
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr("getpass.getpass", lambda _prompt: _PASSPHRASE)
    root = tmp_path / "vault"
    CredentialStore.create(root, _PASSPHRASE)
    project = tmp_path / "project"
    project.mkdir()
    (project / "private-source.py").write_bytes(b"\x00\x05\x16\x07" + b"\x00" * 4096)

    code = cmd_credentials(_args("scan", root, project_root=[project]))
    report = json.loads(output.getvalue().splitlines()[-1])
    assert code == 2
    assert report["complete"] is False
    assert report["project"]["error_categories"]["unsupported_binary"] == 1
    assert report["project"]["errors"] == 1
    assert "private-source.py" not in output.getvalue()


def test_walk_error_does_not_prevent_archive_phase(tmp_path: Path, monkeypatch) -> None:
    output = _TTY()
    monkeypatch.setattr(sys, "stdin", _TTY())
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr("getpass.getpass", lambda _prompt: _PASSPHRASE)
    root = tmp_path / "vault"
    CredentialStore.create(root, _PASSPHRASE)
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("SERVICE_API_KEY=aaaabbbbcccc11112222\n")
    archive = SecureHistoryArchive.create(tmp_path / "archive", "synthetic archive passphrase")
    source = tmp_path / "chat.jsonl"
    source.write_text('{"content":"ARCHIVE_API_KEY=archiveVALUE12345678"}\n')
    archive.archive_file(source, "codex")
    monkeypatch.setattr("muninn.history.secure_archive.SecureHistoryArchive", lambda _root: archive)

    def walk(_root, *, followlinks, onerror):
        yield str(project), [], [".env"]
        onerror(OSError("private inaccessible child"))

    monkeypatch.setattr(discovery.os, "walk", walk)
    code = cmd_credentials(_args("scan", root, project_root=[project, tmp_path / "missing"],
                                 archive_root=archive.root))

    report = json.loads(output.getvalue().splitlines()[-1])
    assert code == 2
    assert report["complete"] is False
    assert report["project"]["walk_errors"] == 2
    assert report["project"]["errors"] == 2
    assert report["project"]["error_categories"]["walk"] == 1
    assert report["project"]["error_categories"]["root"] == 1
    assert sum(report["project"]["error_categories"].values()) == 2
    assert report["archive"]["succeeded"] == 1
    assert "private inaccessible child" not in output.getvalue()
