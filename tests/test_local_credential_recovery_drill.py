"""The operator drill uses isolated synthetic data and never needs a live passphrase."""

import json
from types import SimpleNamespace

import pytest

from scripts import run_local_credential_recovery_drill as drill
from muninn.history.private_acl import create_private_directory
from scripts.run_local_credential_recovery_drill import run
from tests.test_credential_store import _PASSPHRASE, _VALUE, _add, _new


def _paths(tmp_path):
    private = tmp_path / "private"
    create_private_directory(private)
    return (private / "backup", private / "restore", private / "status.jsonl")


def test_drill_restores_real_encrypted_tables_and_prompts_once(tmp_path, capsys):
    store = _new(tmp_path)
    ident = _add(store)
    backup, restore, status = _paths(tmp_path)
    prompts = []

    def prompt(label):
        prompts.append(label)
        return _PASSPHRASE

    assert run(store.root, backup, restore, status, prompt=prompt) == 0
    stages = [json.loads(line) for line in status.read_text(encoding="utf-8").splitlines()]
    assert [item["stage"] for item in stages] == [
        "awaiting_passphrase", "backup_started", "backup_validated",
        "restore_started", "restore_validated", "complete",
    ]
    assert stages[4]["credential_records"] == 1
    assert len(prompts) == 1
    assert _VALUE not in status.read_text(encoding="utf-8")
    assert _PASSPHRASE not in capsys.readouterr().out
    assert restore.is_dir() and backup.is_dir()
    from muninn.history.credential_store import CredentialStore
    assert CredentialStore(restore).reveal(ident, passphrase=_PASSPHRASE) == _VALUE


def test_drill_wrong_passphrase_fails_without_publishing_backup(tmp_path):
    store = _new(tmp_path)
    _add(store)
    backup, restore, status = _paths(tmp_path)
    assert run(store.root, backup, restore, status,
               prompt=lambda _: "incorrect passphrase") == 2
    assert not backup.exists() and not restore.exists()
    stages = [json.loads(line)["stage"] for line in status.read_text(encoding="utf-8").splitlines()]
    assert stages == ["awaiting_passphrase", "backup_started", "failed"]


def test_drill_rejects_existing_and_nested_destinations_before_prompt(tmp_path):
    store = _new(tmp_path)
    backup, restore, status = _paths(tmp_path)
    backup.mkdir()
    try:
        run(store.root, backup, restore, status, prompt=lambda _: (_ for _ in ()).throw(AssertionError()))
    except ValueError:
        pass
    else:
        raise AssertionError("Existing backup must be rejected")
    assert not status.exists()


def test_cli_creates_private_drill_root_and_refuses_reuse(tmp_path, monkeypatch):
    store = _new(tmp_path)
    _add(store)
    monkeypatch.setattr(drill.getpass, "getpass", lambda _: _PASSPHRASE)
    monkeypatch.setattr(drill, "_verify_nonreplaceable_ancestors", lambda _: None)
    root = tmp_path / "drill"
    args = ["--vault-root", str(store.root), "--drill-root", str(root)]
    assert drill.main(args) == 0
    assert (root / "status.jsonl").is_file()
    assert drill.main(args) == 2


def test_conditional_allow_ace_fails_closed():
    security = SimpleNamespace(ACCESS_ALLOWED_ACE_TYPE=0,
                               ACCESS_DENIED_ACE_TYPE=1,
                               ConvertSidToStringSid=str)
    with pytest.raises(ValueError, match="unsupported ACL entry"):
        drill._ace_may_replace_child(((9, 0), 0x40, "foreign"), security,
                                     {"owner"}, 0x40, 0x08)
    assert drill._ace_may_replace_child(((0, 0), 0x40, "foreign"), security,
                                        {"owner"}, 0x40, 0x08)
    assert not drill._ace_may_replace_child(((0, 0x08), 0x40, "foreign"), security,
                                            {"owner"}, 0x40, 0x08)
