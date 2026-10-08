"""Isolated recovery regressions; no live vault, credentials, or lifecycle."""
import os
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

import pytest

from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.credential_store import CredentialStore
from tests.test_credential_store import _PASSPHRASE, _VALUE, _add, _new


def test_backup_streams_record_validation_without_fetchall(tmp_path, monkeypatch):
    store = _new(tmp_path)
    _add(store)
    original = store._connect

    class Cursor:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __iter__(self):
            return iter(self.wrapped)

        def fetchall(self):
            pytest.fail("Recovery must not materialize the entire credential queue")

    class Connection:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def execute(self, query, *args):
            result = self.wrapped.execute(query, *args)
            if query.startswith(("SELECT * FROM credentials", "SELECT * FROM ambiguity_queue")):
                return Cursor(result)
            return result

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    @contextmanager
    def connected(**kwargs):
        with original(**kwargs) as db:
            yield Connection(db)

    monkeypatch.setattr(store, "_connect", connected)
    assert store.backup(tmp_path / "backup", passphrase=_PASSPHRASE) == 1


@pytest.mark.parametrize("table", ["reveal_audit", "scan_receipts", "ambiguity_audit", "sqlite_sequence"])
@pytest.mark.parametrize("comparison_enabled", [True, False])
def test_backup_detects_lost_control_rows_in_copied_snapshot(tmp_path, monkeypatch, table, comparison_enabled):
    store = _new(tmp_path)
    ident = _add(store)
    store.reveal(ident, passphrase=_PASSPHRASE)
    with store._connect() as db:
        db.execute("INSERT INTO scan_receipts VALUES('fixture-receipt',1,0)")
        db.execute("INSERT INTO ambiguity_audit(ambiguity_id,action,actor,at) "
                   "VALUES('fixture-id','rejected','fixture',0)")
    original = store._backup_sqlite

    def torn(destination):
        original(destination)
        with closing(sqlite3.connect(destination)) as db:
            with db:
                db.execute(f"DELETE FROM {table}")

    monkeypatch.setattr(store, "_backup_sqlite", torn)
    destination = tmp_path / "backup"
    if comparison_enabled:
        with pytest.raises(VaultIntegrityError, match="differs"):
            store.backup(destination, passphrase=_PASSPHRASE)
        assert not destination.exists()
    else:
        # Targeted isolated old-defect mutation: without the comparison,
        # valid credential ciphertext can hide lost receipts/audits/sequences.
        monkeypatch.setattr(CredentialStore, "_compare_recovery_snapshot", staticmethod(lambda *args: None))
        assert store.backup(destination, passphrase=_PASSPHRASE) == 1


def test_backup_rejects_uncatalogued_persistent_table(tmp_path):
    store = _new(tmp_path)
    with store._connect() as db:
        db.execute("CREATE TABLE future_control(value TEXT)")
    with pytest.raises(VaultIntegrityError, match="schema"):
        store.backup(tmp_path / "backup", passphrase=_PASSPHRASE)
    assert not (tmp_path / "backup").exists()


def test_restore_has_no_arbitrary_database_byte_ceiling(tmp_path, monkeypatch):
    store = _new(tmp_path)
    ident = _add(store)
    backup = tmp_path / "backup"
    store.backup(backup, passphrase=_PASSPHRASE)
    original = Path.stat

    def stat(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path == backup / "records.db":
            fields = list(result)
            fields[6] = 1_073_741_825
            return os.stat_result(fields)
        return result

    monkeypatch.setattr(Path, "stat", stat)
    # Size-gate sensitivity only, not a claim of measured GiB transfer speed.
    restored = CredentialStore.restore(backup, tmp_path / "restored", passphrase=_PASSPHRASE)
    assert restored.reveal(ident, passphrase=_PASSPHRASE) == _VALUE


def test_restore_uses_sqlite_snapshot_not_raw_database_copy(tmp_path, monkeypatch):
    from muninn.history import credential_store

    store = _new(tmp_path)
    ident = _add(store)
    backup = tmp_path / "backup"
    store.backup(backup, passphrase=_PASSPHRASE)
    original = credential_store.shutil.copyfileobj

    def copy(read, write, *args):
        assert Path(read.name).name != "records.db", "A live SQLite file is not a consistent snapshot"
        return original(read, write, *args)

    monkeypatch.setattr(credential_store.shutil, "copyfileobj", copy)
    restored = CredentialStore.restore(backup, tmp_path / "restored", passphrase=_PASSPHRASE)
    assert restored.reveal(ident, passphrase=_PASSPHRASE) == _VALUE


def test_restore_does_no_fallible_reopen_after_publication(tmp_path, monkeypatch):
    store = _new(tmp_path)
    ident = _add(store)
    backup, destination = tmp_path / "backup", tmp_path / "restored"
    store.backup(backup, passphrase=_PASSPHRASE)
    original = CredentialStore.__init__

    def init(self, root):
        if Path(root) == destination:
            raise OSError("Injected post-publication reopen failure")
        original(self, root)

    monkeypatch.setattr(CredentialStore, "__init__", init)
    restored = CredentialStore.restore(backup, destination, passphrase=_PASSPHRASE)
    assert restored.root == destination
    assert restored.header_path.parent == restored.db_path.parent == restored.lock_path.parent == destination
    assert restored.reveal(ident, passphrase=_PASSPHRASE) == _VALUE


@pytest.mark.parametrize("action", ["backup", "restore"])
def test_raced_empty_destination_is_preserved(tmp_path, monkeypatch, action):
    store = _new(tmp_path)
    _add(store)
    backup, destination = tmp_path / "backup", tmp_path / "destination"
    if action == "restore":
        store.backup(backup, passphrase=_PASSPHRASE)
    original = Path.exists
    checks = 0

    def exists(path):
        nonlocal checks
        if path == destination:
            checks += 1
            if checks == 2:
                destination.mkdir()
                return False  # Competing owner appears just after the last check.
        return original(path)

    monkeypatch.setattr(Path, "exists", exists)
    with pytest.raises((FileExistsError, VaultIntegrityError)):
        if action == "backup":
            store.backup(destination, passphrase=_PASSPHRASE)
        else:
            CredentialStore.restore(backup, destination, passphrase=_PASSPHRASE)
    assert destination.is_dir() and list(destination.iterdir()) == []
    assert len(list(tmp_path.glob(".destination.incomplete-*"))) == 1
