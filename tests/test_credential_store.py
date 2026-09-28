"""Synthetic-only persistence and portable recovery for the inactive vault."""

from __future__ import annotations

import multiprocessing
import shutil
import threading
import time
from pathlib import Path

import pytest

from muninn.history.credential_crypto import VaultHeader, VaultIntegrityError
from muninn.history.credential_store import CredentialStore, source_fingerprint
from muninn.history.private_acl import VaultPermissionError

_PASSPHRASE = "a local test phrase with enough entropy"
_VALUE = "synthetic-secret-value-918273645"


def _new(tmp_path: Path) -> CredentialStore:
    return CredentialStore.create(tmp_path / "credential-vault", _PASSPHRASE)


def _add(store: CredentialStore) -> str:
    return store.add(passphrase=_PASSPHRASE, value=_VALUE, service="example", project="test-project",
                     source_hash=source_fingerprint("test-source"))


def test_existing_vault_opens_from_relative_data_directory(tmp_path: Path, monkeypatch) -> None:
    _new(tmp_path)
    monkeypatch.chdir(tmp_path)
    store = CredentialStore(Path("credential-vault"))
    assert store.search("example") == []


def _other_process_add(root: str, started, done) -> None:
    store = CredentialStore(Path(root))
    started.set()
    _add(store)
    done.set()


def test_metadata_search_and_explicit_reveal_only(tmp_path: Path) -> None:
    store = _new(tmp_path)
    record_id = _add(store)
    assert store.search("example") == [{"id": record_id, "service": "example", "project": "test-project",
                                        "source_hash": source_fingerprint("test-source")}]
    assert store.search("missing") == []
    assert _VALUE.encode() not in store.db_path.read_bytes()
    with pytest.raises(VaultIntegrityError):
        store.reveal(record_id, passphrase="wrong-passphrase")
    assert store.reveal(record_id, passphrase=_PASSPHRASE) == _VALUE
    with store._connect(readonly=True) as db:
        assert db.execute("SELECT COUNT(*) FROM reveal_audit").fetchone()[0] == 1


def test_header_and_sentinel_tamper_fail_closed(tmp_path: Path) -> None:
    store = _new(tmp_path)
    header = store.header
    bad = VaultHeader(**{**header.__dict__, "vault_id": "0" * 32})
    store.header_path.write_text(bad.to_json(), encoding="utf-8")
    with pytest.raises(VaultIntegrityError):
        store.add(passphrase=_PASSPHRASE, value=_VALUE, service="example", project="test",
                  source_hash=source_fingerprint("source"))
    store.header_path.write_text(header.to_json(), encoding="utf-8")
    with store._connect() as db:
        db.execute("DELETE FROM sentinel")
    with pytest.raises(VaultIntegrityError):
        store.reveal("a" * 32, passphrase=_PASSPHRASE)
    store.header_path.unlink()
    with pytest.raises(VaultPermissionError):
        CredentialStore(store.root)


def test_backup_is_portable_and_detects_invalid_journal(tmp_path: Path) -> None:
    store = _new(tmp_path)
    record_id = _add(store)
    backup = tmp_path / "backup"
    assert store.backup(backup, passphrase=_PASSPHRASE) == 1
    restored = CredentialStore(backup)
    assert restored.reveal(record_id, passphrase=_PASSPHRASE) == _VALUE
    with pytest.raises(VaultIntegrityError):
        store.backup(tmp_path / "bad-backup", passphrase="wrong-passphrase")
    assert not (tmp_path / "bad-backup").exists()
    (backup / "records.db-wal").touch()
    with pytest.raises(VaultIntegrityError):
        CredentialStore(backup)


def test_portable_restore_rehomes_permissive_transfer_copy(tmp_path: Path) -> None:
    store = _new(tmp_path)
    record_id = _add(store)
    store.backup(tmp_path / "backup", passphrase=_PASSPHRASE)
    transfer = tmp_path / "transfer"
    transfer.mkdir()
    shutil.copyfile(tmp_path / "backup" / "header.json", transfer / "header.json")
    shutil.copyfile(tmp_path / "backup" / "records.db", transfer / "records.db")
    with pytest.raises(VaultIntegrityError):
        CredentialStore.restore(transfer, tmp_path / "wrong", passphrase="wrong-passphrase")
    assert not (tmp_path / "wrong").exists()
    restored = CredentialStore.restore(transfer, tmp_path / "restored", passphrase=_PASSPHRASE)
    assert restored.reveal(record_id, passphrase=_PASSPHRASE) == _VALUE


def test_backup_blocks_same_process_write_and_copies_one_snapshot(tmp_path: Path) -> None:
    store = _new(tmp_path)
    first = _add(store)
    entered = threading.Event()
    release = threading.Event()
    writer_done = threading.Event()
    errors: list[BaseException] = []
    real_backup = store._backup_sqlite

    def paused_backup(destination: Path) -> None:
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test backup was not released")
        real_backup(destination)

    store._backup_sqlite = paused_backup  # type: ignore[method-assign]

    def run_backup() -> None:
        try:
            assert store.backup(tmp_path / "backup", passphrase=_PASSPHRASE) == 1
        except BaseException as exc:
            errors.append(exc)

    def run_writer() -> None:
        try:
            _add(store)
            writer_done.set()
        except BaseException as exc:
            errors.append(exc)

    backup_thread = threading.Thread(target=run_backup)
    backup_thread.start()
    assert entered.wait(5)
    writer_thread = threading.Thread(target=run_writer)
    writer_thread.start()
    assert not writer_done.wait(0.1)
    release.set()
    backup_thread.join(timeout=5)
    writer_thread.join(timeout=5)
    assert not backup_thread.is_alive() and not writer_thread.is_alive()
    assert not errors
    assert writer_done.is_set()
    restored = CredentialStore(tmp_path / "backup")
    assert [r["id"] for r in restored.search("example")] == [first]


def test_failed_backup_does_not_publish_destination(tmp_path: Path) -> None:
    store = _new(tmp_path)
    _add(store)

    def fail_backup(_destination: Path) -> None:
        raise OSError("injected backup failure")

    store._backup_sqlite = fail_backup  # type: ignore[method-assign]
    destination = tmp_path / "backup"
    with pytest.raises(OSError, match="injected backup failure"):
        store.backup(destination, passphrase=_PASSPHRASE)
    assert not destination.exists()
    incomplete = list(tmp_path.glob(".backup.incomplete-*"))
    assert len(incomplete) == 1
    with pytest.raises(VaultIntegrityError):
        store.backup(incomplete[0], passphrase=_PASSPHRASE)


def test_backup_blocks_second_process_writer(tmp_path: Path) -> None:
    store = _new(tmp_path)
    first = _add(store)
    entered = threading.Event()
    release = threading.Event()
    real_backup = store._backup_sqlite
    errors: list[BaseException] = []

    def paused_backup(destination: Path) -> None:
        entered.set()
        if not release.wait(10):
            raise TimeoutError("test backup was not released")
        real_backup(destination)

    store._backup_sqlite = paused_backup  # type: ignore[method-assign]

    def run_backup() -> None:
        try:
            assert store.backup(tmp_path / "backup", passphrase=_PASSPHRASE) == 1
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_backup)
    thread.start()
    assert entered.wait(5)
    context = multiprocessing.get_context("spawn")
    started, done = context.Event(), context.Event()
    process = context.Process(target=_other_process_add, args=(str(store.root), started, done))
    process.start()
    try:
        assert started.wait(10)
        time.sleep(0.2)
        assert not done.is_set()
    finally:
        release.set()
        thread.join(timeout=10)
        process.join(timeout=10)
    assert not thread.is_alive() and not process.is_alive()
    assert process.exitcode == 0 and not errors
    assert done.is_set()
    restored = CredentialStore(tmp_path / "backup")
    assert [r["id"] for r in restored.search("example")] == [first]


def test_inherited_directory_cannot_be_opened_as_vault(tmp_path: Path) -> None:
    root = tmp_path / "inherited"
    root.mkdir()
    with pytest.raises(VaultPermissionError):
        CredentialStore.create(root, _PASSPHRASE)
