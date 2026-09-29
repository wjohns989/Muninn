"""Synthetic-only persistence and portable recovery for the credential vault."""

from __future__ import annotations

import multiprocessing
import shutil
import sqlite3
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
                                        "source_hash": source_fingerprint("test-source"),
                                        "source_hint": ""}]
    assert store.search("missing") == []
    assert _VALUE.encode() not in store.db_path.read_bytes()
    with pytest.raises(VaultIntegrityError):
        store.reveal(record_id, passphrase="wrong-passphrase")
    assert store.reveal(record_id, passphrase=_PASSPHRASE) == _VALUE
    with store._connect(readonly=True) as db:
        assert db.execute("SELECT COUNT(*) FROM reveal_audit").fetchone()[0] == 1


def test_project_relative_env_hint_is_searchable_and_authenticated(tmp_path: Path) -> None:
    store = _new(tmp_path)
    record_id = store.add(
        passphrase=_PASSPHRASE, value=_VALUE, service="openrouter",
        project="example-project", source_hash=source_fingerprint("private-source"),
        source_hint="config/.env.local",
    )
    matches = store.search(".env.local")
    assert matches[0]["source_hint"] == "config/.env.local"
    assert _VALUE not in str(matches)
    assert store.reveal(record_id, passphrase=_PASSPHRASE) == _VALUE
    with store._connect() as db:
        db.execute("UPDATE credentials SET source_hint='.env' WHERE id=?", (record_id,))
    with pytest.raises(VaultIntegrityError):
        store.reveal(record_id, passphrase=_PASSPHRASE)


@pytest.mark.parametrize("hint", [
    "C:/Users/user/.env", "../.env", "config//.env", "config\\.env",
    "/.env", "config/./.env", "config/../.env",
])
def test_credential_source_hint_rejects_unsafe_locations(tmp_path: Path, hint: str) -> None:
    store = _new(tmp_path)
    with pytest.raises(VaultIntegrityError):
        store.add(passphrase=_PASSPHRASE, value=_VALUE, service="example", project="project",
                  source_hash=source_fingerprint("source"), source_hint=hint)


def test_pre_receipt_v2_vault_migrates_without_changing_records(tmp_path: Path) -> None:
    store = _new(tmp_path)
    record_id = _add(store)
    with store._connect() as db:
        db.execute("DROP TABLE scan_receipts")

    migrated = CredentialStore(store.root)

    assert migrated.reveal(record_id, passphrase=_PASSPHRASE) == _VALUE
    with migrated._connect(readonly=True) as db:
        assert db.execute("SELECT count(*) FROM scan_receipts").fetchone()[0] == 0


def test_malformed_receipt_schema_fails_closed(tmp_path: Path) -> None:
    store = _new(tmp_path)
    with store._connect() as db:
        db.execute("DROP TABLE scan_receipts")
        db.execute("CREATE TABLE scan_receipts (receipt_id TEXT)")

    with pytest.raises(VaultIntegrityError, match="receipt schema"):
        CredentialStore(store.root)


def test_pre_receipt_backup_restores_with_empty_receipt_table(tmp_path: Path) -> None:
    store = _new(tmp_path)
    record_id = _add(store)
    backup = tmp_path / "backup"
    assert store.backup(backup, passphrase=_PASSPHRASE) == 1
    with CredentialStore(backup)._connect() as db:
        db.execute("DROP TABLE scan_receipts")

    restored = CredentialStore.restore(backup, tmp_path / "restored", passphrase=_PASSPHRASE)

    assert restored.reveal(record_id, passphrase=_PASSPHRASE) == _VALUE
    with restored._connect(readonly=True) as db:
        assert db.execute("SELECT count(*) FROM scan_receipts").fetchone()[0] == 0
    with sqlite3.connect(backup / "records.db") as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE name='scan_receipts'").fetchone() is None


def test_exact_legacy_schema_migrates_without_changing_record_aad(tmp_path: Path) -> None:
    store = _new(tmp_path)
    record_id = _add(store)
    with store._connect() as db:
        db.execute("CREATE TABLE old_credentials (id TEXT PRIMARY KEY, service TEXT NOT NULL, "
                   "project TEXT NOT NULL, source_hash TEXT NOT NULL, envelope TEXT NOT NULL)")
        db.execute("INSERT INTO old_credentials SELECT id, service, project, source_hash, envelope "
                   "FROM credentials")
        db.execute("DROP TABLE credentials")
        db.execute("ALTER TABLE old_credentials RENAME TO credentials")
        db.execute("PRAGMA user_version=0")
    migrated = CredentialStore(store.root)
    assert migrated.search("example")[0]["source_hint"] == ""
    assert migrated.reveal(record_id, passphrase=_PASSPHRASE) == _VALUE
    assert migrated.backup(tmp_path / "legacy-backup", passphrase=_PASSPHRASE) == 1


def test_unknown_legacy_schema_version_fails_closed(tmp_path: Path) -> None:
    store = _new(tmp_path)
    with store._connect() as db:
        db.execute("CREATE TABLE old_credentials (id TEXT PRIMARY KEY, service TEXT NOT NULL, "
                   "project TEXT NOT NULL, source_hash TEXT NOT NULL, envelope TEXT NOT NULL)")
        db.execute("DROP TABLE credentials")
        db.execute("ALTER TABLE old_credentials RENAME TO credentials")
        db.execute("PRAGMA user_version=99")
    with pytest.raises(VaultIntegrityError, match="schema version"):
        CredentialStore(store.root)


def test_empty_scan_validates_source_and_stales_active_rows(tmp_path: Path) -> None:
    store = _new(tmp_path)
    with pytest.raises(VaultIntegrityError):
        store.scan_source(passphrase=_PASSPHRASE, source_hash="bad", project="test-project", origin="project", findings=[])
    source = source_fingerprint("project/.env")
    store.scan_source(passphrase=_PASSPHRASE, source_hash=source, project="test-project", origin="project", findings=[("service", "value", ".env")])
    assert store.scan_source(passphrase=_PASSPHRASE, source_hash=source, project="test-project", origin="project", findings=[]) == {"created": 0, "rotated": 0, "staled": 1}
    assert store.search("service") == []
    assert store.search("service", active_only=False)[0]["active"] == 0


def test_scan_session_reuses_unlock_and_invalidates_after_exit(tmp_path: Path) -> None:
    store = _new(tmp_path)
    with store.scan_session(_PASSPHRASE) as session:
        for number in range(3):
            result = session.scan_source(
                passphrase=_PASSPHRASE,
                source_hash=source_fingerprint(f"archive/{number}"),
                project="test-project", origin="transcript",
                findings=[("service", f"value-{number}", "")],
            )
            assert result["created"] == 1
    with pytest.raises(VaultIntegrityError, match="closed"):
        session.scan_source(passphrase=_PASSPHRASE, source_hash=source_fingerprint("after"),
                            project="test-project", origin="transcript", findings=[])


@pytest.mark.parametrize("tamper", ["header", "sentinel"])
def test_scan_session_rechecks_vault_each_source_and_rolls_back(tmp_path: Path, tamper: str) -> None:
    store = _new(tmp_path)
    original_header = store.header_path.read_text(encoding="utf-8")
    with store.scan_session(_PASSPHRASE) as session:
        session.scan_source(passphrase=_PASSPHRASE, source_hash=source_fingerprint("first"),
                            project="test-project", origin="transcript",
                            findings=[("first", "value", "")])
        if tamper == "header":
            store.header_path.write_text(original_header.replace(store.header.vault_id, "0" * 32), encoding="utf-8")
        else:
            with store._connect() as db:
                db.execute("DELETE FROM sentinel")
        with pytest.raises(VaultIntegrityError):
            session.scan_source(passphrase=_PASSPHRASE, source_hash=source_fingerprint("second"),
                                project="test-project", origin="transcript",
                                findings=[("second", "value", "")])
    assert store.search("second") == []


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


def test_project_scan_is_idempotent_rotates_and_stales(tmp_path: Path) -> None:
    store = _new(tmp_path)
    source = source_fingerprint("project/.env")
    findings = [("openrouter", "one", ".env"), ("github", "two", "config/.env.local")]
    first = store.scan_source(passphrase=_PASSPHRASE, source_hash=source, project="test-project", origin="project", findings=findings)
    assert first["created"] == 2
    second = store.scan_source(passphrase=_PASSPHRASE, source_hash=source, project="test-project", origin="project", findings=findings)
    assert second == {"created": 0, "rotated": 0, "staled": 0}
    store.scan_source(passphrase=_PASSPHRASE, source_hash=source, project="test-project", origin="project", findings=[findings[0]])
    assert len(store.search("test-project")) == 1
    with store._connect(readonly=True) as db:
        assert db.execute("SELECT COUNT(*) FROM credentials WHERE active=0").fetchone()[0] == 1
    assert store.search("github") == []
    rows = store.search("github", active_only=False)
    assert rows and rows[0]["active"] == 0 and rows[0]["origin"] == "project"
    record_id = rows[0]["id"]
    assert store.reveal(record_id, passphrase=_PASSPHRASE) == "two"
    backup = tmp_path / "stale-backup"
    assert store.backup(backup, passphrase=_PASSPHRASE) == 2
    assert CredentialStore(backup).reveal(record_id, passphrase=_PASSPHRASE) == "two"


def test_transcript_scan_retains_distinct_values_and_tamper_fails(tmp_path: Path) -> None:
    store = _new(tmp_path)
    source = source_fingerprint("transcript/1")
    for value in ("old", "new"):
        store.scan_source(passphrase=_PASSPHRASE, source_hash=source, project="test-project", origin="transcript", findings=[("service", value, "")])
    with store._connect(readonly=True) as db:
        rows = db.execute("SELECT id FROM credentials WHERE origin='transcript'").fetchall()
        assert len(rows) == 2
    with store._connect() as db:
        db.execute("UPDATE credentials SET active=0 WHERE id=?", (rows[0][0],))
    with pytest.raises(VaultIntegrityError):
        store.reveal(rows[0][0], passphrase=_PASSPHRASE)


def test_scan_generator_failure_rolls_back_everything(tmp_path: Path) -> None:
    store = _new(tmp_path)
    source = source_fingerprint("project/.env")

    def failing():
        yield ("first", "secret-1", ".env")
        raise RuntimeError("late archive verification failure")

    with pytest.raises(RuntimeError, match="late archive"):
        store.scan_source(passphrase=_PASSPHRASE, source_hash=source, project="test-project", origin="project", findings=failing())
    assert store.search("first") == []


def test_concurrent_scans_are_idempotent(tmp_path: Path) -> None:
    store = _new(tmp_path)
    source = source_fingerprint("project/.env")
    errors = []

    def run():
        try:
            store.scan_source(passphrase=_PASSPHRASE, source_hash=source, project="test-project", origin="project", findings=[("service", "value", ".env")])
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    with store._connect(readonly=True) as db:
        assert db.execute("SELECT COUNT(*) FROM credentials").fetchone()[0] == 1
