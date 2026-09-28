"""Inactive, portable credential store; not connected to ingestion or APIs.

Only minimized metadata is searchable. Values never enter the normal Muninn
database, vector/graph indexes, or model prompts through this module.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from muninn.history.credential_crypto import (
    EncryptedValue,
    VaultHeader,
    VaultIntegrityError,
    decrypt_record,
    derive_key,
    encrypt_record,
)
from muninn.history.private_acl import create_private_directory, create_private_file, verify_private

_SENTINEL_ID = "__vault_sentinel__"
_SENTINEL_VALUE = "muninn-credential-vault-v1"
_SAFE_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}\Z")
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")


def _metadata(service: str, project: str, source_hash: str) -> dict[str, str]:
    if not isinstance(service, str) or not _SAFE_LABEL.fullmatch(service):
        raise VaultIntegrityError("Invalid credential metadata")
    if not isinstance(project, str) or not _SAFE_LABEL.fullmatch(project):
        raise VaultIntegrityError("Invalid credential metadata")
    if not isinstance(source_hash, str) or not _SHA256.fullmatch(source_hash):
        raise VaultIntegrityError("Invalid credential metadata")
    return {"service": service, "project": project, "source_hash": source_hash}


class CredentialStore:
    """Closed-by-default encrypted records, no long-lived passphrase or key."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.header_path = self.root / "header.json"
        self.db_path = self.root / "records.db"
        self.lock_path = self.root / "vault.lock"
        for path in (self.root, self.header_path, self.db_path, self.lock_path):
            verify_private(path)
        self.header = VaultHeader.from_json(self.header_path.read_text(encoding="utf-8"))
        self._lock = threading.RLock()
        self._check_schema()

    @classmethod
    def create(cls, root: Path, passphrase: str) -> CredentialStore:
        root = Path(root)
        create_private_directory(root)
        header = VaultHeader.new()
        key = derive_key(passphrase, header)
        header_path = root / "header.json"
        db_path = root / "records.db"
        lock_path = root / "vault.lock"
        create_private_file(header_path)
        create_private_file(db_path)
        create_private_file(lock_path)
        lock_path.write_bytes(b"\0")
        header_path.write_text(header.to_json(), encoding="utf-8")
        db = sqlite3.connect(db_path)
        try:
            with db:
                db.execute("PRAGMA journal_mode=DELETE")
                cls._create_schema(db)
                sentinel_meta = {"header": header.to_json()}
                sentinel = encrypt_record(key, header, _SENTINEL_ID, sentinel_meta, _SENTINEL_VALUE)
                db.execute("INSERT INTO sentinel (id, envelope) VALUES (1, ?)", (sentinel.to_json(),))
        finally:
            db.close()
        return cls(root)

    @staticmethod
    def _create_schema(db: sqlite3.Connection) -> None:
        db.execute("CREATE TABLE sentinel (id INTEGER PRIMARY KEY CHECK(id=1), envelope TEXT NOT NULL)")
        db.execute("CREATE TABLE credentials (id TEXT PRIMARY KEY, service TEXT NOT NULL, "
                   "project TEXT NOT NULL, source_hash TEXT NOT NULL, envelope TEXT NOT NULL)")
        db.execute("CREATE TABLE reveal_audit (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                   "record_id TEXT NOT NULL, at REAL NOT NULL)")

    @contextmanager
    def _connect(self, *, readonly: bool = False):
        verify_private(self.root)
        verify_private(self.header_path)
        verify_private(self.db_path)
        mode = "ro" if readonly else "rw"
        db = sqlite3.connect(f"{self.db_path.as_uri()}?mode={mode}", uri=True, timeout=10)
        try:
            db.row_factory = sqlite3.Row
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def _process_lock(self):
        """Exclusive inter-process lock for writes and backup snapshots."""
        verify_private(self.lock_path)
        with self.lock_path.open("r+b") as handle:
            if os.name == "nt":
                import msvcrt

                deadline = time.monotonic() + 60
                while True:
                    try:
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as exc:
                        if time.monotonic() >= deadline:
                            raise VaultIntegrityError("Credential vault is busy") from exc
                        time.sleep(0.05)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def _check_schema(self) -> None:
        with self._connect(readonly=True) as db:
            if db.execute("PRAGMA journal_mode").fetchone()[0].lower() != "delete":
                raise VaultIntegrityError("Unsupported credential vault journal mode")
            names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {"sentinel", "credentials", "reveal_audit"} <= names:
                raise VaultIntegrityError("Incomplete credential vault database")
        if self.db_path.with_name("records.db-wal").exists():
            raise VaultIntegrityError("Credential vault WAL must be recovered before opening")

    def _unlock(self, db: sqlite3.Connection, passphrase: str) -> bytes:
        # Re-read the header so a changed header cannot be ignored by an open instance.
        header = VaultHeader.from_json(self.header_path.read_text(encoding="utf-8"))
        if header != self.header:
            raise VaultIntegrityError("Credential vault header changed")
        key = derive_key(passphrase, header)
        row = db.execute("SELECT envelope FROM sentinel WHERE id=1").fetchone()
        if row is None:
            raise VaultIntegrityError("Credential vault sentinel missing")
        try:
            plaintext = decrypt_record(key, header, _SENTINEL_ID,
                                       {"header": header.to_json()}, EncryptedValue.from_json(row[0]))
        except VaultIntegrityError as exc:
            raise VaultIntegrityError("Credential vault unlock failed") from exc
        if plaintext != _SENTINEL_VALUE:
            raise VaultIntegrityError("Credential vault unlock failed")
        return key

    def add(self, *, passphrase: str, value: str, service: str,
            project: str, source_hash: str) -> str:
        meta = _metadata(service, project, source_hash)
        record_id = uuid.uuid4().hex
        with self._lock, self._process_lock(), self._connect() as db:
            key = self._unlock(db, passphrase)
            encrypted = encrypt_record(key, self.header, record_id, meta, value)
            db.execute("INSERT INTO credentials VALUES (?, ?, ?, ?, ?)",
                       (record_id, service, project, source_hash, encrypted.to_json()))
        return record_id

    def search(self, query: str, *, limit: int = 50) -> list[dict[str, str]]:
        if not isinstance(query, str) or not 1 <= len(query) <= 64 or not 1 <= limit <= 100:
            raise ValueError("Invalid credential search")
        pattern = "%" + query.replace("\\", "\\\\").replace("%", "\\%")\
            .replace("_", "\\_") + "%"
        with self._lock, self._connect(readonly=True) as db:
            rows = db.execute(
                "SELECT id, service, project, source_hash FROM credentials "
                "WHERE service LIKE ? ESCAPE '\\' OR project LIKE ? ESCAPE '\\' LIMIT ?",
                (pattern, pattern, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def reveal(self, record_id: str, *, passphrase: str) -> str:
        if not isinstance(record_id, str) or not re.fullmatch(r"[a-f0-9]{32}", record_id):
            raise VaultIntegrityError("Credential record unavailable")
        with self._lock, self._process_lock(), self._connect() as db:
            key = self._unlock(db, passphrase)
            row = db.execute("SELECT * FROM credentials WHERE id=?", (record_id,)).fetchone()
            if row is None:
                raise VaultIntegrityError("Credential record unavailable")
            meta = _metadata(row["service"], row["project"], row["source_hash"])
            value = decrypt_record(key, self.header, record_id, meta, EncryptedValue.from_json(row["envelope"]))
            db.execute("INSERT INTO reveal_audit (record_id, at) VALUES (?, ?)", (record_id, time.time()))
            return value

    def _backup_sqlite(self, destination: Path) -> None:
        with self._connect(readonly=True) as source:
            target = sqlite3.connect(destination)
            try:
                source.backup(target)
            finally:
                target.close()

    def backup(self, destination: Path, *, passphrase: str) -> int:
        """Create a consistent encrypted portable backup; never copy a live DB file."""
        destination = Path(destination)
        if destination.exists() or destination.is_symlink():
            raise VaultIntegrityError("Credential backup destination exists")
        staging = destination.with_name(f".{destination.name}.incomplete-{uuid.uuid4().hex}")
        with self._lock, self._process_lock(), self._connect(readonly=True) as db:
            key = self._unlock(db, passphrase)
            source_rows = db.execute("SELECT * FROM credentials ORDER BY id").fetchall()
            # Verify all records before and after snapshot, not just the sentinel.
            for row in source_rows:
                meta = _metadata(row["service"], row["project"], row["source_hash"])
                decrypt_record(key, self.header, row["id"], meta, EncryptedValue.from_json(row["envelope"]))
            create_private_directory(staging)
            create_private_file(staging / "header.json")
            create_private_file(staging / "records.db")
            create_private_file(staging / "vault.lock")
            (staging / "vault.lock").write_bytes(b"\0")
            shutil.copyfile(self.header_path, staging / "header.json")
            self._backup_sqlite(staging / "records.db")
            restored = CredentialStore(staging)
            with restored._connect(readonly=True) as check:
                restored_key = restored._unlock(check, passphrase)
                backup_rows = check.execute("SELECT * FROM credentials ORDER BY id").fetchall()
                if len(backup_rows) != len(source_rows):
                    raise VaultIntegrityError("Credential backup count mismatch")
                for original, copied in zip(source_rows, backup_rows):
                    if tuple(original) != tuple(copied):
                        raise VaultIntegrityError("Credential backup differs from source")
                    meta = _metadata(copied["service"], copied["project"], copied["source_hash"])
                    decrypt_record(restored_key, restored.header, copied["id"], meta,
                                   EncryptedValue.from_json(copied["envelope"]))
            if destination.exists() or destination.is_symlink():
                raise VaultIntegrityError("Credential backup destination appeared during backup")
            # The staging directory remains clearly marked incomplete on any failure.
            os.rename(staging, destination)
            return len(backup_rows)


def source_fingerprint(source: str) -> str:
    """Hash a source locator before storing it as searchable metadata."""
    if not isinstance(source, str) or not source:
        raise ValueError("Invalid source locator")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()
