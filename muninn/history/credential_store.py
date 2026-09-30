"""Portable, locally unlocked credential store with discovery receipts.

Only minimized metadata is searchable. Values never enter the normal Muninn
database, vector/graph indexes, or model prompts through this module.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import shutil
import sqlite3
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass
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

_RECEIPT_VERSION = 2
_RECEIPT_SCHEMA_V1 = (
    "CREATE TABLE scan_receipts (receipt_id TEXT PRIMARY KEY NOT NULL, "
    "scanner_version INTEGER NOT NULL CHECK(scanner_version=1), "
    "scanned_at REAL NOT NULL)"
)
_RECEIPT_SCHEMA = (
    "CREATE TABLE scan_receipts (receipt_id TEXT PRIMARY KEY NOT NULL, "
    "scanner_version INTEGER NOT NULL CHECK(scanner_version IN (1,2)), "
    "scanned_at REAL NOT NULL)"
)
_SENTINEL_ID = "__vault_sentinel__"
_SENTINEL_VALUE = "muninn-credential-vault-v1"
_EMPTY_AMBIGUITY_VALUE = "muninn-ambiguity-empty-v1"
_SAFE_LABEL = re.compile(r"[A-Za-z0-9_][A-Za-z0-9 _.-]{0,63}\Z")
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_OLD_COLUMNS = ["id", "service", "project", "source_hash", "envelope"]
_LEGACY_COLUMNS = [*_OLD_COLUMNS, "source_hint"]
_NEW_COLUMNS = [*_LEGACY_COLUMNS, "origin", "active", "discovery_key"]
_ORIGINS = {"manual", "project", "transcript"}


@dataclass(frozen=True)
class AmbiguousCandidate:
    """A bounded, non-accepted candidate retained only in the encrypted vault."""

    name: str
    reason: str
    candidate: str
    source_hint: str


def _utf16_units(value: str) -> int:
    return sum(2 if ord(char) > 0xFFFF else 1 for char in value)


def _safe_display_text(value: str) -> bool:
    # Control, format (including bidi overrides), and surrogate characters can
    # spoof an agent-visible location even though the field is not executable.
    return all(unicodedata.category(char) not in {"Cc", "Cf", "Cs", "Zl", "Zp"}
               for char in value)


def _valid_project_label(value: str) -> bool:
    return (isinstance(value, str) and value not in {"", ".", ".."}
            and _utf16_units(value) <= 255 and "/" not in value and "\\" not in value
            and _safe_display_text(value))


def _validated_source_hint(value: str) -> str:
    """Keep agent-facing file locations project-relative and traversal-free."""
    if value == "":
        return value
    if (not isinstance(value, str) or _utf16_units(value) > 32767
            or "\\" in value or ":" in value or not _safe_display_text(value)):
        raise VaultIntegrityError("Invalid credential source hint")
    parts = value.split("/")
    if any(part in ("", ".", "..") or _utf16_units(part) > 255 for part in parts):
        raise VaultIntegrityError("Invalid credential source hint")
    return value


def _metadata(service: str, project: str, source_hash: str,
              source_hint: str = "", *, origin: str = "manual", active: bool = True,
              discovery_key: str = "") -> dict[str, str | int]:
    if not isinstance(service, str) or not _SAFE_LABEL.fullmatch(service):
        raise VaultIntegrityError("Invalid credential metadata")
    if not _valid_project_label(project):
        raise VaultIntegrityError("Invalid credential metadata")
    if not isinstance(source_hash, str) or not _SHA256.fullmatch(source_hash):
        raise VaultIntegrityError("Invalid credential metadata")
    meta = {"service": service, "project": project, "source_hash": source_hash}
    if origin not in _ORIGINS or not isinstance(active, bool):
        raise VaultIntegrityError("Invalid credential metadata")
    if not isinstance(discovery_key, str) or (discovery_key and not _SHA256.fullmatch(discovery_key)):
        raise VaultIntegrityError("Invalid credential metadata")
    # Manual records (including migrated 5/6-column records) retain the
    # original AAD exactly. Discovery records bind their lifecycle metadata.
    if origin != "manual" or discovery_key:
        meta.update({"origin": origin, "active": int(active), "discovery_key": discovery_key})
    hint = _validated_source_hint(source_hint)
    if hint:
        meta["source_hint"] = hint
    return meta


class _ScanSession:
    __slots__ = ("_store", "_key", "_valid")

    def __init__(self, store: CredentialStore, key: bytes):
        self._store, self._key, self._valid = store, key, True

    def _invalidate(self) -> None:
        self._valid = False
        self._key = b""

    def scan_source(self, *, passphrase: str = "", source_hash: str, project: str,
                    origin: str, findings, receipt_identity: str | None = None) -> dict[str, int]:
        del passphrase
        if not self._valid:
            raise VaultIntegrityError("Credential scan session is closed")
        return self._store._scan_source(key=self._key, source_hash=source_hash,
                                        project=project, origin=origin, findings=findings,
                                        receipt_identity=receipt_identity)


class CredentialStore:
    """Closed-by-default encrypted records, no long-lived passphrase or key."""

    def __init__(self, root: Path):
        # SQLite's read-only URI needs an absolute path; keep the spelling
        # (rather than resolving links) so verify_private can reject links.
        self.root = Path(root).absolute()
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
                   "project TEXT NOT NULL, source_hash TEXT NOT NULL, envelope TEXT NOT NULL, "
                   "source_hint TEXT NOT NULL DEFAULT '', origin TEXT NOT NULL DEFAULT 'manual', "
                   "active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0,1)), discovery_key TEXT NOT NULL DEFAULT '')")
        db.execute("CREATE UNIQUE INDEX credential_discovery_identity ON credentials(discovery_key) "
                   "WHERE discovery_key <> ''")
        db.execute("PRAGMA user_version=2")
        db.execute("CREATE TABLE reveal_audit (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                   "record_id TEXT NOT NULL, at REAL NOT NULL)")
        db.execute(_RECEIPT_SCHEMA)
        db.execute("CREATE TABLE ambiguity_queue (id TEXT PRIMARY KEY NOT NULL, "
                   "source_hash TEXT NOT NULL, project TEXT NOT NULL, origin TEXT NOT NULL, "
                   "name TEXT NOT NULL, reason TEXT NOT NULL, source_hint TEXT NOT NULL, "
                   "ambiguity_key TEXT NOT NULL UNIQUE, group_digest TEXT NOT NULL DEFAULT '', envelope TEXT NOT NULL, "
                   "status TEXT NOT NULL CHECK(status IN ('pending','accepted','rejected','deferred')), "
                   "created_at REAL NOT NULL, decided_at REAL, decision_actor TEXT, decision_reason TEXT)")
        db.execute("CREATE TABLE ambiguity_audit (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                   "ambiguity_id TEXT NOT NULL, action TEXT NOT NULL, actor TEXT NOT NULL, at REAL NOT NULL)")
        db.execute("CREATE UNIQUE INDEX ambiguity_queue_key ON ambiguity_queue(ambiguity_key)")

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
            if "credentials_v1" in names:
                raise VaultIntegrityError("Incomplete credential vault migration")
            columns = [row[1] for row in db.execute("PRAGMA table_info(credentials)")]
            version = db.execute("PRAGMA user_version").fetchone()[0]
        if columns in (_OLD_COLUMNS, _LEGACY_COLUMNS) and version != 0:
            raise VaultIntegrityError("Unsupported credential vault schema version")
        if columns in (_OLD_COLUMNS, _LEGACY_COLUMNS):
            with self._process_lock(), self._connect() as db:
                columns = [row[1] for row in db.execute("PRAGMA table_info(credentials)")]
                if columns in (_OLD_COLUMNS, _LEGACY_COLUMNS):
                    if db.execute("PRAGMA user_version").fetchone()[0] != 0:
                        raise VaultIntegrityError("Unsupported credential vault schema version")
                    db.execute("ALTER TABLE credentials RENAME TO credentials_v1")
                    db.execute("CREATE TABLE credentials (id TEXT PRIMARY KEY, service TEXT NOT NULL, project TEXT NOT NULL, source_hash TEXT NOT NULL, envelope TEXT NOT NULL, source_hint TEXT NOT NULL DEFAULT '', origin TEXT NOT NULL DEFAULT 'manual', active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0,1)), discovery_key TEXT NOT NULL DEFAULT '')")
                    if columns == _OLD_COLUMNS:
                        db.execute("INSERT INTO credentials(id,service,project,source_hash,envelope) SELECT id,service,project,source_hash,envelope FROM credentials_v1")
                    else:
                        db.execute("INSERT INTO credentials(id,service,project,source_hash,envelope,source_hint) SELECT id,service,project,source_hash,envelope,source_hint FROM credentials_v1")
                    db.execute("DROP TABLE credentials_v1")
                    db.execute("CREATE UNIQUE INDEX credential_discovery_identity ON credentials(discovery_key) WHERE discovery_key <> ''")
                    db.execute("PRAGMA user_version=2")
                    columns = [row[1] for row in db.execute("PRAGMA table_info(credentials)")]
        if columns != _NEW_COLUMNS:
            raise VaultIntegrityError("Unexpected credential vault schema")
        with self._connect(readonly=True) as db:
            if db.execute("PRAGMA user_version").fetchone()[0] != 2:
                raise VaultIntegrityError("Unsupported credential vault schema version")
            index = db.execute("SELECT sql FROM sqlite_master WHERE type='index' AND name=?",
                               ("credential_discovery_identity",)).fetchone()
            if index is None or "UNIQUE" not in index[0].upper() or "WHERE discovery_key <> ''" not in index[0]:
                raise VaultIntegrityError("Incomplete credential discovery index")
            receipt = db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='scan_receipts'").fetchone()
        if receipt is None:
            # Old v2 vaults and portable backups receive the compatible table atomically.
            with self._process_lock(), self._connect() as db:
                current = db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='scan_receipts'").fetchone()
                if current is None:
                    db.execute(_RECEIPT_SCHEMA)
                    current = db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='scan_receipts'").fetchone()
                receipt = current
        if receipt is None:
            raise VaultIntegrityError("Invalid credential scan receipt schema")
        if receipt[0] == _RECEIPT_SCHEMA_V1:
            with self._process_lock(), self._connect() as db:
                db.execute("ALTER TABLE scan_receipts RENAME TO scan_receipts_v1")
                db.execute(_RECEIPT_SCHEMA)
                db.execute("INSERT INTO scan_receipts SELECT * FROM scan_receipts_v1")
                db.execute("DROP TABLE scan_receipts_v1")
                receipt = ( _RECEIPT_SCHEMA, )
        if receipt[0] != _RECEIPT_SCHEMA:
            raise VaultIntegrityError("Invalid credential scan receipt schema")
        with self._connect(readonly=True) as db:
            names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "ambiguity_queue" not in names or "ambiguity_audit" not in names:
            with self._process_lock(), self._connect() as db:
                if "ambiguity_queue" not in names:
                    db.execute("CREATE TABLE ambiguity_queue (id TEXT PRIMARY KEY NOT NULL, source_hash TEXT NOT NULL, project TEXT NOT NULL, origin TEXT NOT NULL, name TEXT NOT NULL, reason TEXT NOT NULL, source_hint TEXT NOT NULL, ambiguity_key TEXT NOT NULL UNIQUE, group_digest TEXT NOT NULL DEFAULT '', envelope TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('pending','accepted','rejected','deferred')), created_at REAL NOT NULL, decided_at REAL, decision_actor TEXT, decision_reason TEXT)")
                    db.execute("CREATE INDEX ambiguity_queue_group ON ambiguity_queue(group_digest, status)")
                    db.execute("CREATE UNIQUE INDEX ambiguity_queue_key ON ambiguity_queue(ambiguity_key)")
                if "ambiguity_audit" not in names:
                    db.execute("CREATE TABLE ambiguity_audit (id INTEGER PRIMARY KEY AUTOINCREMENT, ambiguity_id TEXT NOT NULL, action TEXT NOT NULL, actor TEXT NOT NULL, at REAL NOT NULL)")
        with self._connect(readonly=True) as db:
            columns = {row[1] for row in db.execute("PRAGMA table_info(ambiguity_queue)")}
            group_index = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name='ambiguity_queue_group'"
            ).fetchone()
        if "group_digest" not in columns or group_index is None:
            with self._process_lock(), self._connect() as db:
                current = {row[1] for row in db.execute("PRAGMA table_info(ambiguity_queue)")}
                if "group_digest" not in current:
                    db.execute("ALTER TABLE ambiguity_queue ADD COLUMN group_digest TEXT NOT NULL DEFAULT ''")
                db.execute("CREATE INDEX IF NOT EXISTS ambiguity_queue_group ON ambiguity_queue(group_digest, status)")
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

    def _verify_key(self, db: sqlite3.Connection, key: bytes) -> None:
        """Revalidate the current header and sentinel without deriving again."""
        header = VaultHeader.from_json(self.header_path.read_text(encoding="utf-8"))
        if header != self.header:
            raise VaultIntegrityError("Credential vault header changed")
        row = db.execute("SELECT envelope FROM sentinel WHERE id=1").fetchone()
        if row is None:
            raise VaultIntegrityError("Credential vault sentinel missing")
        try:
            value = decrypt_record(key, header, _SENTINEL_ID,
                                   {"header": header.to_json()}, EncryptedValue.from_json(row[0]))
        except VaultIntegrityError as exc:
            raise VaultIntegrityError("Credential vault unlock failed") from exc
        if value != _SENTINEL_VALUE:
            raise VaultIntegrityError("Credential vault unlock failed")

    @contextmanager
    def scan_session(self, passphrase: str):
        with self._lock, self._process_lock(), self._connect() as db:
            key = self._unlock(db, passphrase)
        session = _ScanSession(self, key)
        try:
            yield session
        finally:
            session._invalidate()

    def add(self, *, passphrase: str, value: str, service: str,
            project: str, source_hash: str, source_hint: str = "") -> str:
        meta = _metadata(service, project, source_hash, source_hint)
        record_id = uuid.uuid4().hex
        with self._lock, self._process_lock(), self._connect() as db:
            key = self._unlock(db, passphrase)
            encrypted = encrypt_record(key, self.header, record_id, meta, value)
            db.execute("INSERT INTO credentials (id, service, project, source_hash, envelope, source_hint) "
                       "VALUES (?, ?, ?, ?, ?, ?)",
                       (record_id, service, project, source_hash, encrypted.to_json(), source_hint))
        return record_id

    def search(self, query: str, *, limit: int = 50, active_only: bool = True) -> list[dict[str, str]]:
        if not isinstance(query, str) or not 1 <= len(query) <= 64 or not 1 <= limit <= 100:
            raise ValueError("Invalid credential search")
        pattern = "%" + query.replace("\\", "\\\\").replace("%", "\\%")\
            .replace("_", "\\_") + "%"
        with self._lock, self._connect(readonly=True) as db:
            rows = db.execute(
                "SELECT id, service, project, source_hash, source_hint, origin, active FROM credentials "
                "WHERE (? = 0 OR active=1) AND (service LIKE ? ESCAPE '\\' OR project LIKE ? ESCAPE '\\' "
                "OR source_hint LIKE ? ESCAPE '\\') LIMIT ?",
                (int(active_only), pattern, pattern, pattern, limit),
            ).fetchall()
            remaining = limit - len(rows)
            ambiguities = db.execute(
                "SELECT id,name,project,source_hash,source_hint,origin,status "
                "FROM ambiguity_queue WHERE status IN ('pending','deferred') "
                "AND (name LIKE ? ESCAPE '\\' OR project LIKE ? ESCAPE '\\' "
                "OR source_hint LIKE ? ESCAPE '\\') ORDER BY created_at,id LIMIT ?",
                (pattern, pattern, pattern, remaining),
            ).fetchall() if remaining else []
        found = [({k: row[k] for k in ("id", "service", "project", "source_hash", "source_hint")}
                  if row["origin"] == "manual" else {**dict(row), "candidate_status": "unverified"})
                 for row in rows if _validated_source_hint(row["source_hint"]) is not None]
        found.extend({"id": row["id"], "service": row["name"], "project": row["project"],
                      "source_hash": row["source_hash"], "source_hint": row["source_hint"],
                      "origin": row["origin"], "candidate_status": "needs_review",
                      "review_status": row["status"], "vault_record_type": "ambiguity"}
                     for row in ambiguities if _validated_source_hint(row["source_hint"]) is not None)
        return found

    def scan_source(self, *, passphrase: str, source_hash: str, project: str,
                    origin: str, findings, receipt_identity: str | None = None) -> dict[str, int]:
        with self.scan_session(passphrase) as session:
            return session.scan_source(passphrase=passphrase, source_hash=source_hash,
                                       project=project, origin=origin, findings=findings,
                                       receipt_identity=receipt_identity)

    def _scan_source(self, *, key: bytes, source_hash: str, project: str,
                     origin: str, findings, receipt_identity: str | None = None) -> dict[str, int]:
        if origin not in {"project", "transcript"}:
            raise ValueError("Invalid discovery origin")
        if not _valid_project_label(project):
            raise VaultIntegrityError("Invalid credential metadata")
        if receipt_identity is not None and (origin != "transcript" or
                                             not isinstance(receipt_identity, str) or
                                             not 1 <= len(receipt_identity) <= 512):
            raise VaultIntegrityError("Invalid credential scan receipt")
        receipt_id = (hmac.new(key, b"credential-scan-snapshot-v1\0" +
                               receipt_identity.encode("utf-8"), hashlib.sha256).hexdigest()
                      if receipt_identity is not None else None)
        seen: set[str] = set()
        _metadata("scan", project, source_hash)
        counts = {"created": 0, "rotated": 0, "staled": 0, "ambiguities": 0}
        with self._lock, self._process_lock(), self._connect() as db:
            self._verify_key(db, key)
            if receipt_id is not None and db.execute(
                    "SELECT 1 FROM scan_receipts WHERE receipt_id=? AND scanner_version=?",
                    (receipt_id, _RECEIPT_VERSION)).fetchone() is not None:
                return {"created": 0, "rotated": 0, "staled": 0, "skipped": 1}
            stream = findings
            for item in stream:
                if isinstance(item, AmbiguousCandidate):
                    ambiguity_id = uuid.uuid4().hex
                    if (not isinstance(item.name, str) or not _SAFE_LABEL.fullmatch(item.name)
                            or not isinstance(item.reason, str) or not 1 <= len(item.reason) <= 128
                            or not _safe_display_text(item.reason)
                            or not isinstance(item.candidate, str) or len(item.candidate) > 512):
                        raise VaultIntegrityError("Invalid credential ambiguity")
                    source_hint = _validated_source_hint(item.source_hint)
                    ambiguity_key = hmac.new(
                        key, (project + "\0" + origin + "\0" + source_hash + "\0"
                              + item.name + "\0" + item.reason + "\0"
                              + item.candidate).encode("utf-8"), hashlib.sha256
                    ).hexdigest()
                    meta = _metadata(item.name, project, source_hash, source_hint, origin=origin, active=True, discovery_key=ambiguity_key)
                    envelope = encrypt_record(key, self.header, ambiguity_id, meta,
                                              item.candidate or _EMPTY_AMBIGUITY_VALUE).to_json()
                    # Reuse one local-model judgment only within the same
                    # project and origin; context can differ across projects.
                    group_digest = "v2:" + hmac.new(
                        key,
                        (project + "\0" + origin + "\0" + item.name + "\0"
                         + item.reason + "\0" + item.candidate).encode("utf-8"),
                        hashlib.sha256,
                    ).hexdigest()
                    cursor = db.execute("INSERT OR IGNORE INTO ambiguity_queue (id,source_hash,project,origin,name,reason,source_hint,ambiguity_key,group_digest,envelope,status,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                               (ambiguity_id, source_hash, project, origin, item.name, item.reason, source_hint, ambiguity_key, group_digest, envelope, "pending", time.time()))
                    counts["ambiguities"] += int(cursor.rowcount == 1)
                    continue
                service, value, source_hint = item
                identity = service + "\0" + project + "\0" + source_hash
                if origin == "transcript":
                    identity += "\0" + value
                discovery_key = hmac.new(key, identity.encode(), hashlib.sha256).hexdigest()
                meta = _metadata(service, project, source_hash, source_hint, origin=origin, active=True, discovery_key=discovery_key)
                row = db.execute("SELECT * FROM credentials WHERE discovery_key=?", (discovery_key,)).fetchone()
                if row is None:
                    record_id = uuid.uuid4().hex
                    envelope = encrypt_record(key, self.header, record_id, meta, value).to_json()
                    db.execute("INSERT INTO credentials VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (record_id, service, project, source_hash, envelope, source_hint, origin, 1, discovery_key))
                    counts["created"] += 1
                else:
                    old_meta = _metadata(row["service"], row["project"], row["source_hash"], row["source_hint"], origin=row["origin"], active=bool(row["active"]), discovery_key=row["discovery_key"])
                    old_value = decrypt_record(key, self.header, row["id"], old_meta, EncryptedValue.from_json(row["envelope"]))
                    if old_value != value or row["active"] == 0 or row["source_hint"] != source_hint:
                        envelope = encrypt_record(key, self.header, row["id"], meta, value).to_json()
                        db.execute("UPDATE credentials SET service=?,project=?,source_hash=?,envelope=?,source_hint=?,origin=?,active=1 WHERE id=?", (service, project, source_hash, envelope, source_hint, origin, row["id"]))
                        counts["rotated"] += 1
                seen.add(discovery_key)
            if origin == "project":
                stale_rows = db.execute("SELECT * FROM credentials WHERE origin='project' AND active=1 AND project=? AND source_hash=?", (project, source_hash)).fetchall()
                for row in stale_rows:
                    if row["discovery_key"] in seen:
                        continue
                    old_meta = _metadata(row["service"], row["project"], row["source_hash"], row["source_hint"], origin="project", active=True, discovery_key=row["discovery_key"])
                    value = decrypt_record(key, self.header, row["id"], old_meta, EncryptedValue.from_json(row["envelope"]))
                    new_meta = _metadata(row["service"], row["project"], row["source_hash"], row["source_hint"], origin="project", active=False, discovery_key=row["discovery_key"])
                    envelope = encrypt_record(key, self.header, row["id"], new_meta, value).to_json()
                    db.execute("UPDATE credentials SET envelope=?, active=0 WHERE id=?", (envelope, row["id"]))
                    counts["staled"] += 1
            for row in db.execute("SELECT * FROM credentials WHERE project=? AND source_hash=? AND origin=?",
                                  (project, source_hash, origin)):
                if row["discovery_key"] not in seen:
                    continue
                meta = _metadata(row["service"], row["project"], row["source_hash"], row["source_hint"], origin=row["origin"], active=bool(row["active"]), discovery_key=row["discovery_key"])
                decrypt_record(key, self.header, row["id"], meta, EncryptedValue.from_json(row["envelope"]))
            if receipt_id is not None:
                db.execute("INSERT INTO scan_receipts (receipt_id,scanner_version,scanned_at) VALUES (?,?,?)",
                           (receipt_id, _RECEIPT_VERSION, time.time()))
        return counts

    def ambiguity_status(self) -> dict[str, int]:
        """Return metadata-only queue counts; never decrypts candidates."""
        with self._lock, self._connect(readonly=True) as db:
            rows = db.execute("SELECT status,COUNT(*) AS count FROM ambiguity_queue GROUP BY status").fetchall()
        return {row["status"]: row["count"] for row in rows}

    def list_ambiguities(self, *, status: str = "pending", limit: int = 50) -> list[dict[str, str | int | float]]:
        if status not in {"pending", "accepted", "rejected", "deferred"} or not 1 <= limit <= 100:
            raise ValueError("Invalid ambiguity query")
        with self._lock, self._connect(readonly=True) as db:
            rows = db.execute("SELECT id,source_hash,project,origin,name,reason,source_hint,status,created_at,decided_at,decision_actor,decision_reason FROM ambiguity_queue WHERE status=? ORDER BY created_at,id LIMIT ?", (status, limit)).fetchall()
        return [dict(row) for row in rows]

    def list_ambiguity_groups(self, *, status: str = "pending", limit: int = 50) -> list[dict[str, str | int]]:
        if status not in {"pending", "accepted", "rejected", "deferred"} or not 1 <= limit <= 100:
            raise ValueError("Invalid ambiguity group query")
        with self._lock, self._connect(readonly=True) as db:
            rows = db.execute(
                "WITH grouped AS (SELECT MIN(id) AS representative_id, "
                "CASE WHEN group_digest LIKE 'v2:%' THEN group_digest ELSE id END AS grouping_key, "
                "COUNT(*) AS count, MIN(created_at) AS first_created "
                "FROM ambiguity_queue WHERE status=? "
                "GROUP BY CASE WHEN group_digest LIKE 'v2:%' THEN group_digest ELSE id END) "
                "SELECT grouped.representative_id, rep.name, rep.reason, "
                "rep.group_digest, grouped.count FROM grouped "
                "JOIN ambiguity_queue AS rep ON rep.id=grouped.representative_id "
                "ORDER BY grouped.first_created,grouped.representative_id LIMIT ?",
                (status, limit),
            ).fetchall()
        return [{"representative_id": row["representative_id"], "name": row["name"],
                 "reason": row["reason"], "group_digest": row["group_digest"],
                 "count": row["count"]} for row in rows]

    def reveal_ambiguity(self, ambiguity_id: str, *, passphrase: str) -> str:
        if not isinstance(ambiguity_id, str) or not re.fullmatch(r"[a-f0-9]{32}", ambiguity_id):
            raise VaultIntegrityError("Credential ambiguity unavailable")
        with self._lock, self._process_lock(), self._connect() as db:
            key = self._unlock(db, passphrase)
            row = db.execute("SELECT * FROM ambiguity_queue WHERE id=?", (ambiguity_id,)).fetchone()
            if row is None:
                raise VaultIntegrityError("Credential ambiguity unavailable")
            meta = _metadata(row["name"], row["project"], row["source_hash"], row["source_hint"], origin=row["origin"], active=True, discovery_key=row["ambiguity_key"])
            value = decrypt_record(key, self.header, ambiguity_id, meta, EncryptedValue.from_json(row["envelope"]))
            db.execute("INSERT INTO ambiguity_audit (ambiguity_id,action,actor,at) VALUES (?,?,?,?)", (ambiguity_id, "reveal", "local-user", time.time()))
            return "" if value == _EMPTY_AMBIGUITY_VALUE else value

    def decide_ambiguity(self, ambiguity_id: str, *, passphrase: str, decision: str,
                         actor: str = "local-user", reason: str = "") -> None:
        if not isinstance(ambiguity_id, str) or not re.fullmatch(r"[a-f0-9]{32}", ambiguity_id):
            raise VaultIntegrityError("Credential ambiguity unavailable")
        if (decision not in {"rejected", "deferred"}
                or not isinstance(actor, str) or not 1 <= len(actor) <= 64
                or not _safe_display_text(actor)
                or reason not in {"", "user-confirmed", "user-rejected", "not-a-secret", "stale-source"}):
            raise ValueError("Invalid ambiguity decision")
        with self._lock, self._process_lock(), self._connect() as db:
            self._unlock(db, passphrase)
            updated = db.execute(
                "UPDATE ambiguity_queue SET status=?,decided_at=?,decision_actor=?,decision_reason=? "
                "WHERE id=? AND status='pending'",
                (decision, time.time(), actor, reason, ambiguity_id),
            )
            if updated.rowcount != 1:
                raise VaultIntegrityError("Credential ambiguity unavailable")
            db.execute("INSERT INTO ambiguity_audit (ambiguity_id,action,actor,at) VALUES (?,?,?,?)", (ambiguity_id, decision, actor, time.time()))

    def decide_ambiguity_group(self, representative_id: str, *, passphrase: str,
                               decision: str, actor: str = "local-user", reason: str = "") -> int:
        if not isinstance(representative_id, str) or not re.fullmatch(r"[a-f0-9]{32}", representative_id):
            raise VaultIntegrityError("Credential ambiguity group unavailable")
        if decision not in {"rejected", "deferred"} or not isinstance(actor, str) or not 1 <= len(actor) <= 64 or not _safe_display_text(actor) or reason not in {"", "user-confirmed", "user-rejected", "not-a-secret", "stale-source"}:
            raise ValueError("Invalid ambiguity group decision")
        with self._lock, self._process_lock(), self._connect() as db:
            self._unlock(db, passphrase)
            representative = db.execute("SELECT group_digest FROM ambiguity_queue WHERE id=? AND status='pending'", (representative_id,)).fetchone()
            if representative is None:
                raise VaultIntegrityError("Credential ambiguity group unavailable")
            digest = representative["group_digest"]
            if not digest.startswith("v2:"):
                ids = [representative_id]
            else:
                ids = [row["id"] for row in db.execute("SELECT id FROM ambiguity_queue WHERE group_digest=? AND status='pending'", (digest,))]
            now = time.time()
            for ambiguity_id in ids:
                db.execute("UPDATE ambiguity_queue SET status=?,decided_at=?,decision_actor=?,decision_reason=? WHERE id=? AND status='pending'", (decision, now, actor, reason, ambiguity_id))
                db.execute("INSERT INTO ambiguity_audit (ambiguity_id,action,actor,at) VALUES (?,?,?,?)", (ambiguity_id, "group-" + decision, actor, now))
            return len(ids)

    def reveal(self, record_id: str, *, passphrase: str) -> str:
        if not isinstance(record_id, str) or not re.fullmatch(r"[a-f0-9]{32}", record_id):
            raise VaultIntegrityError("Credential record unavailable")
        with self._lock, self._process_lock(), self._connect() as db:
            key = self._unlock(db, passphrase)
            row = db.execute("SELECT * FROM credentials WHERE id=?", (record_id,)).fetchone()
            if row is None:
                raise VaultIntegrityError("Credential record unavailable")
            meta = _metadata(row["service"], row["project"], row["source_hash"], row["source_hint"], origin=row["origin"], active=bool(row["active"]), discovery_key=row["discovery_key"])
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
                meta = _metadata(row["service"], row["project"], row["source_hash"], row["source_hint"], origin=row["origin"], active=bool(row["active"]), discovery_key=row["discovery_key"])
                decrypt_record(key, self.header, row["id"], meta, EncryptedValue.from_json(row["envelope"]))
            source_ambiguities = db.execute("SELECT * FROM ambiguity_queue ORDER BY id").fetchall()
            for row in source_ambiguities:
                meta = _metadata(row["name"], row["project"], row["source_hash"], row["source_hint"], origin=row["origin"], active=True, discovery_key=row["ambiguity_key"])
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
                    meta = _metadata(copied["service"], copied["project"], copied["source_hash"], copied["source_hint"], origin=copied["origin"], active=bool(copied["active"]), discovery_key=copied["discovery_key"])
                    decrypt_record(restored_key, restored.header, copied["id"], meta,
                                   EncryptedValue.from_json(copied["envelope"]))
                backup_ambiguities = check.execute("SELECT * FROM ambiguity_queue ORDER BY id").fetchall()
                if [tuple(row) for row in source_ambiguities] != [tuple(row) for row in backup_ambiguities]:
                    raise VaultIntegrityError("Credential ambiguity backup differs from source")
                for row in backup_ambiguities:
                    meta = _metadata(row["name"], row["project"], row["source_hash"], row["source_hint"], origin=row["origin"], active=True, discovery_key=row["ambiguity_key"])
                    decrypt_record(restored_key, restored.header, row["id"], meta, EncryptedValue.from_json(row["envelope"]))
            if destination.exists() or destination.is_symlink():
                raise VaultIntegrityError("Credential backup destination appeared during backup")
            # The staging directory remains clearly marked incomplete on any failure.
            os.rename(staging, destination)
            return len(backup_rows)

    @classmethod
    def restore(cls, source: Path, destination: Path, *, passphrase: str) -> CredentialStore:
        """Re-home a portable encrypted backup under this account's private ACL.

        The source may have been copied from another machine/account. It is never
        opened as a live vault, modified, or trusted before private staging and
        full authenticated validation.
        """
        source, destination = Path(source), Path(destination)
        if destination.exists() or destination.is_symlink():
            raise VaultIntegrityError("Credential restore destination exists")
        header_source, db_source = source / "header.json", source / "records.db"
        if (source.is_symlink() or not source.is_dir()
                or any(path.is_symlink() or not path.is_file() for path in (header_source, db_source))
                or header_source.stat().st_size > 4096 or db_source.stat().st_size > 1_073_741_824
                or (source / "records.db-wal").exists()):
            raise VaultIntegrityError("Invalid credential backup source")
        staging = destination.with_name(f".{destination.name}.incomplete-{uuid.uuid4().hex}")
        create_private_directory(staging)
        for name in ("header.json", "records.db", "vault.lock"):
            create_private_file(staging / name)
        (staging / "vault.lock").write_bytes(b"\0")
        shutil.copyfile(header_source, staging / "header.json")
        shutil.copyfile(db_source, staging / "records.db")
        restored = cls(staging)
        with restored._connect(readonly=True) as db:
            key = restored._unlock(db, passphrase)
            for row in db.execute("SELECT * FROM credentials"):
                meta = _metadata(row["service"], row["project"], row["source_hash"], row["source_hint"], origin=row["origin"], active=bool(row["active"]), discovery_key=row["discovery_key"])
                decrypt_record(key, restored.header, row["id"], meta,
                               EncryptedValue.from_json(row["envelope"]))
            for row in db.execute("SELECT * FROM ambiguity_queue"):
                meta = _metadata(row["name"], row["project"], row["source_hash"], row["source_hint"], origin=row["origin"], active=True, discovery_key=row["ambiguity_key"])
                decrypt_record(key, restored.header, row["id"], meta, EncryptedValue.from_json(row["envelope"]))
        if destination.exists() or destination.is_symlink():
            raise VaultIntegrityError("Credential restore destination appeared during restore")
        os.rename(staging, destination)
        return cls(destination)


def source_fingerprint(source: str) -> str:
    """Hash a source locator before storing it as searchable metadata."""
    if not isinstance(source, str) or not source:
        raise ValueError("Invalid source locator")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()
