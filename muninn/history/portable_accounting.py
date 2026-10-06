"""Authenticated, disabled accounting for a self-contained history recovery bundle.

Only the nonsecret managed billing database is copied. No environment, credential
vault, API key, model configuration, or source policy is modified. The encrypted
snapshot is bound to the portable history key; plaintext sibling bookkeeping is
never the restore authority.
"""
import os
import sqlite3
from contextlib import closing
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.private_acl import create_private_directory, create_private_file, verify_private
from muninn.history.remote_policy import _paths, read_policy, write_policy

SNAPSHOT = "runtime-accounting.enc"
MARKER = "muninn-runtime-backup"
MAGIC = b"muninn-runtime-accounting-v1\0"
POLICY_MARKER = b"muninn-managed-remote-policy-v1\n"
ADMISSION_MARKER = b"muninn-managed-remote-admission-v1\n"
_COLUMNS = {
    "policy": {"id", "version", "enabled", "daily_usd", "monthly_usd", "override_ceiling",
               "generation", "accounting_version"},
    "audit": {"generation", "changed_at", "enabled", "daily_usd", "monthly_usd", "override_ceiling"},
    "remote_admissions": {"id", "generation", "state", "started", "start_day", "start_month",
                          "finished", "end_day", "end_month", "cost_micro", "resolution", "batch_owner",
                          "classification_job", "classification_input"},
    "batch_policy": {"id", "enabled", "generation", "max_batches"},
    "batch_policy_audit": {"generation", "changed_at", "enabled", "max_batches"},
    "batch_consent": {"batch_id", "retention_generation", "remote_generation", "input_sha256"},
}


def require_snapshot_support():
    if not all(callable(getattr(sqlite3.Connection, name, None))
               for name in ("serialize", "deserialize")):
        raise VaultIntegrityError(
            "Managed accounting recovery requires SQLite serialize/deserialize support (Python 3.11+)")


def _validate(db):
    if db.execute("PRAGMA integrity_check").fetchone() != ("ok",):
        raise VaultIntegrityError("Recovery accounting integrity failed")
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table' "
                                         "AND name NOT LIKE 'sqlite_%'")}
    if not {"policy", "audit"} <= tables <= _COLUMNS.keys():
        raise VaultIntegrityError("Recovery accounting schema is not allowlisted")
    for table in tables:
        if {row[1] for row in db.execute(f"PRAGMA table_info({table})")} - _COLUMNS[table]:
            raise VaultIntegrityError("Recovery accounting fields are not allowlisted")
    columns = {row[1] for row in db.execute("PRAGMA table_info(policy)")}
    version = db.execute("SELECT accounting_version FROM policy WHERE id=1").fetchone() \
        if "accounting_version" in columns else (0,)
    if version not in {(0,), (1,)} or (version == (1,)) != ("remote_admissions" in tables):
        raise VaultIntegrityError("Recovery accounting sentinel is inconsistent")
    if "remote_admissions" in tables:
        fields = {row[1] for row in db.execute("PRAGMA table_info(remote_admissions)")}
        bound = {"classification_job", "classification_input"}
        if fields & bound and not bound <= fields:
            raise VaultIntegrityError("Recovery classification binding is incomplete")
        if bound <= fields:
            query = "SELECT classification_job,classification_input" + (",batch_owner" if "batch_owner" in fields else "")
            for row in db.execute(query + " FROM remote_admissions"):
                job, digest = row[:2]
                if job is None and digest is None:
                    continue
                if (not isinstance(job, str) or len(job) != 32 or any(c not in "0123456789abcdef" for c in job)
                        or not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                        or len(row) == 3 and row[2] is not None):
                    raise VaultIntegrityError("Recovery classification ownership is invalid")
    return version == (1,)


def _write_marker(path, value):
    create_private_file(path)
    with path.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _marker(path, expected):
    verify_private(path)
    with path.open("rb") as stream:
        if stream.read(len(expected) + 1) != expected:
            raise VaultIntegrityError("Recovery accounting marker is invalid")


def has_accounting(root):
    directory = _paths(root)[0]
    return directory.exists() or directory.is_symlink()


def snapshot_into(archive, source_root, destination_root):
    """Call AFTER journal snapshot so every dispatched claim's hold is included."""
    require_snapshot_support()
    source_directory, source_marker, source_db = _paths(source_root)
    for path in (source_directory, source_marker, source_db):
        verify_private(path)
    _marker(source_marker, POLICY_MARKER)
    directory, marker, database = _paths(destination_root)
    create_private_directory(directory)
    create_private_file(database)
    with closing(sqlite3.connect(source_db.as_uri() + "?mode=ro", uri=True, timeout=5)) as original:
        with closing(sqlite3.connect(database)) as copied:
            original.backup(copied)
            accounting = _validate(copied)
    source_admission = source_directory / "admission-managed"
    if accounting:
        _marker(source_admission, ADMISSION_MARKER)
    elif source_admission.exists() or source_admission.is_symlink():
        raise VaultIntegrityError("Recovery accounting marker is inconsistent")
    _write_marker(marker, POLICY_MARKER)
    if accounting:
        _write_marker(directory / "admission-managed", ADMISSION_MARKER)
    prior = read_policy(destination_root, lambda: (False, 1, 30, False))
    write_policy(destination_root, enabled=False, daily_usd=prior.daily_usd, monthly_usd=prior.monthly_usd,
                 override_ceiling=prior.override_ceiling, fallback=lambda: (False, 1, 30, False))
    with closing(sqlite3.connect(database)) as db:
        batch = db.execute("SELECT 1 FROM sqlite_master WHERE name='batch_policy'").fetchone()
        quota = db.execute("SELECT max_batches FROM batch_policy WHERE id=1").fetchone() if batch else None
    if batch:
        from muninn.history.batch_activation import configure_batch
        configure_batch(destination_root, enabled=False, max_batches=quota[0])
    with closing(sqlite3.connect(database)) as db, closing(sqlite3.connect(":memory:")) as memory:
        db.backup(memory)
        serialized = memory.serialize()
    nonce = os.urandom(12)
    sealed = MAGIC + nonce + AESGCM(archive._key).encrypt(
        nonce, serialized, MAGIC + archive.vault_id.encode("ascii"))
    _write_marker(archive.root / SNAPSHOT, sealed)
    _write_marker(Path(destination_root) / MARKER, MAGIC)


def _read_snapshot(archive):
    path = archive.root / SNAPSHOT
    verify_private(path)
    sealed = path.read_bytes()
    if not sealed.startswith(MAGIC) or len(sealed) < len(MAGIC) + 28:
        raise VaultIntegrityError("Recovery accounting snapshot is invalid")
    nonce = sealed[len(MAGIC):len(MAGIC) + 12]
    try:
        raw = AESGCM(archive._key).decrypt(nonce, sealed[len(MAGIC) + 12:],
                                         MAGIC + archive.vault_id.encode("ascii"))
    except InvalidTag as exc:
        raise VaultIntegrityError("Recovery accounting authentication failed") from exc
    return raw


def _disabled(db):
    accounting = _validate(db)
    if db.execute("SELECT enabled FROM policy WHERE id=1").fetchone() != (0,):
        raise VaultIntegrityError("Recovery remote policy is enabled")
    if db.execute("SELECT 1 FROM sqlite_master WHERE name='batch_policy'").fetchone():
        if db.execute("SELECT enabled FROM batch_policy WHERE id=1").fetchone() != (0,):
            raise VaultIntegrityError("Recovery batch policy is enabled")
    return accounting


def verify_snapshot(archive, runtime_root):
    """Read back the actual ciphertext and compare its exact copied bookkeeping."""
    require_snapshot_support()
    raw = _read_snapshot(archive)
    _marker(Path(runtime_root) / MARKER, MAGIC)
    database = _paths(runtime_root)[2]
    verify_private(database)
    try:
        with closing(sqlite3.connect(":memory:")) as memory, closing(sqlite3.connect(
                database.as_uri() + "?mode=ro", uri=True, timeout=5)) as copied:
            memory.deserialize(raw)
            _disabled(memory)
            _disabled(copied)
            for table in _COLUMNS:
                exists = memory.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone()
                other = copied.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone()
                if exists != other:
                    raise VaultIntegrityError("Recovery accounting snapshot differs")
                if exists:
                    fields = [row[1] for row in memory.execute(f"PRAGMA table_info({table})")]
                    if fields != [row[1] for row in copied.execute(f"PRAGMA table_info({table})")]:
                        raise VaultIntegrityError("Recovery accounting snapshot differs")
                    query = f"SELECT * FROM {table} ORDER BY " + ",".join(fields)
                    if memory.execute(query).fetchall() != copied.execute(query).fetchall():
                        raise VaultIntegrityError("Recovery accounting snapshot differs")
    except sqlite3.Error as exc:
        raise VaultIntegrityError("Recovery accounting database is invalid") from exc


def restore_into(archive, destination_root):
    """Reconstruct only authenticated bookkeeping; never enable provider use."""
    require_snapshot_support()
    raw = _read_snapshot(archive)
    directory, marker, database = _paths(destination_root)
    if directory.exists() or directory.is_symlink():
        raise VaultIntegrityError("Recovery accounting destination already exists")
    with closing(sqlite3.connect(":memory:")) as memory:
        try:
            memory.deserialize(raw)
            accounting = _disabled(memory)
            create_private_directory(directory)
            create_private_file(database)
            with closing(sqlite3.connect(database)) as target:
                memory.backup(target)
        except sqlite3.Error as exc:
            raise VaultIntegrityError("Recovery accounting database is invalid") from exc
    _write_marker(marker, POLICY_MARKER)
    if accounting:
        _write_marker(directory / "admission-managed", ADMISSION_MARKER)
    read_policy(destination_root, lambda: (False, 1, 30, False))
    _write_marker(Path(destination_root) / MARKER, MAGIC)
