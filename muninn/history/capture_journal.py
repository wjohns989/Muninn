"""Owner-only, encrypted-locator journal for strict-history hook capture.

The journal stores no transcript text or plaintext source path. A successful
enqueue means a SQLite FULL-synchronous transaction committed, not that the
source was already archived. One service process owns this database.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.private_acl import create_private_file, verify_private
from muninn.history.secure_archive import SecureHistoryArchive

_SESSION_UUID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.I)
_ERROR_CODES = {"missing", "changed", "permission", "disk", "archive", "locked", "unknown"}


@dataclass(frozen=True)
class CaptureJob:
    key: str
    path: Path
    provider: str
    revision: int
    observed_size: int
    observed_mtime_ns: int


class CaptureJournal:
    def __init__(self, archive: SecureHistoryArchive, *, recover: bool = True):
        self.archive = archive
        # SQLite URI connections require an absolute path even when the archive
        # CLI was given a relative --root.
        self.path = (archive.root / "capture-jobs.db").absolute()
        verify_private(archive.root)
        if not self.path.exists():
            create_private_file(self.path)
        verify_private(self.path)
        self._key = hmac.new(archive._key, b"muninn-capture-journal-key-v1", hashlib.sha256).digest()
        self._aad = b"muninn-capture-locator-v1\0" + archive.vault_id.encode("ascii")
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs ("
                       "source_key TEXT PRIMARY KEY, sealed_locator BLOB NOT NULL, "
                       "provider TEXT NOT NULL, revision INTEGER NOT NULL, "
                       "observed_size INTEGER NOT NULL, observed_mtime_ns INTEGER NOT NULL, "
                       "state TEXT NOT NULL, due_at REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, "
                       "last_error_code TEXT NOT NULL DEFAULT '', updated_at REAL NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS scan_state ("
                       "id INTEGER PRIMARY KEY CHECK(id=1), generation INTEGER NOT NULL, "
                       "complete INTEGER NOT NULL, seen INTEGER NOT NULL, queued INTEGER NOT NULL, "
                       "unchanged INTEGER NOT NULL, excluded INTEGER NOT NULL, "
                       "missing INTEGER NOT NULL, errors INTEGER NOT NULL, finished_at REAL NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS scan_seen ("
                       "source_key TEXT PRIMARY KEY, generation INTEGER NOT NULL)")
            db.execute("INSERT OR IGNORE INTO scan_state VALUES (1, 0, 1, 0, 0, 0, 0, 0, 0, 0)")
            # An interrupted worker cannot hold a claim after service restart.
            # Read-only backup access must not steal an active worker's claim.
            if recover:
                db.execute("UPDATE jobs SET state='pending', due_at=0 WHERE state='capturing'")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        verify_private(self.archive.root)
        verify_private(self.path)
        db = sqlite3.connect(f"{self.path.as_uri()}?mode=rw", uri=True, timeout=0.1)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA journal_mode=DELETE")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA busy_timeout=100")
            with db:
                yield db
        finally:
            db.close()

    def _source_key(self, path: Path, provider: str) -> str:
        if provider not in {"codex", "claude_code", "gemini_cli"}:
            raise ValueError("Unsupported capture provider")
        match = _SESSION_UUID.search(path.name) if provider in {"codex", "claude_code"} else None
        identity = match.group(0).lower() if match else str(path)
        return hmac.new(self._key, (provider + "\0" + identity).encode("utf-8"), hashlib.sha256).hexdigest()

    def source_key(self, path: Path, provider: str) -> str:
        """Opaque source identity for scanner checkpoints, never a display path."""
        return self._source_key(Path(path), provider)

    def _seal(self, path: Path, provider: str) -> bytes:
        nonce = os.urandom(12)
        return nonce + AESGCM(self._key).encrypt(
            nonce, str(path).encode("utf-8"), self._aad + b"\0" + provider.encode("ascii")
        )

    def _open(self, sealed: bytes, provider: str) -> Path:
        try:
            raw = AESGCM(self._key).decrypt(
                sealed[:12], sealed[12:], self._aad + b"\0" + provider.encode("ascii")
            )
            return Path(raw.decode("utf-8"))
        except (InvalidTag, ValueError, UnicodeError) as exc:
            raise VaultIntegrityError("Capture journal locator authentication failed") from exc

    @staticmethod
    def _fingerprint(path: Path) -> tuple[int, int]:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return -1, -1
        return stat.st_size, stat.st_mtime_ns

    def enqueue(self, path: Path, provider: str, *, force: bool = False,
                immediate: bool = False) -> str:
        """Commit a validated locator before the hook may be acknowledged."""
        path = Path(path).resolve(strict=False)
        key = self._source_key(path, provider)
        sealed = self._seal(path, provider)
        size, mtime_ns = self._fingerprint(path)
        now = time.time()
        due = now if force or immediate else now + 2.0
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT revision, observed_size, observed_mtime_ns, state, due_at "
                               "FROM jobs WHERE source_key=?", (key,)).fetchone()
            if (prior and (size, mtime_ns) == (prior["observed_size"], prior["observed_mtime_ns"])
                    and (not force or prior["state"] in {"pending", "capturing", "retry"})):
                return "coalesced"
            if prior:
                due = min(due, prior["due_at"]) if prior["state"] != "archived" else due
                db.execute("UPDATE jobs SET sealed_locator=?, provider=?, revision=revision+1, "
                           "observed_size=?, observed_mtime_ns=?, state='pending', due_at=?, "
                           "attempts=0, last_error_code='', updated_at=? WHERE source_key=?",
                           (sealed, provider, size, mtime_ns, due, now, key))
            else:
                db.execute("INSERT INTO jobs (source_key, sealed_locator, provider, revision, "
                           "observed_size, observed_mtime_ns, state, due_at, updated_at) "
                           "VALUES (?, ?, ?, 1, ?, ?, 'pending', ?, ?)",
                           (key, sealed, provider, size, mtime_ns, due, now))
        return "queued"

    def claim_due(self) -> CaptureJob | None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE state IN ('pending', 'retry') AND due_at<=? "
                             "ORDER BY due_at, updated_at LIMIT 1", (time.time(),)).fetchone()
            if row is None:
                return None
            locator = self._open(row["sealed_locator"], row["provider"])
            db.execute("UPDATE jobs SET state='capturing', updated_at=? WHERE source_key=?",
                       (time.time(), row["source_key"]))
        return CaptureJob(row["source_key"], locator,
                          row["provider"], row["revision"], row["observed_size"],
                          row["observed_mtime_ns"])

    def finish(self, job: CaptureJob, *, archived: bool) -> bool:
        """Only the claimed revision may become archived after manifest commit."""
        if not archived:
            self.fail(job, "archive")
            return False
        size, mtime_ns = self._fingerprint(job.path)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT revision FROM jobs WHERE source_key=?", (job.key,)).fetchone()
            if row is None:
                raise VaultIntegrityError("Capture journal job is missing")
            if row["revision"] != job.revision or (size, mtime_ns) != (
                job.observed_size, job.observed_mtime_ns
            ):
                db.execute("UPDATE jobs SET revision=revision+1, observed_size=?, observed_mtime_ns=?, "
                           "state='pending', due_at=0, updated_at=? WHERE source_key=? AND revision=?",
                           (size, mtime_ns, time.time(), job.key, job.revision))
                return False
            db.execute("UPDATE jobs SET state='archived', last_error_code='', updated_at=? "
                       "WHERE source_key=? AND revision=?", (time.time(), job.key, job.revision))
        return True

    def fail(self, job: CaptureJob, code: str) -> None:
        if code not in _ERROR_CODES:
            code = "unknown"
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT revision, attempts FROM jobs WHERE source_key=?", (job.key,)).fetchone()
            if row is None or row["revision"] != job.revision:
                return
            attempts = row["attempts"] + 1
            delay = min(600, 2 ** min(attempts, 9))
            db.execute("UPDATE jobs SET state='retry', attempts=?, due_at=?, last_error_code=?, "
                       "updated_at=? WHERE source_key=? AND revision=?",
                       (attempts, time.time() + delay, code, time.time(), job.key, job.revision))

    def status(self) -> dict[str, int]:
        with self._connect() as db:
            return {row["state"]: row["count"] for row in db.execute(
                "SELECT state, COUNT(*) AS count FROM jobs GROUP BY state"
            )}

    def backup_to(self, destination: Path) -> None:
        """Snapshot committed encrypted locators with SQLite's online backup API."""
        destination = Path(destination)
        create_private_file(destination)
        with self._connect() as source:
            target = sqlite3.connect(destination)
            try:
                source.backup(target)
            finally:
                target.close()
        verify_private(destination)

    def verify_all(self) -> int:
        """Authenticate every sealed locator and the SQLite snapshot structure."""
        with self._connect() as db:
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise VaultIntegrityError("Capture journal integrity check failed")
            count = 0
            for row in db.execute("SELECT source_key, sealed_locator, provider FROM jobs"):
                path = self._open(row["sealed_locator"], row["provider"])
                if row["source_key"] != self._source_key(path, row["provider"]):
                    raise VaultIntegrityError("Capture journal source identity mismatch")
                count += 1
            return count

    def begin_scan(self) -> int:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT generation, complete FROM scan_state WHERE id=1").fetchone()
            if row["complete"]:
                generation = row["generation"] + 1
                db.execute("UPDATE scan_state SET generation=?, complete=0, seen=0, queued=0, "
                           "unchanged=0, excluded=0, missing=0, errors=0, finished_at=0 WHERE id=1",
                           (generation,))
                return generation
            return row["generation"]

    def scan_seen(self, source_key: str, generation: int) -> bool:
        with self._connect() as db:
            row = db.execute("SELECT 1 FROM scan_seen WHERE source_key=? AND generation=?",
                             (source_key, generation)).fetchone()
            return row is not None

    def record_scan_batch(self, generation: int, outcomes: list[tuple[str, str]]) -> None:
        if not outcomes:
            return
        counts = {name: 0 for name in ("queued", "unchanged", "excluded", "missing", "errors")}
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute("SELECT generation, complete FROM scan_state WHERE id=1").fetchone()
            if current["generation"] != generation or current["complete"]:
                raise RuntimeError("Capture scan generation changed")
            for key, outcome in outcomes:
                if outcome not in counts:
                    raise ValueError("Invalid capture scan outcome")
                db.execute("INSERT INTO scan_seen (source_key, generation) VALUES (?, ?) "
                           "ON CONFLICT(source_key) DO UPDATE SET generation=excluded.generation",
                           (key, generation))
                counts[outcome] += 1
            db.execute("UPDATE scan_state SET seen=seen+?, queued=queued+?, unchanged=unchanged+?, "
                       "excluded=excluded+?, missing=missing+?, errors=errors+? WHERE id=1",
                       (len(outcomes), counts["queued"], counts["unchanged"], counts["excluded"],
                        counts["missing"], counts["errors"]))

    def finish_scan(self, generation: int) -> dict[str, int | float]:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT generation FROM scan_state WHERE id=1").fetchone()
            if row["generation"] != generation:
                raise RuntimeError("Capture scan generation changed")
            db.execute("UPDATE scan_state SET complete=1, finished_at=? WHERE id=1", (time.time(),))
            result = db.execute("SELECT * FROM scan_state WHERE id=1").fetchone()
            return {key: result[key] for key in result.keys() if key != "id"}
