# ruff: noqa: E501
"""Owner-only, encrypted-locator journal for strict-history hook capture.

The journal stores no transcript text or plaintext source path. A successful
enqueue means a SQLite FULL-synchronous transaction committed, not that the
source was already archived. One service process owns this database.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from muninn.history.blind_index import _terms as _search_terms
from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.private_acl import create_private_file, verify_private
from muninn.history.secure_archive import SecureHistoryArchive

_SESSION_UUID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.I)
_ERROR_CODES = {"missing", "changed", "permission", "disk", "archive", "locked", "unknown"}
_SEARCH_RETRY_CODES = {"locked", "archive_unavailable", "worker_timeout"}
_SEARCH_TERMINAL_CODES = {"vault_integrity", "invalid_query", "unknown"}
_SEARCH_STATES = {"pending", "running", "retry", "succeeded", "failed", "cancelled"}
_SEARCH_SCHEMA = b"secure-history-search-v1"
_SEARCH_PURPOSE = b"durable-async-search"
_SEARCH_LEASE = 60.0
_SEARCH_RESULT_TTL = 300.0
_ANALYSIS_LEASE = 120.0
_ANALYSIS_RESULT_TTL = 86400.0
_ANALYSIS_ACTIVE = {"pending", "running", "retry"}
_ANALYSIS_STATES = _ANALYSIS_ACTIVE | {
    "succeeded",
    "failed",
    "cancelled",
    "not_queued",
    "not_applicable",
    "outcome_unknown",
}
_ANALYSIS_RETRY_CODES = {
    "locked", "worker_timeout", "local_unavailable", "model_unavailable", "deferred",
    "remote_consent_revoked", "daily_zdr_cap_unverified", "gpu_busy",
}
_ANALYSIS_TERMINAL_CODES = {
    "invalid_target", "queue_full", "vault_integrity", "snapshot_unavailable",
    "insufficient_context", "outcome_unknown", "unknown", "cancelled",
}


@dataclass(frozen=True)
class CaptureJob:
    key: str
    path: Path
    provider: str
    revision: int
    observed_size: int
    observed_mtime_ns: int


@dataclass(frozen=True, repr=False)
class SearchJob:
    job_id: str
    vault_id: str
    state: str
    attempt: int
    lease_token: str | None
    sealed_query: bytes
    sealed_result: bytes | None
    created_at: float
    updated_at: float
    result_expires_at: float | None
    query: str
    limit: int


class SearchJobError(ValueError):
    """A caller-visible validation or state error without sensitive detail."""


@dataclass(frozen=True, repr=False)
class AnalysisJob:
    job_id: str
    vault_id: str
    state: str
    attempt: int
    lease_token: str | None
    created_at: float
    updated_at: float
    result_expires_at: float | None
    provider: str | None = None
    model: str | None = None
    target: dict[str, Any] | None = None
    remote_policy_generation: int = -1

    def __repr__(self) -> str:
        return (
            f"AnalysisJob(job_id={self.job_id!r}, vault_id={self.vault_id!r}, "
            f"state={self.state!r}, attempt={self.attempt}, "
            f"provider={self.provider!r}, model={self.model!r})"
        )


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
            db.execute(
                "CREATE TABLE IF NOT EXISTS jobs ("
                "source_key TEXT PRIMARY KEY, sealed_locator BLOB NOT NULL, "
                "provider TEXT NOT NULL, revision INTEGER NOT NULL, "
                "observed_size INTEGER NOT NULL, observed_mtime_ns INTEGER NOT NULL, "
                "state TEXT NOT NULL, due_at REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, "
                "last_error_code TEXT NOT NULL DEFAULT '', updated_at REAL NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS scan_state ("
                "id INTEGER PRIMARY KEY CHECK(id=1), generation INTEGER NOT NULL, "
                "complete INTEGER NOT NULL, seen INTEGER NOT NULL, queued INTEGER NOT NULL, "
                "unchanged INTEGER NOT NULL, excluded INTEGER NOT NULL, "
                "missing INTEGER NOT NULL, errors INTEGER NOT NULL, finished_at REAL NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS scan_seen (source_key TEXT PRIMARY KEY, generation INTEGER NOT NULL)"
            )
            db.execute("INSERT OR IGNORE INTO scan_state VALUES (1, 0, 1, 0, 0, 0, 0, 0, 0, 0)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS hook_receipts ("
                "provider TEXT NOT NULL, event TEXT NOT NULL, "
                "accepted_invocations INTEGER NOT NULL, last_accepted_at REAL NOT NULL, "
                "last_outcome TEXT NOT NULL, PRIMARY KEY(provider,event))"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS history_search_jobs ("
                "job_id TEXT PRIMARY KEY, vault_id TEXT NOT NULL, sealed_query BLOB NOT NULL, "
                "sealed_result BLOB, state TEXT NOT NULL, attempt INTEGER NOT NULL DEFAULT 0, "
                "lease_token TEXT, lease_until REAL, created_at REAL NOT NULL, updated_at REAL NOT NULL, "
                "result_expires_at REAL, due_at REAL NOT NULL DEFAULT 0, error_code TEXT NOT NULL DEFAULT '')"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS history_analysis_jobs ("
                "job_id TEXT PRIMARY KEY, vault_id TEXT NOT NULL, dedup_key TEXT NOT NULL UNIQUE, "
                "sealed_target BLOB NOT NULL, sealed_result BLOB, state TEXT NOT NULL, attempt INTEGER NOT NULL DEFAULT 0, "
                "lease_token TEXT, lease_until REAL, created_at REAL NOT NULL, updated_at REAL NOT NULL, "
                "due_at REAL NOT NULL DEFAULT 0, result_expires_at REAL, error_code TEXT NOT NULL DEFAULT '', "
                "provider TEXT, model TEXT, remote_dispatched INTEGER NOT NULL DEFAULT 0, "
                "remote_policy_generation INTEGER NOT NULL DEFAULT -1)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS history_analysis_due ON history_analysis_jobs(state,due_at,created_at)"
            )
            try:
                db.execute("ALTER TABLE history_analysis_jobs ADD COLUMN remote_dispatched INTEGER NOT NULL DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            try:
                db.execute("ALTER TABLE history_analysis_jobs ADD COLUMN remote_policy_generation INTEGER NOT NULL DEFAULT -1")
            except sqlite3.OperationalError:
                pass
            for column, definition in (("analysis_job_id", "TEXT"), ("analysis_state", "TEXT")):
                try:
                    db.execute(f"ALTER TABLE history_search_jobs ADD COLUMN {column} {definition}")
                except sqlite3.OperationalError:
                    pass
            try:
                db.execute("ALTER TABLE history_search_jobs ADD COLUMN due_at REAL NOT NULL DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            if recover:
                now = time.time()
                db.execute(
                    "UPDATE history_search_jobs SET state=CASE WHEN attempt < 3 AND ?-created_at <= 3600 THEN 'retry' ELSE 'failed' END, lease_token=NULL, lease_until=NULL, updated_at=? WHERE state='running' AND lease_until IS NOT NULL AND lease_until<=?",
                    (now, now, now),
                )
                db.execute(
                    "UPDATE history_analysis_jobs SET state=CASE WHEN remote_dispatched=1 THEN 'outcome_unknown' WHEN attempt < 3 THEN 'retry' ELSE 'failed' END, "
                    "error_code=CASE WHEN remote_dispatched=1 THEN 'outcome_unknown' ELSE error_code END, lease_token=NULL, lease_until=NULL, due_at=0, updated_at=? "
                    "WHERE state='running' AND lease_until IS NOT NULL AND lease_until<=?",
                    (now, now),
                )
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
            raw = AESGCM(self._key).decrypt(sealed[:12], sealed[12:], self._aad + b"\0" + provider.encode("ascii"))
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

    def enqueue(self, path: Path, provider: str, *, force: bool = False, immediate: bool = False) -> str:
        """Commit a validated locator before the hook may be acknowledged."""
        path = Path(path).resolve(strict=False)
        key = self._source_key(path, provider)
        sealed = self._seal(path, provider)
        size, mtime_ns = self._fingerprint(path)
        now = time.time()
        due = now if force or immediate else now + 2.0
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT revision, observed_size, observed_mtime_ns, state, due_at FROM jobs WHERE source_key=?", (key,)
            ).fetchone()
            if (
                prior
                and (size, mtime_ns) == (prior["observed_size"], prior["observed_mtime_ns"])
                and (not force or prior["state"] in {"pending", "capturing", "retry"})
            ):
                return "coalesced"
            if prior:
                due = min(due, prior["due_at"]) if prior["state"] != "archived" else due
                db.execute(
                    "UPDATE jobs SET sealed_locator=?, provider=?, revision=revision+1, "
                    "observed_size=?, observed_mtime_ns=?, state='pending', due_at=?, "
                    "attempts=0, last_error_code='', updated_at=? WHERE source_key=?",
                    (sealed, provider, size, mtime_ns, due, now, key),
                )
            else:
                db.execute(
                    "INSERT INTO jobs (source_key, sealed_locator, provider, revision, "
                    "observed_size, observed_mtime_ns, state, due_at, updated_at) "
                    "VALUES (?, ?, ?, 1, ?, ?, 'pending', ?, ?)",
                    (key, sealed, provider, size, mtime_ns, due, now),
                )
        return "queued"

    def claim_due(self) -> CaptureJob | None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM jobs WHERE state IN ('pending', 'retry') AND due_at<=? "
                "ORDER BY due_at, updated_at LIMIT 1",
                (time.time(),),
            ).fetchone()
            if row is None:
                return None
            locator = self._open(row["sealed_locator"], row["provider"])
            db.execute(
                "UPDATE jobs SET state='capturing', updated_at=? WHERE source_key=?", (time.time(), row["source_key"])
            )
        return CaptureJob(
            row["source_key"], locator, row["provider"], row["revision"], row["observed_size"], row["observed_mtime_ns"]
        )

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
            if row["revision"] != job.revision or (size, mtime_ns) != (job.observed_size, job.observed_mtime_ns):
                db.execute(
                    "UPDATE jobs SET revision=revision+1, observed_size=?, observed_mtime_ns=?, "
                    "state='pending', due_at=0, updated_at=? WHERE source_key=? AND revision=?",
                    (size, mtime_ns, time.time(), job.key, job.revision),
                )
                return False
            db.execute(
                "UPDATE jobs SET state='archived', last_error_code='', updated_at=? WHERE source_key=? AND revision=?",
                (time.time(), job.key, job.revision),
            )
        return True

    def fail(self, job: CaptureJob, code: str) -> str | None:
        if code not in _ERROR_CODES:
            code = "unknown"
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT revision, attempts FROM jobs WHERE source_key=?", (job.key,)).fetchone()
            if row is None or row["revision"] != job.revision:
                return None
            attempts = row["attempts"] + 1
            # Hooks may report a locator before its transcript exists. Do not
            # wake forever for an absent source. A later file
            # changes its fingerprint and enqueue() reactivates this row.
            state = "unavailable" if code == "missing" and attempts >= 8 else "retry"
            delay = 0 if state == "unavailable" else min(600, 2 ** min(attempts, 9))
            db.execute(
                "UPDATE jobs SET state=?, attempts=?, due_at=?, last_error_code=?, "
                "updated_at=? WHERE source_key=? AND revision=?",
                (state, attempts, time.time() + delay, code, time.time(), job.key, job.revision),
            )
            return state

    def status(self) -> dict[str, int]:
        with self._connect() as db:
            return {
                row["state"]: row["count"]
                for row in db.execute("SELECT state, COUNT(*) AS count FROM jobs GROUP BY state")
            }

    def record_hook_receipt(self, provider: str, event: str, outcome: str) -> None:
        """Count accepted endpoint invocations, never unique host events.

        This is non-secret telemetry only. Capture intent is committed separately
        before this method is called; failure here must not revoke that receipt.
        """
        if provider not in {"codex", "claude_code", "gemini_cli"}:
            raise ValueError("Unsupported hook provider")
        if event not in {"SessionStart", "PreCompact", "PostCompact", "PreCompress",
                         "SessionEnd", "Stop", "AfterAgent"}:
            raise ValueError("Unsupported hook event")
        if outcome not in {"capture_intent", "briefing", "no_transcript"}:
            raise ValueError("Unsupported hook outcome")
        with self._connect() as db:
            db.execute(
                "INSERT INTO hook_receipts(provider,event,accepted_invocations,last_accepted_at,last_outcome) "
                "VALUES (?,?,1,?,?) ON CONFLICT(provider,event) DO UPDATE SET "
                "accepted_invocations=MIN(accepted_invocations+1,9223372036854775807), "
                "last_accepted_at=excluded.last_accepted_at,last_outcome=excluded.last_outcome",
                (provider, event, time.time(), outcome),
            )

    def hook_receipts(self) -> list[dict[str, Any]]:
        """Aggregate, path-free hook acceptance evidence for authenticated status."""
        with self._connect() as db:
            return [dict(row) for row in db.execute(
                "SELECT provider,event,accepted_invocations,last_accepted_at,last_outcome "
                "FROM hook_receipts ORDER BY provider,event"
            )]

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
            for row in db.execute("SELECT * FROM history_search_jobs"):
                if row["vault_id"] != self.archive.vault_id or not re.fullmatch(r"[0-9a-f]{32}", row["job_id"]):
                    raise VaultIntegrityError("Search journal identity is invalid")
                if row["state"] not in _SEARCH_STATES:
                    raise VaultIntegrityError("Search journal state is invalid")
                self._search_row(row)
                if row["sealed_result"] is not None:
                    self._allow_result(self._open_search(row["sealed_result"], row["job_id"], "result"))
            for row in db.execute("SELECT * FROM history_analysis_jobs"):
                if row["vault_id"] != self.archive.vault_id or not re.fullmatch(r"[0-9a-f]{32}", row["job_id"]):
                    raise VaultIntegrityError("Analysis journal identity is invalid")
                target = self._open_search(row["sealed_target"], row["job_id"], "analysis-target")
                if (
                    self._analysis_target(
                        target, target.get("terms", []) if isinstance(target, dict) else [], self.archive.vault_id
                    )
                    is None
                ):
                    raise VaultIntegrityError("Analysis target format is invalid")
                if row["sealed_result"] is not None:
                    self._allow_analysis_result(
                        self._open_search(row["sealed_result"], row["job_id"], "analysis-result")
                    )
            return count

    def _search_key(self, job_id: str) -> bytes:
        return hmac.new(
            self.archive._key, b"muninn-search-job-key-v1\0" + job_id.encode("ascii"), hashlib.sha256
        ).digest()

    def _search_aad(self, job_id: str, purpose: str) -> bytes:
        return (
            b"muninn-search\0"
            + self.archive.vault_id.encode("ascii")
            + b"\0"
            + job_id.encode("ascii")
            + b"\0"
            + _SEARCH_PURPOSE
            + b"\0"
            + _SEARCH_SCHEMA
            + b"\0"
            + purpose.encode("ascii")
        )

    def _seal_search(self, value: Any, job_id: str, purpose: str) -> bytes:
        nonce = os.urandom(12)
        return nonce + AESGCM(self._search_key(job_id)).encrypt(
            nonce,
            json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            self._search_aad(job_id, purpose),
        )

    def _open_search(self, sealed: bytes, job_id: str, purpose: str) -> Any:
        try:
            raw = AESGCM(self._search_key(job_id)).decrypt(sealed[:12], sealed[12:], self._search_aad(job_id, purpose))
            return json.loads(raw.decode("utf-8"))
        except (InvalidTag, ValueError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
            raise VaultIntegrityError("Search journal field authentication failed") from exc

    @staticmethod
    def _validate_search(query: str, terms: list[str] | None, limit: int) -> None:
        if not isinstance(query, str):
            raise SearchJobError("Invalid search request")
        try:
            query_bytes = query.encode("utf-8")
        except UnicodeError as exc:
            raise SearchJobError("Invalid search request") from exc
        if not 1 <= len(query_bytes) <= 512:
            raise SearchJobError("Invalid search request")
        if terms is not None and (
            not isinstance(terms, list)
            or not 1 <= len(terms) <= 8
            or any(not isinstance(t, str) or not 1 <= len(t) <= 512 for t in terms)
        ):
            raise SearchJobError("Invalid search request")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise SearchJobError("Invalid search request")

    def enqueue_search(self, query: str, limit: int = 20) -> str:
        # Bound the input before tokenization, including malformed Unicode.
        self._validate_search(query, None, limit)
        try:
            terms = list(dict.fromkeys(_search_terms(query)))
        except Exception as exc:
            raise SearchJobError("Invalid search request") from exc
        self._validate_search(query, terms, limit)
        now = time.time()
        job_id = os.urandom(16).hex()
        sealed = self._seal_search({"query": query, "terms": terms, "limit": limit}, job_id, "query")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute(
                "SELECT COUNT(*) FROM history_search_jobs WHERE state IN ('pending','running','retry')"
            ).fetchone()[0]
            if count >= 32:
                raise SearchJobError("Search queue is full")
            db.execute(
                "INSERT INTO history_search_jobs(job_id,vault_id,sealed_query,state,created_at,updated_at) VALUES(?,?,?,'pending',?,?)",
                (job_id, self.archive.vault_id, sealed, now, now),
            )
        return job_id

    def _search_row(self, row: sqlite3.Row) -> SearchJob:
        payload = self._open_search(row["sealed_query"], row["job_id"], "query")
        if not isinstance(payload, dict) or set(payload) != {"query", "terms", "limit"}:
            raise VaultIntegrityError("Search journal query format is invalid")
        try:
            self._validate_search(payload["query"], payload["terms"], payload["limit"])
        except SearchJobError as exc:
            raise VaultIntegrityError("Search journal query format is invalid") from exc
        return SearchJob(
            row["job_id"],
            row["vault_id"],
            row["state"],
            row["attempt"],
            row["lease_token"],
            row["sealed_query"],
            row["sealed_result"],
            row["created_at"],
            row["updated_at"],
            row["result_expires_at"],
            payload["query"],
            payload["limit"],
        )

    def claim_search(self) -> SearchJob | None:
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE history_search_jobs SET state=CASE WHEN attempt < 3 AND ?-created_at <= 3600 THEN 'retry' ELSE 'failed' END, lease_token=NULL, lease_until=NULL, due_at=CASE WHEN attempt < 3 AND ?-created_at <= 3600 THEN ? ELSE 0 END, updated_at=? WHERE state='running' AND lease_until IS NOT NULL AND lease_until<=?",
                (now, now, now, now, now),
            )
            row = db.execute(
                "SELECT * FROM history_search_jobs WHERE state IN ('pending','retry') AND due_at<=? AND (state='pending' OR created_at>?) ORDER BY due_at,created_at LIMIT 1",
                (now, now - 3600),
            ).fetchone()
            if row is None:
                return None
            token = os.urandom(16).hex()
            db.execute(
                "UPDATE history_search_jobs SET state='running', attempt=attempt+1, lease_token=?, lease_until=?, due_at=0, updated_at=? WHERE job_id=? AND state IN ('pending','retry')",
                (token, now + _SEARCH_LEASE, now, row["job_id"]),
            )
            row = db.execute("SELECT * FROM history_search_jobs WHERE job_id=?", (row["job_id"],)).fetchone()
        return self._search_row(row)

    def heartbeat_search(self, job_id: str, lease_token: str) -> bool:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cur = db.execute(
                "UPDATE history_search_jobs SET lease_until=?,updated_at=? WHERE job_id=? AND state='running' AND lease_token=? AND lease_until>?",
                (time.time() + _SEARCH_LEASE, time.time(), job_id, lease_token, time.time()),
            )
            return cur.rowcount == 1

    @staticmethod
    def _allow_result(result: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(result, dict) or set(result) != {
            "matches",
            "total",
            "ready",
            "missing",
            "overflow",
            "complete",
            "truncated",
        }:
            raise SearchJobError("Invalid search result")
        if (
            not isinstance(result["matches"], list)
            or len(result["matches"]) > 100
            or any(
                not isinstance(m, dict)
                or set(m)
                != {"ref", "provider", "kind", "captured_day_utc", "size_bucket_kib", "versions", "fetch_capability"}
                for m in result["matches"]
            )
        ):
            raise SearchJobError("Invalid search result")
        if any(type(result[k]) is not int or result[k] < 0 for k in ("total", "ready", "missing", "overflow")):
            raise SearchJobError("Invalid search result")
        if not all(isinstance(result[k], bool) for k in ("complete", "truncated")):
            raise SearchJobError("Invalid search result")
        for match in result["matches"]:
            if (
                not all(
                    isinstance(match[k], str) and len(match[k]) <= 512
                    for k in ("ref", "provider", "kind", "captured_day_utc", "fetch_capability")
                )
                or type(match["size_bucket_kib"]) is not int
                or match["size_bucket_kib"] < 0
                or type(match["versions"]) is not int
                or match["versions"] < 0
            ):
                raise SearchJobError("Invalid search result")
        if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > 100_000:
            raise SearchJobError("Invalid search result")
        return result

    @staticmethod
    def _analysis_target(target: Any, terms: list[str], vault_id: str) -> dict[str, Any] | None:
        if not isinstance(target, dict) or set(target) != {"vault_id", "blob", "sha256", "version", "terms"}:
            return None
        if (
            target["vault_id"] != vault_id
            or not isinstance(target["blob"], str)
            or not re.fullmatch(r"[0-9a-f]{32}", target["blob"])
            or not isinstance(target["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", target["sha256"])
            or type(target["version"]) is not int
            or target["version"] < 0
            or target["terms"] != terms
            or not isinstance(target["terms"], list)
            or not 1 <= len(terms) <= 8
            or any(not isinstance(t, str) or not 3 <= len(t) <= 64 for t in terms)
        ):
            return None
        return {
            "vault_id": vault_id,
            "blob": target["blob"],
            "sha256": target["sha256"],
            "version": target["version"],
            "terms": list(terms),
        }

    @staticmethod
    def _allow_analysis_result(result: Any) -> dict[str, Any]:
        if not isinstance(result, dict) or set(result) != {"status", "provider", "model", "analysis"}:
            raise SearchJobError("Invalid analysis result")
        if result["status"] != "ok" or result["provider"] not in {"ollama", "openrouter"}:
            raise SearchJobError("Invalid analysis result")
        if not isinstance(result["model"], str) or not 1 <= len(result["model"]) <= 128:
            raise SearchJobError("Invalid analysis result")
        analysis = result["analysis"]
        if not isinstance(analysis, dict) or set(analysis) != {"summary", "decisions", "open_items", "uncertainty"}:
            raise SearchJobError("Invalid analysis result")
        if (not isinstance(analysis["summary"], str) or len(analysis["summary"]) > 1200
                or not isinstance(analysis["uncertainty"], str) or len(analysis["uncertainty"]) > 700):
            raise SearchJobError("Invalid analysis result")
        if any(
            not isinstance(analysis[k], list)
            or len(analysis[k]) > 12
            or any(not isinstance(x, str) or len(x) > 350 for x in analysis[k])
            for k in ("decisions", "open_items")
        ):
            raise SearchJobError("Invalid analysis result")
        if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > 50_000:
            raise SearchJobError("Invalid analysis result")
        return result

    def finish_search(
        self, job_id: str, lease_token: str, result: dict[str, Any], *,
        analysis_target: dict[str, Any] | None = None, analysis_reason: str = "target_unavailable",
        remote_policy_generation: int = -1,
    ) -> bool:
        if type(remote_policy_generation) is not int or remote_policy_generation < -1:
            raise ValueError("Invalid remote policy generation")
        result = self._allow_result(result)
        sealed = self._seal_search(result, job_id, "result")
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM history_search_jobs WHERE job_id=? AND state='running' AND lease_token=? AND lease_until>?",
                (job_id, lease_token, now),
            ).fetchone()
            if row is None:
                return False
            payload = self._open_search(row["sealed_query"], job_id, "query")
            terms = payload["terms"]
            target = self._analysis_target(analysis_target, terms, row["vault_id"])
            matches = result["matches"]
            analysis_state = "not_applicable" if not matches else "not_queued"
            analysis_id = None
            analysis_error = "" if not matches else (
                analysis_reason if target is None and analysis_reason in {"disabled", "target_unavailable"}
                else "invalid_target" if target is None else "queue_full"
            )
            if target is not None and matches:
                raw = json.dumps(target, sort_keys=True, separators=(",", ":")).encode()
                dedup = hmac.new(self._key, b"analysis-dedup-v1\0" + raw, hashlib.sha256).hexdigest()
                existing = db.execute("SELECT job_id FROM history_analysis_jobs WHERE dedup_key=?", (dedup,)).fetchone()
                active = db.execute(
                    "SELECT COUNT(*) FROM history_analysis_jobs WHERE state IN ('pending','running','retry')"
                ).fetchone()[0]
                if existing:
                    analysis_id = existing["job_id"]
                    analysis_state = db.execute(
                        "SELECT state FROM history_analysis_jobs WHERE job_id=?", (analysis_id,)
                    ).fetchone()[0]
                elif active < 32:
                    analysis_id = os.urandom(16).hex()
                    db.execute(
                        "INSERT INTO history_analysis_jobs(job_id,vault_id,dedup_key,sealed_target,state,created_at,updated_at,remote_policy_generation) VALUES(?,?,?,?,?,?,?,?)",
                        (
                            analysis_id,
                            row["vault_id"],
                            dedup,
                            self._seal_search(target, analysis_id, "analysis-target"),
                            "pending",
                            now,
                            now,
                            remote_policy_generation,
                        ),
                    )
                    analysis_state = "queued"
                else:
                    analysis_error = "queue_full"
            cur = db.execute(
                "UPDATE history_search_jobs SET state='succeeded',sealed_result=?,result_expires_at=?,analysis_job_id=?,analysis_state=?,error_code=CASE WHEN ?='' THEN error_code ELSE ? END,lease_token=NULL,lease_until=NULL,updated_at=? WHERE job_id=? AND state='running' AND lease_token=? AND lease_until>?",
                (
                    sealed,
                    now + _SEARCH_RESULT_TTL,
                    analysis_id,
                    analysis_state,
                    analysis_error,
                    analysis_error,
                    now,
                    job_id,
                    lease_token,
                    now,
                ),
            )
            return cur.rowcount == 1

    def _analysis_row(self, row: sqlite3.Row) -> AnalysisJob:
        if row["vault_id"] != self.archive.vault_id or row["state"] not in _ANALYSIS_STATES:
            raise VaultIntegrityError("Analysis journal identity is invalid")
        target = self._open_search(row["sealed_target"], row["job_id"], "analysis-target")
        if (
            self._analysis_target(
                target, target.get("terms", []) if isinstance(target, dict) else [], self.archive.vault_id
            )
            is None
        ):
            raise VaultIntegrityError("Analysis target format is invalid")
        return AnalysisJob(
            row["job_id"],
            row["vault_id"],
            row["state"],
            row["attempt"],
            row["lease_token"],
            row["created_at"],
            row["updated_at"],
            row["result_expires_at"],
            row["provider"],
            row["model"],
            target,
            row["remote_policy_generation"],
        )

    def claim_analysis(self) -> AnalysisJob | None:
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE history_analysis_jobs SET state=CASE WHEN remote_dispatched=1 THEN 'outcome_unknown' ELSE 'retry' END,error_code=CASE WHEN remote_dispatched=1 THEN 'outcome_unknown' ELSE error_code END,lease_token=NULL,lease_until=NULL,updated_at=? WHERE state='running' AND lease_until<=?",
                (now, now),
            )
            row = db.execute(
                "SELECT * FROM history_analysis_jobs WHERE state IN ('pending','retry') AND due_at<=? ORDER BY due_at,created_at LIMIT 1",
                (now,),
            ).fetchone()
            if not row:
                return None
            token = os.urandom(16).hex()
            db.execute(
                "UPDATE history_analysis_jobs SET state='running',attempt=attempt+1,lease_token=?,lease_until=?,updated_at=? WHERE job_id=?",
                (token, now + _ANALYSIS_LEASE, now, row["job_id"]),
            )
            row = db.execute("SELECT * FROM history_analysis_jobs WHERE job_id=?", (row["job_id"],)).fetchone()
        return self._analysis_row(row)

    def heartbeat_analysis(self, job_id: str, lease_token: str) -> bool:
        with self._connect() as db:
            cur = db.execute(
                "UPDATE history_analysis_jobs SET lease_until=?,updated_at=? WHERE job_id=? AND state='running' AND lease_token=? AND lease_until>?",
                (time.time() + _ANALYSIS_LEASE, time.time(), job_id, lease_token, time.time()),
            )
            return cur.rowcount == 1

    def mark_remote_dispatched(self, job_id: str, lease_token: str) -> bool:
        with self._connect() as db:
            cur = db.execute(
                "UPDATE history_analysis_jobs SET remote_dispatched=1,updated_at=? WHERE job_id=? AND state='running' AND remote_dispatched=0 AND lease_token=? AND lease_until>?",
                (time.time(), job_id, lease_token, time.time()),
            )
            return cur.rowcount == 1

    def mark_remote_not_sent(self, job_id: str, lease_token: str) -> bool:
        """Clear a pre-HTTP marker only when the fenced caller proved no POST began."""
        with self._connect() as db:
            cur = db.execute(
                "UPDATE history_analysis_jobs SET remote_dispatched=0,updated_at=? "
                "WHERE job_id=? AND state='running' AND remote_dispatched=1 "
                "AND lease_token=? AND lease_until>?",
                (time.time(), job_id, lease_token, time.time()),
            )
            return cur.rowcount == 1

    def finish_analysis(self, job_id: str, lease_token: str, result: dict[str, Any]) -> bool:
        result = self._allow_analysis_result(result)
        now = time.time()
        with self._connect() as db:
            sealed = self._seal_search(result, job_id, "analysis-result")
            cur = db.execute(
                "UPDATE history_analysis_jobs SET state='succeeded',sealed_result=?,result_expires_at=?,provider=?,model=?,error_code='',lease_token=NULL,lease_until=NULL,updated_at=? WHERE job_id=? AND state='running' AND lease_token=? AND lease_until>?",
                (sealed, now + _ANALYSIS_RESULT_TTL, result["provider"], result["model"], now, job_id, lease_token, now),
            )
            return cur.rowcount == 1

    def defer_analysis(self, job_id: str, lease_token: str, error_code: str = "deferred") -> bool:
        return self._finish_analysis_state(job_id, lease_token, error_code, True)

    def fail_analysis(self, job_id: str, lease_token: str, error_code: str = "unknown") -> bool:
        return self._finish_analysis_state(job_id, lease_token, error_code, False)

    def _finish_analysis_state(self, job_id: str, token: str, code: str, retry: bool) -> bool:
        now = time.time()
        code = code if code in _ANALYSIS_RETRY_CODES | _ANALYSIS_TERMINAL_CODES else "unknown"
        with self._connect() as db:
            row = db.execute(
                "SELECT attempt,remote_dispatched FROM history_analysis_jobs WHERE job_id=? AND state='running' AND lease_token=? AND lease_until>?",
                (job_id, token, now),
            ).fetchone()
            if not row:
                return False
            state = (
                "outcome_unknown" if row["remote_dispatched"]
                else "cancelled" if code == "cancelled"
                else "retry" if retry and code in _ANALYSIS_RETRY_CODES
                else "failed"
            )
            if state == "outcome_unknown":
                code = "outcome_unknown"
            cur = db.execute(
                "UPDATE history_analysis_jobs SET state=?,error_code=?,due_at=?,lease_token=NULL,lease_until=NULL,updated_at=? WHERE job_id=? AND state='running' AND lease_token=?",
                (
                    state,
                    code,
                    now + min(600, 2 ** min(row["attempt"], 9)) if state == "retry" else 0,
                    now,
                    job_id,
                    token,
                ),
            )
            return cur.rowcount == 1

    def cancel_analysis(self, job_id: str) -> bool:
        with self._connect() as db:
            cur = db.execute(
                "UPDATE history_analysis_jobs SET state=CASE WHEN remote_dispatched=1 AND state='running' THEN 'outcome_unknown' ELSE 'cancelled' END,lease_token=NULL,lease_until=NULL,updated_at=? WHERE job_id=? AND state IN ('pending','retry','running')",
                (time.time(), job_id),
            )
            return cur.rowcount == 1

    def get_analysis_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM history_analysis_jobs WHERE job_id=?", (job_id,)).fetchone()
        if not row or row["state"] == "cancelled":
            return None
        job = self._analysis_row(row)
        result = None
        if row["state"] == "succeeded" and row["result_expires_at"] and row["result_expires_at"] > time.time():
            result = self._allow_analysis_result(self._open_search(row["sealed_result"], job_id, "analysis-result"))
        return {
            "job_id": job.job_id,
            "state": job.state,
            "result": result,
            "provider": job.provider,
            "model": job.model,
            "provisional": True,
            "error_code": row["error_code"] if row["error_code"] in _ANALYSIS_RETRY_CODES | _ANALYSIS_TERMINAL_CODES else None,
            "due_at": row["due_at"] if row["state"] == "retry" else None,
        }

    def fail_search(self, job_id: str, lease_token: str, error_code: str = "unknown") -> bool:
        now = time.time()
        code = error_code if error_code in _SEARCH_RETRY_CODES | _SEARCH_TERMINAL_CODES else "unknown"
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT attempt,created_at FROM history_search_jobs WHERE job_id=? AND state='running' AND lease_token=? AND lease_until>?",
                (job_id, lease_token, now),
            ).fetchone()
            if row is None:
                return False
            state = (
                "retry"
                if code in _SEARCH_RETRY_CODES and row["attempt"] < 3 and now - row["created_at"] <= 3600
                else "failed"
            )
            due = now + min(600, 2 ** min(row["attempt"], 9)) if state == "retry" else 0
            cur = db.execute(
                "UPDATE history_search_jobs SET state=?,error_code=?,lease_token=NULL,lease_until=NULL,due_at=?,updated_at=? WHERE job_id=? AND state='running' AND lease_token=? AND lease_until>?",
                (state, code, due, now, job_id, lease_token, now),
            )
            return cur.rowcount == 1

    def cancel_search(self, job_id: str) -> bool:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cur = db.execute(
                "UPDATE history_search_jobs SET state='cancelled',lease_token=NULL,lease_until=NULL,updated_at=? WHERE job_id=? AND state IN ('pending','retry','running')",
                (time.time(), job_id),
            )
            return cur.rowcount == 1

    def get_search_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM history_search_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None or row["state"] == "cancelled":
            return None
        if row["state"] == "succeeded" and (
            row["result_expires_at"] is None or row["result_expires_at"] <= time.time()
        ):
            if row["analysis_job_id"] is None:
                return None
            return {
                "job_id": job_id,
                "state": row["state"],
                "result": None,
                "analysis_job_id": row["analysis_job_id"],
                "analysis_state": row["analysis_state"],
            }
        result = (
            self._open_search(row["sealed_result"], job_id, "result")
            if row["state"] == "succeeded" and row["sealed_result"]
            else None
        )
        response = {
            "job_id": job_id,
            "state": row["state"],
            "result": result,
        }
        if row["analysis_state"] is not None:
            response["analysis_state"] = row["analysis_state"]
        if row["analysis_job_id"] is not None:
            response["analysis_job_id"] = row["analysis_job_id"]
        elif row["analysis_state"] == "not_queued":
            response["analysis_reason"] = row["error_code"]
        if row["state"] == "failed":
            response["error_code"] = (
                row["error_code"] if row["error_code"] in _SEARCH_RETRY_CODES | _SEARCH_TERMINAL_CODES else "unknown"
            )
        return response

    def begin_scan(self) -> int:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT generation, complete FROM scan_state WHERE id=1").fetchone()
            if row["complete"]:
                generation = row["generation"] + 1
                db.execute(
                    "UPDATE scan_state SET generation=?, complete=0, seen=0, queued=0, "
                    "unchanged=0, excluded=0, missing=0, errors=0, finished_at=0 WHERE id=1",
                    (generation,),
                )
                return generation
            return row["generation"]

    def scan_seen(self, source_key: str, generation: int) -> bool:
        with self._connect() as db:
            row = db.execute(
                "SELECT 1 FROM scan_seen WHERE source_key=? AND generation=?", (source_key, generation)
            ).fetchone()
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
                db.execute(
                    "INSERT INTO scan_seen (source_key, generation) VALUES (?, ?) "
                    "ON CONFLICT(source_key) DO UPDATE SET generation=excluded.generation",
                    (key, generation),
                )
                counts[outcome] += 1
            db.execute(
                "UPDATE scan_state SET seen=seen+?, queued=queued+?, unchanged=unchanged+?, "
                "excluded=excluded+?, missing=missing+?, errors=errors+? WHERE id=1",
                (
                    len(outcomes),
                    counts["queued"],
                    counts["unchanged"],
                    counts["excluded"],
                    counts["missing"],
                    counts["errors"],
                ),
            )

    def finish_scan(self, generation: int) -> dict[str, int | float]:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT generation FROM scan_state WHERE id=1").fetchone()
            if row["generation"] != generation:
                raise RuntimeError("Capture scan generation changed")
            db.execute("UPDATE scan_state SET complete=1, finished_at=? WHERE id=1", (time.time(),))
            result = db.execute("SELECT * FROM scan_state WHERE id=1").fetchone()
            return {key: result[key] for key in result.keys() if key != "id"}
