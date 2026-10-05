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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from muninn.history.blind_index import _terms as _search_terms
from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.private_acl import create_private_file, verify_private
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.capture_enrichment import CaptureEnrichmentMixin
from muninn.history.capture_window_jobs import CaptureWindowJobsMixin

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
_ANALYSIS_ACTIVE = {"pending", "running", "retry", "publishing", "publication_pending"}
_ANALYSIS_STATES = _ANALYSIS_ACTIVE | {
    "succeeded",
    "reused",
    "failed",
    "cancelled",
    "not_queued",
    "not_applicable",
    "outcome_unknown",
}
_ANALYSIS_RETRY_CODES = {
    "locked", "worker_timeout", "local_unavailable", "model_unavailable", "deferred",
    "remote_consent_revoked", "daily_zdr_cap_unverified", "gpu_busy",
    "gpu_telemetry_unavailable", "no_eligible_model_fits", "no_chat_model_fits",
    "ollama_model_already_resident", "source_not_remote_safe",
    "remote_admission_threshold_reached", "remote_admission_busy",
}
_LOCAL_OUTPUT_FAILURE_CODES = {
    "json": "local_output_json",
    "cited_schema": "local_output_cited_schema",
    "citation": "local_output_citation",
    "quote_missing_or_ambiguous": "local_output_quote",
    "analysis_schema": "local_output_analysis_schema",
}
_ANALYSIS_TERMINAL_CODES = {
    "invalid_target", "queue_full", "vault_integrity", "snapshot_unavailable",
    "insufficient_context", "outcome_unknown", "unknown", "cancelled",
    "local_output_invalid",
} | set(_LOCAL_OUTPUT_FAILURE_CODES.values())


def analysis_deferral_code(outcome: dict[str, Any]) -> str:
    """Keep fixed local validation categories, never rejected model content."""
    reason = outcome.get("reason", "deferred")
    if not isinstance(reason, str):
        return "unknown"
    if reason != "local_output_invalid":
        return reason
    subcode = outcome.get("output_failure")
    return (_LOCAL_OUTPUT_FAILURE_CODES.get(subcode, reason)
            if isinstance(subcode, str) else reason)


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
    window: dict[str, Any] | None = field(default=None, repr=False)
    extraction: dict[str, Any] | None = field(default=None, repr=False)
    lane: int = 0

    def __repr__(self) -> str:
        return (
            f"AnalysisJob(job_id={self.job_id!r}, vault_id={self.vault_id!r}, "
            f"state={self.state!r}, attempt={self.attempt}, "
            f"provider={self.provider!r}, model={self.model!r})"
        )


class CaptureJournal(CaptureEnrichmentMixin, CaptureWindowJobsMixin):
    def __init__(self, archive: SecureHistoryArchive, *, recover: bool = True,
                 policy_root: Path | None = None):
        self.archive = archive
        self.policy_root = Path(policy_root) if policy_root is not None else archive.root.parent
        # SQLite URI connections require an absolute path even when the archive
        # CLI was given a relative --root.
        self.path = (archive.root / "capture-jobs.db").absolute()
        verify_private(archive.root)
        if not self.path.exists():
            create_private_file(self.path)
        verify_private(self.path)
        self._key = hmac.new(archive._key, b"muninn-capture-journal-key-v1", hashlib.sha256).digest()
        self._aad = b"muninn-capture-locator-v1\0" + archive.vault_id.encode("ascii")
        with self._connect(initialize=True) as db:
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
            columns = {row[1] for row in db.execute("PRAGMA table_info(history_analysis_jobs)")}
            for column, definition in (("sealed_window", "BLOB"), ("sealed_extraction", "BLOB"),
                                       ("extraction_id", "TEXT"), ("sealed_receipt", "BLOB"),
                                       ("sealed_reuse", "BLOB"),
                                       ("cancel_requested", "INTEGER NOT NULL DEFAULT 0"),
                                       ("publication_started", "INTEGER NOT NULL DEFAULT 0"),
                                       ("lane", "INTEGER NOT NULL DEFAULT 0")):
                if column not in columns:
                    db.execute(f"ALTER TABLE history_analysis_jobs ADD COLUMN {column} {definition}")
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
                self._recover_analysis(db, now)
            # An interrupted worker cannot hold a claim after service restart.
            # Read-only backup access must not steal an active worker's claim.
            if recover:
                db.execute("UPDATE jobs SET state='pending', due_at=0 WHERE state='capturing'")
            self._init_enrichment(db)
            self._init_capture_window_jobs(db)

    @contextmanager
    def _connect(self, *, initialize: bool = False) -> Iterator[sqlite3.Connection]:
        verify_private(self.archive.root)
        verify_private(self.path)
        db = sqlite3.connect(f"{self.path.as_uri()}?mode=rw", uri=True, timeout=0.1)
        try:
            db.row_factory = sqlite3.Row
            # Changing database-wide journal mode during ordinary status reads
            # contends with workers. Establish it only before schema setup;
            # runtime connections inspect it without silently repairing drift.
            mode = db.execute("PRAGMA journal_mode=DELETE" if initialize
                              else "PRAGMA journal_mode").fetchone()
            if mode is None or mode[0] != "delete":
                raise VaultIntegrityError("Capture journal mode is unavailable")
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
            self._verify_enrichment(db)
            self._verify_capture_window_jobs(db)
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
                self._validated_analysis_target(row, db)
                if row["sealed_result"] is not None:
                    self._allow_analysis_result(
                        self._open_search(row["sealed_result"], row["job_id"], "analysis-result")
                    )
                self._read_analysis_window(row)
                self._read_extraction(row)
                self._read_publication_receipt(row)
                # Mapped capture reuse was fully checked above. Cheap state
                # validation also rejects stray reuse seals/search-lane reuse,
                # without repeating every source/ledger proof a second time.
                self._capture_reuse_state(row)
            return count

    def verify_publications(self) -> int:
        """Cross-check ACKs in a staged backup; never publish or run inference.

        This supplements, rather than replaces, verify_all(). Components can
        have different snapshot times; extra unacknowledged candidates are
        recoverable, but an ACK without its exact durable candidates is not.
        """
        selection = ("sealed_receipt IS NOT NULL OR "
                     "(state='succeeded' AND publication_started=1 AND sealed_reuse IS NULL)")
        with self._connect() as db:
            if db.execute(f"SELECT 1 FROM history_analysis_jobs WHERE {selection} LIMIT 1").fetchone() is None:
                return 0
        if not (self.archive.root / "memory-ledger" / "ledger.sqlite3").is_file():
            raise VaultIntegrityError("History publication ledger is missing")
        from muninn.history.cited_analysis_source import CitedAnalysisSource

        # Constructors can initialize schemas: keep them outside pinned readers.
        source = CitedAnalysisSource(self.archive)
        try:
            with self._connect() as db, source.ledger.verified_reference_reader() as (contains, _):
                db.execute("BEGIN")
                count = 0
                for row in db.execute(f"SELECT * FROM history_analysis_jobs WHERE {selection}"):
                    stage = self._read_extraction(row)
                    receipt = self._read_publication_receipt(row)
                    if (row["state"] != "succeeded" or not row["publication_started"]
                            or stage is None or receipt is None):
                        raise VaultIntegrityError("History publication state is invalid")
                    refs = receipt["refs"]
                    self._validated_analysis_target(row, db)
                    self._read_analysis_window(row)
                    expected = source.expected_refs(stage["window"], stage["proposals"],
                                                    model_identity=stage["model_identity"])
                    if refs != expected or not contains(refs):
                        raise VaultIntegrityError("History publication references are incomplete")
                    count += 1
                return count
        except (ValueError, RuntimeError) as exc:
            raise VaultIntegrityError("History publication verification failed") from exc

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
                    "SELECT COUNT(*) FROM history_analysis_jobs WHERE state IN ('pending','running','retry','publishing','publication_pending')"
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
        target = self._validated_analysis_target(row)
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
            self._read_analysis_window(row),
            self._read_extraction(row),
            row["lane"],
        )

    @staticmethod
    def _stage_json(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")

    def _window_purpose(self, row, db=None):
        target = self._validated_analysis_target(row, db)
        return "analysis-window-v1:" + hashlib.sha256(self._stage_json(target)).hexdigest()

    def _read_analysis_window(self, row):
        if row["sealed_window"] is None:
            return None
        from muninn.history.cited_analysis_source import CitedAnalysisSource
        try:
            window = CitedAnalysisSource.validate_descriptor(self._open_search(
                row["sealed_window"], row["job_id"], self._window_purpose(row)))
            self._assert_capture_window(row, window)
            return window
        except ValueError as exc:
            raise VaultIntegrityError("Analysis window authentication failed") from exc

    def _validate_extraction(self, stage, *, authenticate=False):
        from muninn.history.cited_analysis_source import CitedAnalysisSource
        from muninn.history.memory_ledger import MemoryLedger, TYPES
        try:
            if (not isinstance(stage, dict)
                    or set(stage) not in ({"format", "window", "proposals", "model_identity", "result"},
                                          {"format", "window", "proposals", "model_identity", "result",
                                           "admission_id"})
                    or type(stage["format"]) is not int or stage["format"] != 1
                    or not MemoryLedger._hex(stage["model_identity"])
                    or not isinstance(stage["proposals"], list) or len(stage["proposals"]) > 12):
                raise ValueError
            if "admission_id" in stage and (not isinstance(stage["admission_id"], str)
                    or not re.fullmatch(r"[0-9a-f]{32}", stage["admission_id"])):
                raise ValueError
            CitedAnalysisSource.validate_descriptor(stage["window"])
            self._allow_analysis_result(stage["result"])
            for proposal in stage["proposals"]:
                if (not isinstance(proposal, dict) or set(proposal) != {"type", "text", "quote", "start"}
                        or not isinstance(proposal["type"], str) or proposal["type"] not in TYPES
                        or any(not isinstance(proposal[k], str) or not 1 <= len(proposal[k]) <= 2048
                               for k in ("text", "quote"))
                        or type(proposal["start"]) is not int or not 0 <= proposal["start"] < 3000):
                    raise ValueError
            raw = self._stage_json(stage)
            if len(raw) > 512000:
                raise ValueError
            # Detach caller-owned mutable dictionaries before authentication.
            checked = json.loads(raw)
            if authenticate:
                CitedAnalysisSource(self.archive).validated_proposals(checked["window"], checked["proposals"])
            return checked
        except (ValueError, TypeError, KeyError, UnicodeError) as exc:
            raise SearchJobError("Invalid cited extraction stage") from exc

    def _extraction_purpose(self, row, window):
        return ("analysis-extraction-v1:" + self._window_purpose(row).split(":", 1)[1]
                + ":" + hashlib.sha256(self._stage_json(window)).hexdigest() + ":" + row["extraction_id"])

    def _read_extraction(self, row):
        if row["sealed_extraction"] is None:
            if row["extraction_id"] is not None or row["sealed_receipt"] is not None:
                raise VaultIntegrityError("Analysis extraction stage is incomplete")
            return None
        from muninn.history.memory_ledger import MemoryLedger
        window = self._read_analysis_window(row)
        if window is None or not MemoryLedger._hex(row["extraction_id"]):
            raise VaultIntegrityError("Analysis extraction identity is invalid")
        try:
            stage = self._validate_extraction(self._open_search(
                row["sealed_extraction"], row["job_id"], self._extraction_purpose(row, window)))
            expected = hmac.new(self._key, b"analysis-stage-v1\0" + self._stage_json(stage),
                                hashlib.sha256).hexdigest()
            if stage["window"] != window or not hmac.compare_digest(expected, row["extraction_id"]):
                raise ValueError
            if row["lane"] == 1:
                provider = stage["result"]["provider"]
                if not (provider == "ollama" and row["remote_dispatched"] == 0
                        or provider == "openrouter" and row["remote_dispatched"] == 1
                        and row["remote_policy_generation"] > 0):
                    raise ValueError
                self._verify_capture_stage_settlement(row, stage)
            return stage
        except (SearchJobError, ValueError, TypeError) as exc:
            raise VaultIntegrityError("Analysis extraction authentication failed") from exc

    def _verify_capture_stage_settlement(self, row, stage):
        if row["lane"] != 1:
            return
        provider = stage["result"]["provider"]
        if provider == "ollama":
            if "admission_id" in stage:
                raise SearchJobError("Local capture stage has remote admission")
            return
        from muninn.history.remote_accounting import AdmissionError, settled_response
        identifier = stage.get("admission_id")
        if not isinstance(identifier, str):
            raise SearchJobError("Remote capture settlement is missing")
        try:
            settled = settled_response(self.policy_root, identifier, row["remote_policy_generation"])
        except AdmissionError as exc:
            raise SearchJobError("Remote capture settlement is unavailable") from exc
        if not settled:
            raise SearchJobError("Remote capture settlement is not verified")

    def _publication_row(self, job_id):
        with self._connect() as db:
            return db.execute("SELECT * FROM history_analysis_jobs WHERE job_id=?", (job_id,)).fetchone()

    def bind_analysis_window(self, job_id, lease_token, window):
        from muninn.history.cited_analysis_source import CitedAnalysisSource
        CitedAnalysisSource(self.archive).reopen(window)  # No journal writer held.
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM history_analysis_jobs WHERE job_id=? AND state='running' "
                             "AND lease_token=? AND lease_until>? AND cancel_requested=0",
                             (job_id, lease_token, time.time())).fetchone()
            if row is None:
                return False
            target = self._analysis_row(row).target
            self._assert_capture_window(row, window)
            if any(window[k] != target[k] for k in ("blob", "sha256", "version")):
                raise SearchJobError("Cited window does not match the queued target")
            old = self._read_analysis_window(row)
            if old is not None and old != window:
                raise SearchJobError("Queued analysis window is immutable")
            if old is None:
                db.execute("UPDATE history_analysis_jobs SET sealed_window=? WHERE job_id=?",
                           (self._seal_search(window, job_id, self._window_purpose(row)), job_id))
            return True

    def stage_analysis(self, job_id, lease_token, stage):
        stage = self._validate_extraction(stage, authenticate=True)  # Outside journal transaction.
        identity = hmac.new(self._key, b"analysis-stage-v1\0" + self._stage_json(stage), hashlib.sha256).hexdigest()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM history_analysis_jobs WHERE job_id=? AND state='running' "
                             "AND lease_token=? AND lease_until>? AND cancel_requested=0",
                             (job_id, lease_token, time.time())).fetchone()
            if row is None:
                return False
            if self._read_analysis_window(row) != stage["window"]:
                raise SearchJobError("Extraction does not match the queued window")
            if row["lane"] == 1:
                provider = stage["result"]["provider"]
                if not (provider == "ollama" and row["remote_dispatched"] == 0
                        or provider == "openrouter" and row["remote_dispatched"] == 1
                        and row["remote_policy_generation"] > 0):
                    raise SearchJobError("Automatic window provider is not authorized")
                self._verify_capture_stage_settlement(row, stage)
            existing = self._read_extraction(row)
            if existing is not None:
                if existing != stage:
                    raise SearchJobError("Queued extraction stage is immutable")
                return True
            bound = dict(row)
            bound["extraction_id"] = identity
            sealed = self._seal_search(stage, job_id, self._extraction_purpose(bound, stage["window"]))
            db.execute("UPDATE history_analysis_jobs SET sealed_extraction=?,extraction_id=? WHERE job_id=?",
                       (sealed, identity, job_id))
            return True

    def begin_publication(self, job_id, lease_token):
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM history_analysis_jobs WHERE job_id=? AND state='running' "
                             "AND lease_token=? AND lease_until>? AND cancel_requested=0",
                             (job_id, lease_token, time.time())).fetchone()
            if row is None or self._read_extraction(row) is None:
                return False
            db.execute("UPDATE history_analysis_jobs SET state='publishing',publication_started=1,updated_at=? "
                       "WHERE job_id=?", (time.time(), job_id))
            return True

    def acknowledge_publication(self, job_id, lease_token, refs):
        from muninn.history.memory_ledger import MemoryLedger
        if not isinstance(refs, list) or len(refs) > 12 or any(not MemoryLedger._hex(ref) for ref in refs):
            raise SearchJobError("Invalid memory publication receipt")
        # Source/ledger authentication may be expensive; never hold the shared
        # journal writer through it. Recheck the lease and stage at final commit.
        before = self._publication_row(job_id)
        if (before is None or before["state"] != "publishing" or before["publication_started"] != 1
                or before["lease_token"] != lease_token or before["lease_until"] <= time.time()):
            return False
        stage_before = self._read_extraction(before)
        if stage_before is None:
            raise SearchJobError("Memory receipt has no extraction stage")
        from muninn.history.cited_analysis_source import CitedAnalysisSource
        source = CitedAnalysisSource(self.archive)
        expected = source.expected_refs(stage_before["window"], stage_before["proposals"],
                                         model_identity=stage_before["model_identity"])
        if refs != expected or not source.ledger.verify_refs(refs):
            raise SearchJobError("Memory receipt has no matching durable records")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM history_analysis_jobs WHERE job_id=? AND state='publishing' "
                             "AND lease_token=? AND lease_until>? AND publication_started=1",
                             (job_id, lease_token, time.time())).fetchone()
            if row is None:
                return False
            stage = self._read_extraction(row)
            if stage is None or row["extraction_id"] != before["extraction_id"]:
                raise SearchJobError("Memory receipt does not match extraction")
            result = stage["result"]
            now = time.time()
            receipt = {"extraction_id": row["extraction_id"], "refs": refs}
            self._ack_capture_window(db, row)
            db.execute("UPDATE history_analysis_jobs SET state='succeeded',sealed_result=?,sealed_receipt=?,"
                       "result_expires_at=?,provider=?,model=?,error_code='',lease_token=NULL,lease_until=NULL,updated_at=? "
                       "WHERE job_id=?", (self._seal_search(result, job_id, "analysis-result"),
                       self._seal_search(receipt, job_id, "analysis-receipt-v1:" + row["extraction_id"]),
                       now + _ANALYSIS_RESULT_TTL, result["provider"], result["model"], now, job_id))
            return True

    def request_analysis_cancel(self, job_id):
        with self._connect() as db:
            cur = db.execute("UPDATE history_analysis_jobs SET cancel_requested=1,"
                             "state=CASE WHEN state='running' THEN state ELSE 'cancelled' END,updated_at=? "
                             "WHERE job_id=? AND publication_started=0 AND state IN ('pending','retry','running')",
                             (time.time(), job_id))
            return cur.rowcount == 1

    def defer_publication(self, job_id, lease_token):
        """Retry only the existing staged local publication, never inference."""
        with self._connect() as db:
            cur = db.execute("UPDATE history_analysis_jobs SET state='publication_pending',"
                             "lease_token=NULL,lease_until=NULL,due_at=?,updated_at=? "
                             "WHERE job_id=? AND state='publishing' AND publication_started=1 "
                             "AND sealed_extraction IS NOT NULL AND lease_token=? AND lease_until>?",
                             (time.time() + 5, time.time(), job_id, lease_token, time.time()))
            return cur.rowcount == 1

    def fail_publication(self, job_id, lease_token, code="vault_integrity"):
        """Stop an invalid admitted stage without turning it into fresh inference."""
        code = code if code in _ANALYSIS_TERMINAL_CODES else "unknown"
        with self._connect() as db:
            cur = db.execute("UPDATE history_analysis_jobs SET state='failed',error_code=?,"
                             "lease_token=NULL,lease_until=NULL,due_at=0,updated_at=? "
                             "WHERE job_id=? AND state='publishing' AND publication_started=1 "
                             "AND lease_token=? AND lease_until>?",
                             (code, time.time(), job_id, lease_token, time.time()))
            return cur.rowcount == 1

    def _read_publication_receipt(self, row):
        if row["sealed_receipt"] is None:
            return None
        from muninn.history.memory_ledger import MemoryLedger
        if self._read_extraction(row) is None:
            raise VaultIntegrityError("Memory receipt has no extraction stage")
        receipt = self._open_search(row["sealed_receipt"], row["job_id"],
                                    "analysis-receipt-v1:" + row["extraction_id"])
        if (not isinstance(receipt, dict) or set(receipt) != {"extraction_id", "refs"}
                or receipt["extraction_id"] != row["extraction_id"]
                or not isinstance(receipt["refs"], list) or len(receipt["refs"]) > 12
                or any(not MemoryLedger._hex(ref) for ref in receipt["refs"])):
            raise VaultIntegrityError("Memory publication receipt is invalid")
        return receipt

    @staticmethod
    def _recover_analysis(db, now):
        db.execute("UPDATE history_analysis_jobs SET state=CASE "
                   "WHEN publication_started=1 THEN 'publication_pending' "
                   "WHEN cancel_requested=1 AND (remote_dispatched=0 OR sealed_extraction IS NOT NULL) THEN 'cancelled' "
                   "WHEN sealed_extraction IS NOT NULL THEN 'retry' "
                   "WHEN remote_dispatched=1 THEN 'outcome_unknown' ELSE 'retry' END,"
                   "error_code=CASE WHEN remote_dispatched=1 AND sealed_extraction IS NULL THEN 'outcome_unknown' ELSE error_code END,"
                   "lease_token=NULL,lease_until=NULL,due_at=0,updated_at=? "
                   "WHERE state IN ('running','publishing') AND lease_until IS NOT NULL AND lease_until<=?", (now, now))

    @staticmethod
    def _foreground_search_pending(db, now):
        return db.execute("SELECT 1 FROM history_search_jobs WHERE state='running' "
                          "OR (state IN ('pending','retry') AND due_at<=?) LIMIT 1", (now,)).fetchone() is not None

    def capture_planning_ready(self) -> bool:
        """Cheap preflight; claims recheck priority after lengthy preparation."""
        with self._connect() as db:
            self._capture_schedule(db)
            return (self._enrichment_baseline(db) is not None
                    and not self._foreground_search_pending(db, time.time())
                    and self._capture_window_capacity(db) > 0)

    def claim_analysis(self, *, include_capture: bool = False,
                       include_search: bool = True,
                       capture_remote_only: bool = False) -> AnalysisJob | None:
        if any(type(value) is not bool for value in
               (include_capture, include_search, capture_remote_only)):
            raise ValueError("Invalid capture lane admission")
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._recover_analysis(db, now)
            foreground = self._foreground_search_pending(db, now)
            maximum_lane = 1 if include_capture and not foreground else 0
            minimum_lane = 0 if include_search else 1
            row = db.execute(
                "SELECT * FROM history_analysis_jobs WHERE state IN ('pending','retry','publication_pending') "
                "AND due_at<=? AND lane>=? AND lane<=? "
                "AND (?=0 OR lane=0 OR publication_started=1 OR (remote_policy_generation>0 "
                "AND NOT(state='retry' AND error_code='source_not_remote_safe'))) "
                "AND (?=1 OR NOT(lane=1 AND state='retry' AND error_code='source_not_remote_safe')) "
                "ORDER BY lane,due_at,created_at LIMIT 1",
                (now, minimum_lane, maximum_lane, int(capture_remote_only),
                 int(self._capture_window_capacity(db) > 0)),
            ).fetchone()
            if not row:
                return None
            self._validated_analysis_target(row, db)
            token = os.urandom(16).hex()
            db.execute(
                "UPDATE history_analysis_jobs SET state=CASE WHEN publication_started=1 THEN 'publishing' ELSE 'running' END,attempt=attempt+1,lease_token=?,lease_until=?,updated_at=? WHERE job_id=?",
                (token, now + _ANALYSIS_LEASE, now, row["job_id"]),
            )
            row = db.execute("SELECT * FROM history_analysis_jobs WHERE job_id=?", (row["job_id"],)).fetchone()
        return self._analysis_row(row)

    def heartbeat_analysis(self, job_id: str, lease_token: str) -> bool:
        with self._connect() as db:
            cur = db.execute(
                "UPDATE history_analysis_jobs SET lease_until=?,updated_at=? WHERE job_id=? AND state IN ('running','publishing') AND lease_token=? AND lease_until>?",
                (time.time() + _ANALYSIS_LEASE, time.time(), job_id, lease_token, time.time()),
            )
            return cur.rowcount == 1

    def mark_remote_dispatched(self, job_id: str, lease_token: str) -> bool:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM history_analysis_jobs WHERE job_id=? AND state='running' "
                "AND remote_dispatched=0 AND cancel_requested=0 AND lease_token=? AND lease_until>?",
                (job_id, lease_token, time.time()),
            ).fetchone()
            if row is None:
                return False
            if row["lane"] == 1:
                self._validated_analysis_target(row, db)
                if row["remote_policy_generation"] < 1:
                    return False
            elif row["lane"] != 0:
                return False
            cur = db.execute(
                "UPDATE history_analysis_jobs SET remote_dispatched=1,updated_at=? WHERE job_id=? "
                "AND state='running' AND remote_dispatched=0 AND cancel_requested=0 "
                "AND lease_token=? AND lease_until>?",
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
                "UPDATE history_analysis_jobs SET state='succeeded',sealed_result=?,result_expires_at=?,provider=?,model=?,error_code='',lease_token=NULL,lease_until=NULL,updated_at=? WHERE job_id=? AND lane=0 AND state='running' AND lease_token=? AND lease_until>?",
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
                "SELECT attempt,remote_dispatched,cancel_requested,sealed_extraction FROM history_analysis_jobs WHERE job_id=? AND state='running' AND lease_token=? AND lease_until>?",
                (job_id, token, now),
            ).fetchone()
            if not row:
                return False
            state = (
                "outcome_unknown" if row["remote_dispatched"] and row["sealed_extraction"] is None
                else "cancelled" if code == "cancelled" or row["cancel_requested"]
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
                "UPDATE history_analysis_jobs SET cancel_requested=1,state=CASE WHEN remote_dispatched=1 AND sealed_extraction IS NULL AND state='running' THEN 'outcome_unknown' ELSE 'cancelled' END,lease_token=NULL,lease_until=NULL,updated_at=? WHERE job_id=? AND publication_started=0 AND state IN ('pending','retry','running')",
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
        response = {
            "job_id": job.job_id,
            "state": job.state,
            "result": result,
            "provider": job.provider,
            "model": job.model,
            "provisional": True,
            "error_code": row["error_code"] if row["error_code"] in _ANALYSIS_RETRY_CODES | _ANALYSIS_TERMINAL_CODES else None,
            "due_at": row["due_at"] if row["state"] == "retry" else None,
        }
        receipt = self._read_publication_receipt(row)
        if receipt is not None:
            response["memory_refs"] = list(receipt["refs"])
        reuse = self._read_capture_reuse(row)
        if reuse is not None:
            response["memory_refs"] = list(reuse["refs"])
            response["coverage_basis"] = reuse.get("basis", "preserved_parent_analysis")
        return response

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
