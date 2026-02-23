"""
Mimir Interop Relay — SQLite Store
=====================================
All database operations for the Mimir module.  This module does NOT
touch the main `memories` table — it operates exclusively on the five
Mimir-specific tables added by SQLiteMetadataStore._initialize().

Tables owned by this store:
  - interop_provider_connections
  - interop_provider_secrets
  - interop_consent_log
  - interop_runs
  - interop_audit_events
  - interop_settings  (added by Mimir, not in original report DDL)

Secret encryption:
  Secrets at rest use Fernet symmetric encryption.  The key is read once
  from env var MUNINN_MIMIR_ENCRYPTION_KEY.  If the variable is absent:
  - in dev mode (MUNINN_DEV_MODE=true): a deterministic dev-only key is used.
  - otherwise: initialization fails closed.
  Call MimirStore.set_encryption_key() before any secret operations when a
  proper key is available.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import (
    AuditEvent,
    AuditEventType,
    AuthType,
    ConnectionStatus,
    ConsentType,
    InteropSettings,
    IRPMode,
    ProviderConnection,
    ProviderName,
    RunRecord,
    RunStatus,
)

logger = logging.getLogger("Muninn.Mimir.store")

# ---------------------------------------------------------------------------
# Encryption helpers
# ---------------------------------------------------------------------------

def _make_fernet():
    """Return a Fernet instance, importing lazily to avoid hard dep at import time."""
    try:
        from cryptography.fernet import Fernet
        return Fernet
    except ImportError as exc:
        raise RuntimeError(
            "The 'cryptography' package is required for Mimir secret storage. "
            "Install it with: pip install cryptography"
        ) from exc


def _derive_key_from_env() -> bytes:
    """
    Derive a Fernet-compatible base64url key.
    Priority:
      1) MUNINN_MIMIR_ENCRYPTION_KEY (explicit key/passphrase)
      2) deterministic dev key when MUNINN_DEV_MODE=true
      3) fail closed in production
    """
    raw = os.environ.get("MUNINN_MIMIR_ENCRYPTION_KEY", "").strip()
    if raw:
        import base64
        # Allow raw bytes encoded as base64 (44 chars) or hex (64 chars)
        try:
            key = base64.urlsafe_b64decode(raw + "==")  # tolerant padding
            if len(key) == 32:
                return base64.urlsafe_b64encode(key)
        except Exception:
            pass
        # Treat as raw passphrase — hash to 32 bytes
        import hashlib
        key_bytes = hashlib.sha256(raw.encode()).digest()
        return base64.urlsafe_b64encode(key_bytes)

    if os.environ.get("MUNINN_DEV_MODE", "").lower() == "true":
        import base64
        import hashlib
        # Fixed dev key for deterministic testing/local development
        dev_bytes = hashlib.sha256(b"muninn-mimir-dev-mode-only").digest()
        logger.warning(
            "MUNINN_MIMIR_ENCRYPTION_KEY not set. Using dev-only key because "
            "MUNINN_DEV_MODE=true. SECRETS ARE NOT SECURE."
        )
        return base64.urlsafe_b64encode(dev_bytes)

    # In production, we must fail if no key is provided.
    raise RuntimeError(
        "MUNINN_MIMIR_ENCRYPTION_KEY must be set for production deployments. "
        "Set this environment variable to a 32-byte secret (base64 or passphrase)."
    )


class _SecretEncryptor:
    """Thin wrapper around Fernet for consistent encrypt/decrypt."""

    def __init__(self) -> None:
        Fernet = _make_fernet()
        self._fernet = Fernet(_derive_key_from_env())

    def encrypt(self, plaintext: str) -> bytes:
        return self._fernet.encrypt(plaintext.encode("utf-8"))

    def decrypt(self, ciphertext: bytes) -> str:
        return self._fernet.decrypt(ciphertext).decode("utf-8")


# ---------------------------------------------------------------------------
# SQL DDL constants for Mimir tables
# (These are imported by sqlite_metadata.py and called in _initialize)
# ---------------------------------------------------------------------------

MIMIR_PROVIDER_CONNECTIONS = """
CREATE TABLE IF NOT EXISTS interop_provider_connections (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    provider        TEXT NOT NULL,
    auth_type       TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'active',
    scopes          TEXT,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    last_verified_at REAL,
    metadata_json   TEXT
);
"""

MIMIR_PROVIDER_CONNECTIONS_IDX = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_interop_provider_user
    ON interop_provider_connections(user_id, provider);
"""

MIMIR_PROVIDER_SECRETS = """
CREATE TABLE IF NOT EXISTS interop_provider_secrets (
    connection_id   TEXT PRIMARY KEY
                        REFERENCES interop_provider_connections(id)
                        ON DELETE CASCADE,
    secret_type     TEXT NOT NULL,
    secret_ciphertext BLOB NOT NULL,
    expires_at      REAL,
    rotated_at      REAL
);
"""

MIMIR_CONSENT_LOG = """
CREATE TABLE IF NOT EXISTS interop_consent_log (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    provider        TEXT NOT NULL,
    consent_type    TEXT NOT NULL,
    scope           TEXT NOT NULL,
    granted_at      REAL NOT NULL,
    revoked_at      REAL
);
"""

MIMIR_RUNS = """
CREATE TABLE IF NOT EXISTS interop_runs (
    run_id          TEXT PRIMARY KEY,
    irp_id          TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    created_at      REAL NOT NULL,
    completed_at    REAL,
    mode            TEXT NOT NULL,
    selected_provider TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',
    error_code      TEXT,
    error_message   TEXT,
    input_tokens    INTEGER DEFAULT 0,
    output_tokens   INTEGER DEFAULT 0,
    latency_ms      INTEGER DEFAULT 0,
    prompt_hash     TEXT,
    redaction_count INTEGER DEFAULT 0
);
"""

MIMIR_RUNS_IDX_STATUS = """
CREATE INDEX IF NOT EXISTS idx_interop_runs_status
    ON interop_runs(status);
"""

MIMIR_RUNS_IDX_USER = """
CREATE INDEX IF NOT EXISTS idx_interop_runs_user_id
    ON interop_runs(user_id, created_at DESC);
"""

MIMIR_AUDIT_EVENTS = """
CREATE TABLE IF NOT EXISTS interop_audit_events (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES interop_runs(run_id) ON DELETE CASCADE,
    ts          REAL NOT NULL,
    event_type  TEXT NOT NULL,
    provider    TEXT NOT NULL,
    status      TEXT NOT NULL,
    payload_json TEXT
);
"""

MIMIR_AUDIT_EVENTS_IDX = """
CREATE INDEX IF NOT EXISTS idx_interop_audit_run_id
    ON interop_audit_events(run_id);
"""

MIMIR_AUDIT_EVENTS_IDX_TS = """
CREATE INDEX IF NOT EXISTS idx_interop_audit_ts
    ON interop_audit_events(ts DESC);
"""

MIMIR_SETTINGS = """
CREATE TABLE IF NOT EXISTS interop_settings (
    user_id                  TEXT PRIMARY KEY,
    enabled                  INTEGER NOT NULL DEFAULT 1,
    allowed_targets          TEXT,
    policy_tools             TEXT NOT NULL DEFAULT 'allowed',
    hop_max                  INTEGER NOT NULL DEFAULT 2,
    memory_context_enabled   INTEGER NOT NULL DEFAULT 1,
    audit_retention_days     INTEGER NOT NULL DEFAULT 90,
    updated_at               REAL NOT NULL
);
"""

# List exported for use by sqlite_metadata._initialize()
MIMIR_DDL_STATEMENTS: list[str] = [
    MIMIR_PROVIDER_CONNECTIONS,
    MIMIR_PROVIDER_CONNECTIONS_IDX,
    MIMIR_PROVIDER_SECRETS,
    MIMIR_CONSENT_LOG,
    MIMIR_RUNS,
    MIMIR_RUNS_IDX_STATUS,
    MIMIR_RUNS_IDX_USER,
    MIMIR_AUDIT_EVENTS,
    MIMIR_AUDIT_EVENTS_IDX,
    MIMIR_AUDIT_EVENTS_IDX_TS,
    MIMIR_SETTINGS,
]


# ---------------------------------------------------------------------------
# MimirStore
# ---------------------------------------------------------------------------

class MimirStore:
    """
    All database operations for the Mimir interop relay.

    Accepts a sqlite3.Connection (from SQLiteMetadataStore) so that all
    Muninn tables live in a single WAL-mode database file.  All public
    methods are synchronous — they are called from async FastAPI routes
    via asyncio.to_thread() where latency matters.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._encryptor: Optional[_SecretEncryptor] = None

    def _enc(self) -> _SecretEncryptor:
        if self._encryptor is None:
            self._encryptor = _SecretEncryptor()
        return self._encryptor

    # ------------------------------------------------------------------
    # interop_runs
    # ------------------------------------------------------------------

    def insert_run(self, record: RunRecord) -> None:
        """Persist a new run record (status=pending)."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO interop_runs
                (run_id, irp_id, user_id, created_at, mode, status, prompt_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.run_id,
                record.irp_id,
                record.user_id,
                record.created_at,
                record.mode.value,
                record.status.value,
                record.prompt_hash,
            ),
        )
        self._conn.commit()

    def update_run(self, record: RunRecord) -> None:
        """Update a run record with final results."""
        self._conn.execute(
            """
            UPDATE interop_runs SET
                completed_at    = ?,
                selected_provider = ?,
                status          = ?,
                error_code      = ?,
                error_message   = ?,
                input_tokens    = ?,
                output_tokens   = ?,
                latency_ms      = ?,
                redaction_count = ?
            WHERE run_id = ?
            """,
            (
                record.completed_at,
                record.selected_provider.value if record.selected_provider else None,
                record.status.value,
                record.error_code,
                record.error_message,
                record.input_tokens,
                record.output_tokens,
                record.latency_ms,
                record.redaction_count,
                record.run_id,
            ),
        )
        self._conn.commit()

    def upsert_run(self, record: RunRecord) -> None:
        """Insert or update a run record (idempotent — safe to call at any run status).

        Immutable fields (run_id, irp_id, user_id, created_at, mode, prompt_hash) are
        set on insert and never overwritten on conflict.  Mutable state fields are
        updated unconditionally so the caller can call this once per status transition
        without worrying about whether the row already exists.
        """
        self._conn.execute(
            """
            INSERT INTO interop_runs
                (run_id, irp_id, user_id, created_at, completed_at, mode,
                 selected_provider, status, error_code, error_message,
                 input_tokens, output_tokens, latency_ms, prompt_hash, redaction_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                completed_at      = excluded.completed_at,
                selected_provider = excluded.selected_provider,
                status            = excluded.status,
                error_code        = excluded.error_code,
                error_message     = excluded.error_message,
                input_tokens      = excluded.input_tokens,
                output_tokens     = excluded.output_tokens,
                latency_ms        = excluded.latency_ms,
                redaction_count   = excluded.redaction_count
            """,
            (
                record.run_id,
                record.irp_id,
                record.user_id,
                record.created_at,
                record.completed_at,
                record.mode.value,
                record.selected_provider.value if record.selected_provider else None,
                record.status.value,
                record.error_code,
                record.error_message,
                record.input_tokens,
                record.output_tokens,
                record.latency_ms,
                record.prompt_hash,
                record.redaction_count,
            ),
        )
        self._conn.commit()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM interop_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_runs(
        self,
        user_id: str = "global_user",
        provider: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM interop_runs WHERE user_id = ?"
        params: List[Any] = [user_id]

        if provider:
            query += " AND selected_provider = ?"
            params.append(provider)
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # interop_audit_events
    # ------------------------------------------------------------------

    def insert_audit_event(self, event: AuditEvent) -> None:
        payload_json = json.dumps(event.payload) if event.payload else None
        self._conn.execute(
            """
            INSERT OR IGNORE INTO interop_audit_events
                (id, run_id, ts, event_type, provider, status, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.run_id,
                event.ts,
                event.event_type.value,
                event.provider,
                event.status.value,
                payload_json,
            ),
        )
        self._conn.commit()

    def get_audit_events(
        self,
        run_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT * FROM interop_audit_events
            WHERE run_id = ?
            ORDER BY ts ASC
            LIMIT ?
            """,
            (run_id, limit),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("payload_json"):
                try:
                    d["payload"] = json.loads(d["payload_json"])
                except (json.JSONDecodeError, TypeError):
                    d["payload"] = None
            result.append(d)
        return result

    def purge_old_audit_events(self, retention_days: int = 90) -> int:
        """Delete audit events older than `retention_days`. Returns count deleted."""
        cutoff = time.time() - (retention_days * 86400)
        cur = self._conn.execute(
            "DELETE FROM interop_audit_events WHERE ts < ?", (cutoff,)
        )
        self._conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------
    # interop_provider_connections
    # ------------------------------------------------------------------

    def upsert_connection(self, conn_record: ProviderConnection) -> None:
        scopes_json = json.dumps(conn_record.scopes) if conn_record.scopes else None
        meta_json = json.dumps(conn_record.metadata) if conn_record.metadata else None
        self._conn.execute(
            """
            INSERT INTO interop_provider_connections
                (id, user_id, provider, auth_type, status, scopes,
                 created_at, updated_at, last_verified_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                auth_type        = excluded.auth_type,
                status           = excluded.status,
                scopes           = excluded.scopes,
                updated_at       = excluded.updated_at,
                last_verified_at = excluded.last_verified_at,
                metadata_json    = excluded.metadata_json
            """,
            (
                conn_record.id,
                conn_record.user_id,
                conn_record.provider.value,
                conn_record.auth_type.value,
                conn_record.status.value,
                scopes_json,
                conn_record.created_at,
                conn_record.updated_at,
                conn_record.last_verified_at,
                meta_json,
            ),
        )
        self._conn.commit()

    def get_connection(
        self, user_id: str, provider: str
    ) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            """
            SELECT * FROM interop_provider_connections
            WHERE user_id = ? AND provider = ?
            """,
            (user_id, provider),
        ).fetchone()
        return dict(row) if row else None

    def list_connections(self, user_id: str = "global_user") -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT * FROM interop_provider_connections
            WHERE user_id = ?
            ORDER BY updated_at DESC
            """,
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # interop_provider_secrets
    # ------------------------------------------------------------------

    def store_secret(
        self,
        connection_id: str,
        secret_type: str,
        plaintext: str,
        expires_at: Optional[float] = None,
    ) -> None:
        """Encrypt and store a provider secret."""
        ciphertext = self._enc().encrypt(plaintext)
        self._conn.execute(
            """
            INSERT INTO interop_provider_secrets
                (connection_id, secret_type, secret_ciphertext, expires_at, rotated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(connection_id) DO UPDATE SET
                secret_type       = excluded.secret_type,
                secret_ciphertext = excluded.secret_ciphertext,
                expires_at        = excluded.expires_at,
                rotated_at        = ?
            """,
            (connection_id, secret_type, ciphertext, expires_at, None,
             time.time()),
        )
        self._conn.commit()

    def retrieve_secret(self, connection_id: str) -> Optional[str]:
        """Decrypt and return the plaintext secret for a connection."""
        row = self._conn.execute(
            "SELECT secret_ciphertext, expires_at FROM interop_provider_secrets WHERE connection_id = ?",
            (connection_id,),
        ).fetchone()
        if not row:
            return None
        if row["expires_at"] and time.time() > row["expires_at"]:
            logger.warning("Secret for connection %s has expired", connection_id)
            return None
        return self._enc().decrypt(bytes(row["secret_ciphertext"]))

    def delete_secret(self, connection_id: str) -> None:
        self._conn.execute(
            "DELETE FROM interop_provider_secrets WHERE connection_id = ?",
            (connection_id,),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # interop_consent_log
    # ------------------------------------------------------------------

    def record_consent(
        self,
        consent_id: str,
        user_id: str,
        provider: str,
        consent_type: ConsentType,
        scope: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR IGNORE INTO interop_consent_log
                (id, user_id, provider, consent_type, scope, granted_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (consent_id, user_id, provider, consent_type.value, scope, time.time()),
        )
        self._conn.commit()

    def revoke_consent(self, consent_id: str) -> None:
        self._conn.execute(
            "UPDATE interop_consent_log SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
            (time.time(), consent_id),
        )
        self._conn.commit()

    def get_active_consent(
        self, user_id: str, provider: str, consent_type: str
    ) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            """
            SELECT * FROM interop_consent_log
            WHERE user_id = ? AND provider = ? AND consent_type = ?
                  AND revoked_at IS NULL
            ORDER BY granted_at DESC
            LIMIT 1
            """,
            (user_id, provider, consent_type),
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # interop_settings
    # ------------------------------------------------------------------

    def get_settings(self, user_id: str = "global_user") -> InteropSettings:
        row = self._conn.execute(
            "SELECT * FROM interop_settings WHERE user_id = ?", (user_id,)
        ).fetchone()
        if not row:
            return InteropSettings(user_id=user_id)
        d = dict(row)
        targets_raw = d.get("allowed_targets")
        allowed = json.loads(targets_raw) if targets_raw else ["claude_code", "codex_cli", "gemini_cli"]
        return InteropSettings(
            user_id=d["user_id"],
            enabled=bool(d["enabled"]),
            allowed_targets=allowed,
            policy_tools=d["policy_tools"],
            hop_max=int(d["hop_max"]),
            memory_context_enabled=bool(d["memory_context_enabled"]),
            audit_retention_days=int(d["audit_retention_days"]),
            updated_at=float(d["updated_at"]),
        )

    def update_settings(self, settings: InteropSettings) -> None:
        self._conn.execute(
            """
            INSERT INTO interop_settings
                (user_id, enabled, allowed_targets, policy_tools, hop_max,
                 memory_context_enabled, audit_retention_days, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                enabled                = excluded.enabled,
                allowed_targets        = excluded.allowed_targets,
                policy_tools           = excluded.policy_tools,
                hop_max                = excluded.hop_max,
                memory_context_enabled = excluded.memory_context_enabled,
                audit_retention_days   = excluded.audit_retention_days,
                updated_at             = excluded.updated_at
            """,
            (
                settings.user_id,
                int(settings.enabled),
                json.dumps(settings.allowed_targets),
                settings.policy_tools,
                settings.hop_max,
                int(settings.memory_context_enabled),
                settings.audit_retention_days,
                time.time(),
            ),
        )
        self._conn.commit()
