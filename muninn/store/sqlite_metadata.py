"""
Muninn SQLite Metadata Store
-----------------------------
Persistent storage for memory records, importance scores, access patterns,
and consolidation state. SQLite provides ACID guarantees and zero-config operation.
"""

import sqlite3
import json
import time
import math
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from muninn.core.types import MemoryRecord, MemoryType, Provenance

logger = logging.getLogger("Muninn.SQLite")

SCHEMA_VERSION = 1

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    memory_type     TEXT DEFAULT 'episodic',

    -- Importance Scoring
    importance      REAL DEFAULT 0.5,
    recency_score   REAL DEFAULT 1.0,
    access_count    INTEGER DEFAULT 0,
    novelty_score   REAL DEFAULT 0.5,

    -- Temporal (Bi-Temporal)
    created_at      REAL NOT NULL,
    ingested_at     REAL NOT NULL,
    last_accessed   REAL,
    expires_at      REAL,

    -- Provenance
    source_agent    TEXT DEFAULT 'unknown',
    project         TEXT DEFAULT 'global',
    branch          TEXT,
    namespace       TEXT DEFAULT 'global',
    provenance      TEXT DEFAULT 'auto_extracted',

    -- Project isolation scope (v3.11.0)
    -- "project" = visible only within its project; never returned cross-project
    -- "global"  = always visible regardless of current project
    scope           TEXT NOT NULL DEFAULT 'project',

    -- Embedding Reference
    vector_id       TEXT,
    embedding_model TEXT DEFAULT 'nomic-embed-text',

    -- Consolidation State
    consolidated    INTEGER DEFAULT 0,
    parent_id       TEXT,
    consolidation_gen INTEGER DEFAULT 0,

    -- Flexible Metadata (JSON)
    metadata        TEXT DEFAULT '{}'
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);",
    "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);",
    "CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project);",
    "CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);",
    "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source_agent);",
    "CREATE INDEX IF NOT EXISTS idx_memories_consolidated ON memories(consolidated);",
    # NOTE: idx_memories_scope is NOT here — it must be created AFTER the scope column
    # migration runs in _initialize() (see _ensure_column_exists call below).
]

SCHEMA_META = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

USER_SCOPE_BACKFILL_FAILURES = """
CREATE TABLE IF NOT EXISTS user_scope_backfill_failures (
    memory_id TEXT PRIMARY KEY,
    attempts  INTEGER DEFAULT 1,
    last_error TEXT,
    updated_at REAL NOT NULL
);
"""

PROJECT_GOALS = """
CREATE TABLE IF NOT EXISTS project_goals (
    user_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    project TEXT NOT NULL,
    goal_statement TEXT NOT NULL,
    constraints_json TEXT NOT NULL DEFAULT '[]',
    goal_embedding_json TEXT,
    updated_at REAL NOT NULL,
    PRIMARY KEY (user_id, namespace, project)
);
"""

HANDOFF_EVENT_LEDGER = """
CREATE TABLE IF NOT EXISTS handoff_event_ledger (
    event_id TEXT PRIMARY KEY,
    source TEXT,
    applied_at REAL NOT NULL
);
"""

RETRIEVAL_FEEDBACK = """
CREATE TABLE IF NOT EXISTS retrieval_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    project TEXT NOT NULL,
    query_text TEXT,
    memory_id TEXT,
    outcome REAL NOT NULL,
    rank INTEGER,
    sampling_prob REAL,
    signals_json TEXT NOT NULL DEFAULT '{}',
    source TEXT,
    created_at REAL NOT NULL
);
"""

PROFILE_POLICY_EVENTS = """
CREATE TABLE IF NOT EXISTS profile_policy_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    updates_json TEXT NOT NULL,
    policy_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""

USER_PROFILES = """
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL DEFAULT '{}',
    source TEXT NOT NULL DEFAULT 'unknown',
    updated_at REAL NOT NULL
);
"""


class SQLiteMetadataStore:
    """Manages memory records in SQLite with full CRUD and query capabilities."""

    def __init__(self, db_path):
        self.db_path = Path(db_path) if not isinstance(db_path, Path) else db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._json1_available = False
        self._initialize()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA cache_size=10000;")
        return self._conn

    def _initialize(self):
        conn = self._get_conn()
        self._json1_available = self._detect_json1(conn)
        conn.execute(CREATE_TABLE)
        for idx in CREATE_INDEXES:
            conn.execute(idx)
        if self._json1_available:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_user_id_json ON memories(json_extract(metadata, '$.user_id'));"
            )
        conn.execute(SCHEMA_META)
        conn.execute(USER_SCOPE_BACKFILL_FAILURES)
        conn.execute(PROJECT_GOALS)
        conn.execute(HANDOFF_EVENT_LEDGER)
        conn.execute(RETRIEVAL_FEEDBACK)
        conn.execute(PROFILE_POLICY_EVENTS)
        conn.execute(USER_PROFILES)
        self._ensure_column_exists(conn, "retrieval_feedback", "rank", "INTEGER")
        self._ensure_column_exists(conn, "retrieval_feedback", "sampling_prob", "REAL")
        # v3.11.0: Project isolation scope migration — add column if upgrading from older DB
        # IMPORTANT: idx_memories_scope must be created AFTER this call; it cannot be in
        # CREATE_INDEXES because the column may not exist yet in pre-v3.11.0 databases.
        self._ensure_column_exists(conn, "memories", "scope", "TEXT NOT NULL DEFAULT 'project'")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_scope_time ON retrieval_feedback(user_id, namespace, project, created_at DESC);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_profile_policy_events_created ON profile_policy_events(created_at DESC);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_profiles_updated_at ON user_profiles(updated_at DESC);"
        )
        conn.execute(
            "INSERT OR IGNORE INTO schema_meta (key, value) VALUES (?, ?)",
            ("version", str(SCHEMA_VERSION))
        )
        conn.commit()
        logger.info(f"SQLite metadata store initialized at {self.db_path}")

    @staticmethod
    def _ensure_column_exists(
        conn: sqlite3.Connection,
        table: str,
        column: str,
        column_type: str,
    ) -> None:
        """Ensure a table has a given column; add it if missing."""
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {row[1] for row in rows}
        if column in existing:
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    @staticmethod
    def _detect_json1(conn: sqlite3.Connection) -> bool:
        """Detect whether SQLite JSON1 extension is available."""
        try:
            row = conn.execute("SELECT json_extract('{\"x\": 1}', '$.x')").fetchone()
            return bool(row and row[0] == 1)
        except sqlite3.OperationalError:
            return False

    def _user_id_condition(self) -> str:
        """
        Return SQL condition fragment for user_id filtering.

        Uses exact JSON extraction when available, otherwise falls back to LIKE.
        """
        if self._json1_available:
            return "json_extract(metadata, '$.user_id') = ?"
        return "metadata LIKE ?"

    def _user_id_param(self, user_id: str) -> str:
        """Return parameter value matching `_user_id_condition`."""
        if self._json1_available:
            return user_id
        return f'%\"user_id\": \"{user_id}\"%'

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        d = dict(row)
        d["consolidated"] = bool(d.get("consolidated", 0))
        d["metadata"] = json.loads(d.get("metadata", "{}"))
        try:
            d["memory_type"] = MemoryType(d.get("memory_type", "episodic"))
        except ValueError:
            d["memory_type"] = MemoryType.EPISODIC
        try:
            d["provenance"] = Provenance(d.get("provenance", "auto_extracted"))
        except ValueError:
            d["provenance"] = Provenance.AUTO_EXTRACTED
        # v3.11.0: normalize scope — treat NULL or unknown values as "project" (backward compat)
        if d.get("scope") not in ("project", "global"):
            d["scope"] = "project"
        return MemoryRecord(**d)


    def set_meta(self, key: str, value: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        conn.commit()

    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        conn = self._get_conn()
        row = conn.execute("SELECT value FROM schema_meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        return row[0]

    def record_profile_policy_event(
        self,
        *,
        source: str,
        updates: Dict[str, Any],
        policy: Dict[str, Any],
    ) -> int:
        """Persist runtime profile-policy change event and return event id."""
        conn = self._get_conn()
        cursor = conn.execute(
            """
            INSERT INTO profile_policy_events (source, updates_json, policy_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                source or "unknown",
                json.dumps(updates or {}),
                json.dumps(policy or {}),
                time.time(),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)

    def get_profile_policy_events(self, *, limit: int = 25) -> List[Dict[str, Any]]:
        """Return recent runtime profile-policy change events."""
        safe_limit = max(1, min(500, int(limit)))
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT id, source, updates_json, policy_json, created_at
            FROM profile_policy_events
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()

        events: List[Dict[str, Any]] = []
        for row in rows:
            try:
                updates = json.loads(row["updates_json"] or "{}")
            except json.JSONDecodeError:
                updates = {}
            try:
                policy = json.loads(row["policy_json"] or "{}")
            except json.JSONDecodeError:
                policy = {}
            events.append(
                {
                    "id": int(row["id"]),
                    "source": row["source"],
                    "updates": updates if isinstance(updates, dict) else {},
                    "policy": policy if isinstance(policy, dict) else {},
                    "created_at": float(row["created_at"]),
                }
            )
        return events

    def set_project_goal(
        self,
        *,
        user_id: str,
        namespace: str,
        project: str,
        goal_statement: str,
        constraints: List[str],
        goal_embedding: Optional[List[float]] = None,
    ) -> None:
        """Persist or update a scoped project goal definition."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO project_goals (
                user_id, namespace, project, goal_statement,
                constraints_json, goal_embedding_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, namespace, project) DO UPDATE SET
                goal_statement = excluded.goal_statement,
                constraints_json = excluded.constraints_json,
                goal_embedding_json = excluded.goal_embedding_json,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                namespace,
                project,
                goal_statement,
                json.dumps(constraints or []),
                json.dumps(goal_embedding) if goal_embedding is not None else None,
                time.time(),
            ),
        )
        conn.commit()

    def get_project_goal(
        self,
        *,
        user_id: str,
        namespace: str,
        project: str,
    ) -> Optional[Dict[str, Any]]:
        """Fetch a scoped project goal if present."""
        conn = self._get_conn()
        row = conn.execute(
            """
            SELECT user_id, namespace, project, goal_statement,
                   constraints_json, goal_embedding_json, updated_at
            FROM project_goals
            WHERE user_id = ? AND namespace = ? AND project = ?
            """,
            (user_id, namespace, project),
        ).fetchone()
        if row is None:
            return None

        constraints = []
        embedding = None
        if row["constraints_json"]:
            try:
                parsed = json.loads(row["constraints_json"])
                if isinstance(parsed, list):
                    constraints = [str(item) for item in parsed]
            except json.JSONDecodeError:
                constraints = []
        if row["goal_embedding_json"]:
            try:
                parsed = json.loads(row["goal_embedding_json"])
                if isinstance(parsed, list):
                    embedding = [float(x) for x in parsed]
            except (json.JSONDecodeError, ValueError, TypeError):
                embedding = None

        return {
            "user_id": row["user_id"],
            "namespace": row["namespace"],
            "project": row["project"],
            "goal_statement": row["goal_statement"],
            "constraints": constraints,
            "goal_embedding": embedding,
            "updated_at": row["updated_at"],
        }

    def set_user_profile(
        self,
        *,
        user_id: str,
        profile: Dict[str, Any],
        source: str = "unknown",
    ) -> None:
        """Persist or update an editable scoped user profile/context object."""
        conn = self._get_conn()
        payload = profile if isinstance(profile, dict) else {}
        conn.execute(
            """
            INSERT INTO user_profiles (user_id, profile_json, source, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                profile_json = excluded.profile_json,
                source = excluded.source,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                json.dumps(payload),
                source or "unknown",
                time.time(),
            ),
        )
        conn.commit()

    def get_user_profile(self, *, user_id: str) -> Optional[Dict[str, Any]]:
        """Fetch editable scoped user profile/context object if present."""
        conn = self._get_conn()
        row = conn.execute(
            """
            SELECT user_id, profile_json, source, updated_at
            FROM user_profiles
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
        if row is None:
            return None

        profile: Dict[str, Any] = {}
        raw_profile = row["profile_json"]
        if raw_profile:
            try:
                parsed = json.loads(raw_profile)
                if isinstance(parsed, dict):
                    profile = parsed
            except json.JSONDecodeError:
                profile = {}

        return {
            "user_id": row["user_id"],
            "profile": profile,
            "source": row["source"],
            "updated_at": float(row["updated_at"]),
        }

    def has_handoff_event(self, event_id: str) -> bool:
        """Return True when an idempotency event has already been applied."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT 1 FROM handoff_event_ledger WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        return row is not None

    def record_handoff_event(self, event_id: str, source: str = "unknown") -> bool:
        """Insert event into ledger; returns True when newly inserted."""
        conn = self._get_conn()
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO handoff_event_ledger (event_id, source, applied_at)
            VALUES (?, ?, ?)
            """,
            (event_id, source, time.time()),
        )
        conn.commit()
        return cursor.rowcount > 0

    def add_retrieval_feedback(
        self,
        *,
        user_id: str,
        namespace: str,
        project: str,
        query_text: str,
        memory_id: str,
        outcome: float,
        rank: Optional[int] = None,
        sampling_prob: Optional[float] = None,
        signals: Optional[Dict[str, float]] = None,
        source: str = "unknown",
    ) -> int:
        """
        Persist retrieval feedback for later weighting calibration.

        outcome is expected in [0,1], where 1 = helpful/accepted.
        """
        clamped = max(0.0, min(1.0, float(outcome)))
        normalized_signals: Dict[str, float] = {}
        for key, value in (signals or {}).items():
            if value is None:
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if v > 0:
                normalized_signals[str(key)] = v
        normalized_rank: Optional[int] = None
        if rank is not None:
            try:
                parsed_rank = int(rank)
                if parsed_rank > 0:
                    normalized_rank = parsed_rank
            except (TypeError, ValueError):
                normalized_rank = None

        normalized_sampling_prob: Optional[float] = None
        if sampling_prob is not None:
            try:
                parsed_prob = float(sampling_prob)
                if parsed_prob > 0:
                    normalized_sampling_prob = max(0.0, min(1.0, parsed_prob))
            except (TypeError, ValueError):
                normalized_sampling_prob = None

        conn = self._get_conn()
        cursor = conn.execute(
            """
            INSERT INTO retrieval_feedback (
                user_id, namespace, project, query_text, memory_id,
                outcome, rank, sampling_prob, signals_json, source, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                namespace,
                project,
                query_text,
                memory_id,
                clamped,
                normalized_rank,
                normalized_sampling_prob,
                json.dumps(normalized_signals),
                source,
                time.time(),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)

    def get_feedback_signal_multipliers(
        self,
        *,
        user_id: str,
        namespace: str,
        project: str,
        lookback_days: int = 30,
        min_total_signal_weight: float = 3.0,
        estimator: str = "weighted_mean",
        propensity_floor: float = 0.05,
        min_effective_samples: float = 2.0,
        default_sampling_prob: float = 1.0,
        floor: float = 0.75,
        ceiling: float = 1.25,
    ) -> Dict[str, float]:
        """
        Compute per-signal multipliers from historical retrieval feedback.

        weighted_mean estimator:
            score = weighted_positive / total_weight  (in [0,1])

        snips estimator:
            score = sum(outcome * w_i / p_i) / sum(w_i / p_i)
            where p_i is clipped propensity per event.

            multiplier = floor + score * (ceiling - floor)
        """
        cutoff_ts = time.time() - (max(1, int(lookback_days)) * 86400.0)
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT outcome, rank, sampling_prob, signals_json
            FROM retrieval_feedback
            WHERE user_id = ? AND namespace = ? AND project = ? AND created_at >= ?
            """,
            (user_id, namespace, project, cutoff_ts),
        ).fetchall()
        if not rows:
            return {}

        estimator_mode = (estimator or "weighted_mean").strip().lower()
        use_snips = estimator_mode == "snips"
        if estimator_mode not in {"weighted_mean", "snips"}:
            logger.warning(
                "Unknown retrieval feedback estimator '%s'; falling back to weighted_mean.",
                estimator_mode,
            )

        safe_propensity_floor = max(1e-4, min(1.0, float(propensity_floor)))
        safe_default_sampling_prob = max(safe_propensity_floor, min(1.0, float(default_sampling_prob)))
        safe_min_effective_samples = max(1.0, float(min_effective_samples))
        weighted_positive: Dict[str, float] = {}
        weighted_total: Dict[str, float] = {}
        snips_positive: Dict[str, float] = {}
        # snips_sum_w  = Σ(ipw)   — SNIPS denominator and Kish ESS numerator
        # snips_sum_w2 = Σ(ipw²)  — Kish ESS denominator
        # (Previously there were two dicts—snips_total and snips_sum_w—that
        # both accumulated ipw identically.  Collapsed into snips_sum_w.)
        snips_sum_w: Dict[str, float] = {}
        snips_sum_w2: Dict[str, float] = {}
        for row in rows:
            outcome = max(0.0, min(1.0, float(row["outcome"])))
            rank = row["rank"]
            sampling_prob = row["sampling_prob"]
            rank_propensity = 1.0
            if rank is not None:
                try:
                    rank_value = int(rank)
                except (TypeError, ValueError):
                    rank_value = 0
                if rank_value > 0:
                    rank_propensity = 1.0 / math.log2(rank_value + 1.0)

            base_prob = safe_default_sampling_prob
            if sampling_prob is not None:
                try:
                    parsed_prob = float(sampling_prob)
                    if parsed_prob > 0:
                        base_prob = min(1.0, parsed_prob)
                except (TypeError, ValueError):
                    pass
            propensity = max(safe_propensity_floor, min(1.0, base_prob * rank_propensity))

            try:
                signal_map = json.loads(row["signals_json"] or "{}")
            except json.JSONDecodeError:
                continue
            if not isinstance(signal_map, dict):
                continue

            for key, raw_weight in signal_map.items():
                try:
                    signal_weight = float(raw_weight)
                except (TypeError, ValueError):
                    continue
                if signal_weight <= 0:
                    continue
                signal_name = str(key)
                weighted_positive[signal_name] = weighted_positive.get(signal_name, 0.0) + (outcome * signal_weight)
                weighted_total[signal_name] = weighted_total.get(signal_name, 0.0) + signal_weight
                ipw = signal_weight / propensity
                snips_positive[signal_name] = snips_positive.get(signal_name, 0.0) + (outcome * ipw)
                snips_sum_w[signal_name] = snips_sum_w.get(signal_name, 0.0) + ipw
                snips_sum_w2[signal_name] = snips_sum_w2.get(signal_name, 0.0) + (ipw * ipw)

        multipliers: Dict[str, float] = {}
        for signal_name, total_weight in weighted_total.items():
            if total_weight < float(min_total_signal_weight):
                continue

            if use_snips:
                # snips_sum_w is both the SNIPS denominator (Σipw) and Kish sum_w
                snips_denom = snips_sum_w.get(signal_name, 0.0)
                if snips_denom <= 0.0:
                    continue
                sum_w2 = snips_sum_w2.get(signal_name, 0.0)
                if sum_w2 <= 0.0:
                    continue
                # Kish effective sample size: (Σw)² / Σw²
                effective_samples = (snips_denom * snips_denom) / sum_w2
                if effective_samples < safe_min_effective_samples:
                    continue
                score = snips_positive.get(signal_name, 0.0) / snips_denom
            else:
                score = weighted_positive.get(signal_name, 0.0) / total_weight

            score = max(0.0, min(1.0, score))
            multipliers[signal_name] = float(floor) + score * (float(ceiling) - float(floor))

        return multipliers

    def record_user_scope_backfill_failure(self, memory_id: str, error: str) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO user_scope_backfill_failures (memory_id, attempts, last_error, updated_at)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                attempts = attempts + 1,
                last_error = excluded.last_error,
                updated_at = excluded.updated_at
            """,
            (memory_id, error[:500], time.time()),
        )
        conn.commit()

    def clear_user_scope_backfill_failure(self, memory_id: str) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM user_scope_backfill_failures WHERE memory_id = ?", (memory_id,))
        conn.commit()

    def get_user_scope_backfill_failures(self, limit: int = 1000) -> List[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT memory_id FROM user_scope_backfill_failures ORDER BY updated_at ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [row[0] for row in rows]

    def count_user_scope_backfill_failures(self) -> int:
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM user_scope_backfill_failures").fetchone()
        return row[0] if row else 0


    def get_missing_user_id_records(self, limit: int = 500) -> List[MemoryRecord]:
        """Fetch a batch of records that do not have metadata.user_id set."""
        conn = self._get_conn()
        if self._json1_available:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE json_extract(metadata, '$.user_id') IS NULL
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE metadata NOT LIKE '%"user_id"%'
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def count_missing_user_id(self) -> int:
        """Count records that do not contain metadata.user_id."""
        conn = self._get_conn()
        if self._json1_available:
            row = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE json_extract(metadata, '$.user_id') IS NULL"
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE metadata NOT LIKE '%\"user_id\"%'"
            ).fetchone()
        return row[0] if row else 0

    # --- CRUD Operations ---

    def add(self, record: MemoryRecord) -> str:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO memories (
                id, content, memory_type, importance, recency_score, access_count,
                novelty_score, created_at, ingested_at, last_accessed, expires_at,
                source_agent, project, branch, namespace, provenance,
                vector_id, embedding_model, consolidated, parent_id,
                consolidation_gen, metadata, scope
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id, record.content, record.memory_type.value,
                record.importance, record.recency_score, record.access_count,
                record.novelty_score, record.created_at, record.ingested_at,
                record.last_accessed, record.expires_at,
                record.source_agent, record.project, record.branch,
                record.namespace, record.provenance.value,
                record.vector_id, record.embedding_model,
                int(record.consolidated), record.parent_id,
                record.consolidation_gen, json.dumps(record.metadata),
                record.scope
            )
        )
        conn.commit()
        return record.id

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def update(self, memory_id: str, **kwargs) -> bool:
        conn = self._get_conn()
        if not kwargs:
            return False
        # Serialize metadata if present
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = json.dumps(kwargs["metadata"])
        if "memory_type" in kwargs and isinstance(kwargs["memory_type"], MemoryType):
            kwargs["memory_type"] = kwargs["memory_type"].value
        if "provenance" in kwargs and isinstance(kwargs["provenance"], Provenance):
            kwargs["provenance"] = kwargs["provenance"].value
        if "consolidated" in kwargs and isinstance(kwargs["consolidated"], bool):
            kwargs["consolidated"] = int(kwargs["consolidated"])

        set_clause = ", ".join(f"{k} = ?" for k in kwargs.keys())
        values = list(kwargs.values()) + [memory_id]
        cursor = conn.execute(f"UPDATE memories SET {set_clause} WHERE id = ?", values)
        conn.commit()
        return cursor.rowcount > 0

    def delete(self, memory_id: str) -> bool:
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()
        return cursor.rowcount > 0

    def delete_all(self, user_id: Optional[str] = None, namespace: Optional[str] = None) -> int:
        """Delete memories, optionally scoped by user_id and/or namespace.

        When user_id is provided, only memories belonging to that user are
        deleted (matched via the JSON metadata column).  When namespace is
        also provided the scope narrows to that namespace *and* user.
        Calling with no arguments deletes **all** memories (dangerous).
        """
        conn = self._get_conn()
        conditions: list = []
        params: list = []

        if namespace:
            conditions.append("namespace = ?")
            params.append(namespace)
        if user_id:
            conditions.append(self._user_id_condition())
            params.append(self._user_id_param(user_id))

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        cursor = conn.execute(f"DELETE FROM memories {where}", params)
        count = cursor.rowcount
        conn.commit()
        return count

    # --- Query Operations ---

    def get_all(
        self,
        limit: int = 100,
        offset: int = 0,
        project: Optional[str] = None,
        namespace: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        user_id: Optional[str] = None,
        created_at_min: Optional[float] = None,
        created_at_max: Optional[float] = None,
        scope: Optional[str] = None,
    ) -> List[MemoryRecord]:
        conn = self._get_conn()
        conditions = []
        params: list = []

        if project:
            conditions.append("project = ?")
            params.append(project)
        if namespace:
            conditions.append("namespace = ?")
            params.append(namespace)
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type.value)
        if user_id:
            conditions.append(self._user_id_condition())
            params.append(self._user_id_param(user_id))
        # v3.10.0: Push temporal range filter to SQL for correct historical retrieval.
        if created_at_min is not None:
            conditions.append("created_at >= ?")
            params.append(created_at_min)
        if created_at_max is not None:
            conditions.append("created_at < ?")
            params.append(created_at_max)
        # v3.11.0: Project isolation scope filter.
        if scope is not None:
            conditions.append("scope = ?")
            params.append(scope)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM memories {where} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    # SQLite's default SQLITE_LIMIT_VARIABLE_NUMBER is 999.  RRF fusion can
    # produce hundreds of candidate IDs (limit*6 across signals), so we must
    # chunk large ID lists to avoid "too many SQL variables" errors.
    _SQLITE_MAX_VARS = 900  # conservative headroom below the 999 hard limit

    def get_by_ids(self, memory_ids: List[str]) -> List[MemoryRecord]:
        if not memory_ids:
            return []
        conn = self._get_conn()
        records: List[MemoryRecord] = []
        ids = list(memory_ids)
        for i in range(0, len(ids), self._SQLITE_MAX_VARS):
            chunk = ids[i : i + self._SQLITE_MAX_VARS]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders})", chunk
            ).fetchall()
            records.extend(self._row_to_record(row) for row in rows)
        return records

    def count(self, project: Optional[str] = None, namespace: Optional[str] = None) -> int:
        conn = self._get_conn()
        conditions = []
        params: list = []
        if project:
            conditions.append("project = ?")
            params.append(project)
        if namespace:
            conditions.append("namespace = ?")
            params.append(namespace)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        row = conn.execute(f"SELECT COUNT(*) FROM memories {where}", params).fetchone()
        return row[0] if row else 0

    def record_access(self, memory_id: str):
        conn = self._get_conn()
        conn.execute(
            "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            (time.time(), memory_id)
        )
        conn.commit()

    def record_access_batch(self, memory_ids: List[str]):
        """Update access metrics for multiple memories in a single transaction."""
        if not memory_ids:
            return
        conn = self._get_conn()
        now = time.time()
        placeholders = ",".join("?" for _ in memory_ids)
        conn.execute(
            f"UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id IN ({placeholders})",
            [now] + memory_ids
        )
        conn.commit()

    def get_for_consolidation(
        self,
        memory_type: Optional[MemoryType] = None,
        min_access_count: Optional[int] = None,
        importance_max: Optional[float] = None,
        importance_min: Optional[float] = None,
        consolidated: Optional[bool] = None,
        limit: int = 100,
    ) -> List[MemoryRecord]:
        conn = self._get_conn()
        conditions = []
        params: list = []

        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type.value)
        if min_access_count is not None:
            conditions.append("access_count >= ?")
            params.append(min_access_count)
        if importance_max is not None:
            conditions.append("importance <= ?")
            params.append(importance_max)
        if importance_min is not None:
            conditions.append("importance >= ?")
            params.append(importance_min)
        if consolidated is not None:
            conditions.append("consolidated = ?")
            params.append(int(consolidated))

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM memories {where} ORDER BY importance DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def search_content(self, query: str, limit: int = 20) -> List[MemoryRecord]:
        """Simple LIKE-based text search (BM25 alternative for small datasets).

        LIKE metacharacters % and _ in the caller-supplied query are escaped so
        they are treated as literals rather than wildcards.  The ESCAPE clause
        designates backslash as the escape character.
        """
        safe_query = query.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
        conn = self._get_conn()
        rows = conn.execute(
            r"SELECT * FROM memories WHERE content LIKE ? ESCAPE '\' ORDER BY importance DESC LIMIT ?",
            (f"%{safe_query}%", limit),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_random(self, limit: int = 10) -> List[MemoryRecord]:
        """Fetch a random sample of memory records."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM memories ORDER BY RANDOM() LIMIT ?",
            (limit,)
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None