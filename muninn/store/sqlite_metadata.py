"""
Muninn SQLite Metadata Store
-----------------------------
Persistent storage for memory records, importance scores, access patterns,
and consolidation state. SQLite provides ACID guarantees and zero-config operation.
"""

import sqlite3
import json
import time
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
]

SCHEMA_META = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class SQLiteMetadataStore:
    """Manages memory records in SQLite with full CRUD and query capabilities."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
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
        conn.execute(CREATE_TABLE)
        for idx in CREATE_INDEXES:
            conn.execute(idx)
        conn.execute(SCHEMA_META)
        conn.execute(
            "INSERT OR IGNORE INTO schema_meta (key, value) VALUES (?, ?)",
            ("version", str(SCHEMA_VERSION))
        )
        conn.commit()
        logger.info(f"SQLite metadata store initialized at {self.db_path}")

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
        return MemoryRecord(**d)

    # --- CRUD Operations ---

    def add(self, record: MemoryRecord) -> str:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO memories (
                id, content, memory_type, importance, recency_score, access_count,
                novelty_score, created_at, ingested_at, last_accessed, expires_at,
                source_agent, project, branch, namespace, provenance,
                vector_id, embedding_model, consolidated, parent_id,
                consolidation_gen, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id, record.content, record.memory_type.value,
                record.importance, record.recency_score, record.access_count,
                record.novelty_score, record.created_at, record.ingested_at,
                record.last_accessed, record.expires_at,
                record.source_agent, record.project, record.branch,
                record.namespace, record.provenance.value,
                record.vector_id, record.embedding_model,
                int(record.consolidated), record.parent_id,
                record.consolidation_gen, json.dumps(record.metadata)
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
        conn.execute(f"UPDATE memories SET {set_clause} WHERE id = ?", values)
        conn.commit()
        return conn.total_changes > 0

    def delete(self, memory_id: str) -> bool:
        conn = self._get_conn()
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()
        return conn.total_changes > 0

    def delete_all(self, user_id: Optional[str] = None, namespace: Optional[str] = None) -> int:
        conn = self._get_conn()
        if namespace:
            conn.execute("DELETE FROM memories WHERE namespace = ?", (namespace,))
        else:
            conn.execute("DELETE FROM memories")
        count = conn.total_changes
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

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM memories {where} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_by_ids(self, memory_ids: List[str]) -> List[MemoryRecord]:
        if not memory_ids:
            return []
        conn = self._get_conn()
        placeholders = ",".join("?" for _ in memory_ids)
        rows = conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})", memory_ids
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

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
        """Simple LIKE-based text search (BM25 alternative for small datasets)."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM memories WHERE content LIKE ? ORDER BY importance DESC LIMIT ?",
            (f"%{query}%", limit)
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
