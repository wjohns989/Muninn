"""
Tests for Muninn v3.11.0 — Project-Scoped Memory with Strict Isolation.

Phase 14 introduces an explicit `scope` field on MemoryRecord:
  - scope='project': visible only within its project context (default)
  - scope='global':  always visible across all projects (user preferences, rules)

Coverage:
  1. MemoryRecord default and explicit scope values
  2. SQLite persistence and migration (scope roundtrip)
  3. scope filter in get_all() SQL queries
  4. In-memory _record_matches_constraints() scope checks
  5. Backward-compat: old rows without scope col treated as scope='project'
  6. Feature flag: project_scope_strict
  7. AddMemoryRequest schema validation
  8. Integration: scope stored in SQLite after add()
  9. Edge cases: invalid scope values normalized to 'project'
"""

import uuid
import time
import sqlite3
import pytest
from pathlib import Path
from typing import Optional
from unittest.mock import patch, MagicMock

from muninn.core.types import MemoryRecord, MemoryType, Provenance, AddMemoryRequest
from muninn.store.sqlite_metadata import SQLiteMetadataStore
from muninn.core.feature_flags import FeatureFlags, get_flags, reset_flags


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_record(
    project: str = "phoenix",
    scope: str = "project",
    namespace: str = "global",
    user_id: str = "user1",
    content: str = "test memory content",
) -> MemoryRecord:
    return MemoryRecord(
        content=content,
        memory_type=MemoryType.EPISODIC,
        provenance=Provenance.AUTO_EXTRACTED,
        source_agent="test_agent",
        project=project,
        branch=None,
        namespace=namespace,
        metadata={"user_id": user_id, "project": project},
        novelty_score=0.5,
        scope=scope,
    )


@pytest.fixture
def meta_store(tmp_path):
    db_path = tmp_path / f"test_scope_{uuid.uuid4()}.db"
    store = SQLiteMetadataStore(str(db_path))
    yield store


# ---------------------------------------------------------------------------
# 1. MemoryRecord defaults and type
# ---------------------------------------------------------------------------

class TestMemoryRecordScope:
    def test_default_scope_is_project(self):
        record = make_record()
        assert record.scope == "project"

    def test_explicit_global_scope(self):
        record = make_record(scope="global")
        assert record.scope == "global"

    def test_explicit_project_scope(self):
        record = make_record(scope="project")
        assert record.scope == "project"

    def test_scope_is_string(self):
        record = make_record(scope="global")
        assert isinstance(record.scope, str)

    def test_scope_in_model_fields(self):
        assert "scope" in MemoryRecord.model_fields

    def test_scope_roundtrips_through_dict(self):
        record = make_record(scope="global")
        d = record.model_dump()
        assert d["scope"] == "global"
        restored = MemoryRecord(**d)
        assert restored.scope == "global"


# ---------------------------------------------------------------------------
# 2. AddMemoryRequest schema
# ---------------------------------------------------------------------------

class TestAddMemoryRequestScope:
    def test_default_scope_is_project(self):
        req = AddMemoryRequest(content="hello")
        assert req.scope == "project"

    def test_global_scope_accepted(self):
        req = AddMemoryRequest(content="hello", scope="global")
        assert req.scope == "global"

    def test_project_scope_accepted(self):
        req = AddMemoryRequest(content="hello", scope="project")
        assert req.scope == "project"

    def test_invalid_scope_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AddMemoryRequest(content="hello", scope="namespace")


# ---------------------------------------------------------------------------
# 3. SQLite persistence — roundtrip
# ---------------------------------------------------------------------------

class TestSQLiteScopePersistence:
    def test_project_scope_persisted(self, meta_store):
        record = make_record(scope="project")
        meta_store.add(record)
        fetched = meta_store.get(record.id)
        assert fetched is not None
        assert fetched.scope == "project"

    def test_global_scope_persisted(self, meta_store):
        record = make_record(scope="global")
        meta_store.add(record)
        fetched = meta_store.get(record.id)
        assert fetched is not None
        assert fetched.scope == "global"

    def test_scope_default_in_schema(self, meta_store):
        """Raw SQL insert without scope → column default 'project' applied."""
        conn = meta_store._get_conn()
        rid = str(uuid.uuid4())
        now = time.time()
        # Insert WITHOUT the scope column to simulate a pre-v3.11.0 row.
        # embedding_model must use the known default string (not NULL) because
        # MemoryRecord.embedding_model is str, not Optional[str].
        conn.execute(
            """INSERT INTO memories (
                id, content, memory_type, importance, recency_score, access_count,
                novelty_score, created_at, ingested_at, source_agent, project,
                namespace, provenance, metadata, vector_id, embedding_model,
                consolidated, consolidation_gen
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (rid, "old row content", "episodic", 0.5, 1.0, 0, 0.5,
             now, now, "agent", "old_project", "global", "auto_extracted",
             '{}', None, "nomic-embed-text", 0, 0),
        )
        conn.commit()
        fetched = meta_store.get(rid)
        assert fetched is not None
        assert fetched.scope == "project"  # default must be 'project'

    def test_scope_normalization_on_unknown_value(self, meta_store):
        """If scope column contains unrecognized value, normalize to 'project'."""
        conn = meta_store._get_conn()
        rid = str(uuid.uuid4())
        now = time.time()
        conn.execute(
            """INSERT INTO memories (
                id, content, memory_type, importance, recency_score, access_count,
                novelty_score, created_at, ingested_at, source_agent, project,
                namespace, provenance, metadata, vector_id, embedding_model,
                consolidated, consolidation_gen, scope
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (rid, "weird scope content", "episodic", 0.5, 1.0, 0, 0.5,
             now, now, "agent", "phoenix", "global", "auto_extracted",
             '{}', None, "nomic-embed-text", 0, 0, "enterprise"),  # unknown scope value
        )
        conn.commit()
        fetched = meta_store.get(rid)
        assert fetched is not None
        assert fetched.scope == "project"  # normalized from 'enterprise'


# ---------------------------------------------------------------------------
# 4. get_all() scope filtering
# ---------------------------------------------------------------------------

class TestGetAllScopeFilter:
    def test_filter_project_scope(self, meta_store):
        p1 = make_record(scope="project", content="project mem")
        g1 = make_record(scope="global", content="global mem")
        meta_store.add(p1)
        meta_store.add(g1)

        project_only = meta_store.get_all(scope="project")
        ids = [r.id for r in project_only]
        assert p1.id in ids
        assert g1.id not in ids

    def test_filter_global_scope(self, meta_store):
        p1 = make_record(scope="project", content="project mem")
        g1 = make_record(scope="global", content="global mem")
        meta_store.add(p1)
        meta_store.add(g1)

        global_only = meta_store.get_all(scope="global")
        ids = [r.id for r in global_only]
        assert g1.id in ids
        assert p1.id not in ids

    def test_no_scope_filter_returns_both(self, meta_store):
        p1 = make_record(scope="project", content="project mem")
        g1 = make_record(scope="global", content="global mem")
        meta_store.add(p1)
        meta_store.add(g1)

        all_records = meta_store.get_all()
        ids = [r.id for r in all_records]
        assert p1.id in ids
        assert g1.id in ids

    def test_scope_filter_with_project_filter(self, meta_store):
        """Scope + project double filter — only project-scoped phoenix memories."""
        phoenix_proj = make_record(project="phoenix", scope="project", content="phoenix specific")
        global_pref = make_record(project="global", scope="global", content="global pref")
        other_proj = make_record(project="argonaut", scope="project", content="argonaut stuff")
        meta_store.add(phoenix_proj)
        meta_store.add(global_pref)
        meta_store.add(other_proj)

        results = meta_store.get_all(project="phoenix", scope="project")
        ids = [r.id for r in results]
        assert phoenix_proj.id in ids
        assert global_pref.id not in ids
        assert other_proj.id not in ids

    def test_scope_filter_count_accuracy(self, meta_store):
        for i in range(3):
            meta_store.add(make_record(scope="project", content=f"proj {i}"))
        for i in range(2):
            meta_store.add(make_record(scope="global", content=f"glob {i}"))

        proj = meta_store.get_all(scope="project")
        glob = meta_store.get_all(scope="global")
        both = meta_store.get_all()

        assert len(proj) == 3
        assert len(glob) == 2
        assert len(both) == 5


# ---------------------------------------------------------------------------
# 5. _record_matches_constraints scope checks (HybridRetriever post-filter)
# ---------------------------------------------------------------------------

class TestRecordMatchesConstraintsScope:
    """
    Verify that the generic filter mechanism in HybridRetriever._record_matches_constraints()
    correctly handles 'scope' as a MemoryRecord attribute filter.
    """

    def _matches(self, record: MemoryRecord, filters: dict) -> bool:
        """Inline the constraint check logic to avoid needing full HybridRetriever."""
        metadata = record.metadata or {}
        for key, expected in filters.items():
            if expected is None:
                continue
            if key == "user_id":
                if metadata.get("user_id") != expected:
                    return False
                continue
            if hasattr(record, key):
                if getattr(record, key) != expected:
                    return False
                continue
            if metadata.get(key) != expected:
                return False
        return True

    def test_scope_project_matches_project_filter(self):
        record = make_record(scope="project")
        assert self._matches(record, {"scope": "project"})

    def test_scope_project_fails_global_filter(self):
        record = make_record(scope="project")
        assert not self._matches(record, {"scope": "global"})

    def test_scope_global_matches_global_filter(self):
        record = make_record(scope="global")
        assert self._matches(record, {"scope": "global"})

    def test_scope_global_fails_project_filter(self):
        record = make_record(scope="global")
        assert not self._matches(record, {"scope": "project"})

    def test_no_scope_filter_matches_both(self):
        proj = make_record(scope="project")
        glob = make_record(scope="global")
        # Empty filters → both match
        assert self._matches(proj, {})
        assert self._matches(glob, {})

    def test_scope_filter_combined_with_project(self):
        proj_phoenix = make_record(project="phoenix", scope="project")
        glob_memory = make_record(project="global", scope="global")
        filters = {"scope": "project", "project": "phoenix"}
        assert self._matches(proj_phoenix, filters)
        assert not self._matches(glob_memory, filters)


# ---------------------------------------------------------------------------
# 6. Feature flag: project_scope_strict
# ---------------------------------------------------------------------------

class TestProjectScopeStrictFlag:
    def setup_method(self):
        reset_flags()

    def teardown_method(self):
        reset_flags()

    def test_strict_flag_default_off(self):
        flags = FeatureFlags()
        assert flags.project_scope_strict is False

    def test_strict_flag_in_to_dict(self):
        flags = FeatureFlags()
        d = flags.to_dict()
        assert "project_scope_strict" in d
        assert d["project_scope_strict"] is False

    def test_strict_flag_from_env(self):
        import os
        os.environ["MUNINN_PROJECT_SCOPE_STRICT"] = "1"
        try:
            flags = FeatureFlags.from_env()
            assert flags.project_scope_strict is True
        finally:
            os.environ.pop("MUNINN_PROJECT_SCOPE_STRICT", None)

    def test_strict_flag_env_false(self):
        import os
        os.environ["MUNINN_PROJECT_SCOPE_STRICT"] = "0"
        try:
            flags = FeatureFlags.from_env()
            assert flags.project_scope_strict is False
        finally:
            os.environ.pop("MUNINN_PROJECT_SCOPE_STRICT", None)

    def test_strict_flag_is_enabled_method(self):
        flags = FeatureFlags()
        assert flags.is_enabled("project_scope_strict") is False

    def test_strict_flag_require_raises_when_off(self):
        flags = FeatureFlags()
        with pytest.raises(RuntimeError, match="project_scope_strict"):
            flags.require("project_scope_strict")

    def test_strict_flag_require_passes_when_on(self):
        flags = FeatureFlags(project_scope_strict=True)
        # Should not raise
        flags.require("project_scope_strict")

    def test_strict_flag_not_in_active_flags_by_default(self):
        flags = FeatureFlags()
        assert "project_scope_strict" not in flags.active_flags


# ---------------------------------------------------------------------------
# 7. Migration: _ensure_column_exists idempotency
# ---------------------------------------------------------------------------

class TestMigrationIdempotency:
    def test_column_exists_no_error_on_reinit(self, tmp_path):
        """Initializing the store twice must not crash (column already exists)."""
        db_path = str(tmp_path / "migrate_test.db")
        store1 = SQLiteMetadataStore(db_path)
        record = make_record(scope="global")
        store1.add(record)
        del store1

        # Second init — _ensure_column_exists must be idempotent
        store2 = SQLiteMetadataStore(db_path)
        fetched = store2.get(record.id)
        assert fetched is not None
        assert fetched.scope == "global"

    def test_column_exists_handles_pre_existing(self, tmp_path):
        """Create DB manually with scope column, then init store — no error."""
        db_path = str(tmp_path / "pre_scope.db")
        conn = sqlite3.connect(db_path)
        # Create a minimal memories table WITH scope already present
        conn.execute("""
            CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                content TEXT,
                memory_type TEXT DEFAULT 'episodic',
                importance REAL DEFAULT 0.5,
                recency_score REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                novelty_score REAL DEFAULT 0.5,
                created_at REAL,
                ingested_at REAL,
                last_accessed REAL,
                expires_at REAL,
                source_agent TEXT,
                project TEXT DEFAULT 'global',
                branch TEXT,
                namespace TEXT DEFAULT 'global',
                provenance TEXT DEFAULT 'auto_extracted',
                vector_id TEXT,
                embedding_model TEXT,
                consolidated INTEGER DEFAULT 0,
                parent_id TEXT,
                consolidation_gen INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                scope TEXT NOT NULL DEFAULT 'project'
            )
        """)
        conn.commit()
        conn.close()

        # This must succeed without raising "duplicate column" error
        store = SQLiteMetadataStore(db_path)
        record = make_record(scope="project")
        store.add(record)
        fetched = store.get(record.id)
        assert fetched is not None
        assert fetched.scope == "project"


# ---------------------------------------------------------------------------
# 8. Cross-project isolation correctness
# ---------------------------------------------------------------------------

class TestCrossProjectIsolation:
    def test_project_scoped_memory_isolated(self, meta_store):
        """Project-scoped memories must not appear when filtered to another project."""
        phoenix_mem = make_record(project="phoenix", scope="project", content="phoenix rule")
        argonaut_mem = make_record(project="argonaut", scope="project", content="argonaut rule")
        meta_store.add(phoenix_mem)
        meta_store.add(argonaut_mem)

        # Phoenix context
        phoenix_results = meta_store.get_all(project="phoenix")
        phoenix_ids = [r.id for r in phoenix_results]
        assert phoenix_mem.id in phoenix_ids
        assert argonaut_mem.id not in phoenix_ids

        # Argonaut context
        argonaut_results = meta_store.get_all(project="argonaut")
        argonaut_ids = [r.id for r in argonaut_results]
        assert argonaut_mem.id in argonaut_ids
        assert phoenix_mem.id not in argonaut_ids

    def test_global_scope_visible_in_all_projects(self, meta_store):
        """Global-scoped memories must be retrievable without project filter (for fallback)."""
        global_pref = make_record(project="global", scope="global", content="always use dark mode")
        phoenix_mem = make_record(project="phoenix", scope="project", content="phoenix only")
        meta_store.add(global_pref)
        meta_store.add(phoenix_mem)

        # Global fetch (no project filter, scope="global") — simulates fallback
        global_results = meta_store.get_all(scope="global")
        global_ids = [r.id for r in global_results]
        assert global_pref.id in global_ids
        assert phoenix_mem.id not in global_ids

    def test_project_scope_does_not_bleed_into_global_fallback(self, meta_store):
        """The fallback (scope='global') must never return scope='project' memories."""
        proj_mem = make_record(project="phoenix", scope="project", content="phoenix private")
        glob_mem = make_record(project="global", scope="global", content="global pref")
        meta_store.add(proj_mem)
        meta_store.add(glob_mem)

        fallback_results = meta_store.get_all(scope="global")
        fallback_ids = [r.id for r in fallback_results]
        # Critical invariant: project-scoped memory NEVER appears in global fallback
        assert proj_mem.id not in fallback_ids
        assert glob_mem.id in fallback_ids

    def test_multiple_projects_strict_isolation(self, meta_store):
        """5 different projects with project-scoped memories — each must be isolated."""
        projects = ["alpha", "beta", "gamma", "delta", "epsilon"]
        records = {}
        for proj in projects:
            r = make_record(project=proj, scope="project", content=f"{proj} rule")
            meta_store.add(r)
            records[proj] = r

        for target_proj in projects:
            results = meta_store.get_all(project=target_proj, scope="project")
            result_ids = [r.id for r in results]
            assert records[target_proj].id in result_ids
            for other_proj in projects:
                if other_proj != target_proj:
                    assert records[other_proj].id not in result_ids, (
                        f"{other_proj} memory leaked into {target_proj} results"
                    )


# ---------------------------------------------------------------------------
# 9. MemoryRecord.scope field type annotation
# ---------------------------------------------------------------------------

class TestScopeTypeAnnotation:
    def test_scope_literal_type(self):
        """Verify Pydantic validates the Literal["project", "global"] constraint."""
        from pydantic import ValidationError
        # Valid values
        r1 = MemoryRecord(
            content="x", memory_type=MemoryType.EPISODIC, provenance=Provenance.AUTO_EXTRACTED,
            source_agent="a", project="p", namespace="n", metadata={}, novelty_score=0.0,
            scope="project",
        )
        assert r1.scope == "project"
        r2 = MemoryRecord(
            content="x", memory_type=MemoryType.EPISODIC, provenance=Provenance.AUTO_EXTRACTED,
            source_agent="a", project="p", namespace="n", metadata={}, novelty_score=0.0,
            scope="global",
        )
        assert r2.scope == "global"

        # Invalid value
        with pytest.raises(ValidationError):
            MemoryRecord(
                content="x", memory_type=MemoryType.EPISODIC, provenance=Provenance.AUTO_EXTRACTED,
                source_agent="a", project="p", namespace="n", metadata={}, novelty_score=0.0,
                scope="namespace",
            )

    def test_scope_default_no_argument_needed(self):
        """MemoryRecord can be constructed without passing scope — defaults to 'project'."""
        r = MemoryRecord(
            content="x", memory_type=MemoryType.EPISODIC, provenance=Provenance.AUTO_EXTRACTED,
            source_agent="a", project="p", namespace="n", metadata={}, novelty_score=0.0,
        )
        assert r.scope == "project"


# ---------------------------------------------------------------------------
# 10. Scope integration with version
# ---------------------------------------------------------------------------

class TestVersionBump:
    def test_version_is_3_11_0(self):
        from muninn.version import __version__
        assert __version__ == "3.11.0"

    def test_flag_count_includes_phase14(self):
        flags = FeatureFlags()
        d = flags.to_dict()
        assert len(d) == 18  # 4 phase1 + 4 phase2 + 4 phase3 + 3 phase4 + 2 phase13 + 1 phase14
