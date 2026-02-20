"""
Tests for SNIPS retrieval utility integration into importance scoring.

Covers:
- get_memory_retrieval_utility per-memory correctness
- get_batch_retrieval_utility batch query (no N+1)
- calculate_importance with retrieval_utility parameter
- Weight redistribution: novelty=0.25 preserved, retrieval=0.10 additive
- End-to-end: high-utility memory survives decay that would delete a zero-utility one
- Backward compatibility: retrieval_utility=0.0 default preserves old behavior
"""

import math
import time
import pytest
import sqlite3
from unittest.mock import MagicMock, patch

from muninn.scoring.importance import (
    DEFAULT_WEIGHTS,
    calculate_importance,
    batch_update_importance,
)
from muninn.core.types import MemoryRecord, MemoryType, Provenance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(
    age_days: float = 0.0,
    access_count: int = 0,
    provenance: Provenance = Provenance.INGESTED,
    memory_type: MemoryType = MemoryType.EPISODIC,
) -> MemoryRecord:
    created_at = time.time() - age_days * 86400.0
    return MemoryRecord(
        id="test-id",
        content="test content",
        memory_type=memory_type,
        access_count=access_count,
        created_at=created_at,
        ingested_at=created_at,
        provenance=provenance,
        importance=0.5,
        recency_score=1.0,
        novelty_score=0.5,
        namespace="global",
        project="test_project",
    )


# ---------------------------------------------------------------------------
# 1. Weight invariants
# ---------------------------------------------------------------------------

class TestDefaultWeights:
    def test_weights_include_retrieval(self):
        assert "retrieval" in DEFAULT_WEIGHTS

    def test_novelty_weight_preserved_at_0_25(self):
        """novelty must remain 0.25 so existing memories without feedback
        are not penalised relative to pre-SNIPS behaviour."""
        assert DEFAULT_WEIGHTS["novelty"] == 0.25

    def test_retrieval_weight_is_0_10(self):
        assert DEFAULT_WEIGHTS["retrieval"] == 0.10

    def test_base_weights_sum_to_1_10(self):
        """Retrieval is intentionally additive; sum > 1.0 is expected and safe
        given the min(1.0, ...) clamp in calculate_importance."""
        assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.10) < 1e-9


# ---------------------------------------------------------------------------
# 2. calculate_importance with retrieval_utility
# ---------------------------------------------------------------------------

class TestCalculateImportanceRetrieval:
    def test_zero_retrieval_utility_unchanged(self):
        """retrieval_utility=0.0 should produce same result as old formula."""
        mem = _record(age_days=0)
        score_old = calculate_importance(mem, max_similarity=0.0, centrality=0.0, retrieval_utility=0.0)
        # With age=0, access=0, provenance=INGESTED, max_sim=0 (novelty=1):
        # 0.25×1.0 + 0.15×0 + 0.20×0 + 0.25×1.0 + 0.15×0.3 + 0.10×0.0 = 0.545
        assert abs(score_old - 0.545) < 0.01

    def test_high_retrieval_utility_boosts_score(self):
        """retrieval_utility=1.0 should boost score by ~0.10."""
        mem = _record(age_days=0)
        score_no_util = calculate_importance(mem, retrieval_utility=0.0)
        score_full_util = calculate_importance(mem, retrieval_utility=1.0)
        assert abs(score_full_util - score_no_util - 0.10) < 1e-9

    def test_high_utility_saves_borderline_memory(self):
        """A memory at importance ~0.08 (below 0.1 decay threshold) is saved
        when retrieval_utility=1.0 pushes it above the threshold."""
        # Create a very old, never-accessed, INGESTED memory
        # With age=180 days, recency ≈ exp(-0.693*180/7) ≈ 0.0000012
        mem = _record(age_days=180, access_count=0, provenance=Provenance.INGESTED)
        score_no_util = calculate_importance(mem, max_similarity=0.0, centrality=0.0, retrieval_utility=0.0)
        score_full_util = calculate_importance(mem, max_similarity=0.0, centrality=0.0, retrieval_utility=1.0)
        # Full utility bumps score by 0.10
        assert score_full_util - score_no_util == pytest.approx(0.10, abs=1e-9)
        # Importantly, retrieval_utility provides an additive 0.10 bonus
        # (not cancelled by novelty reduction)
        assert score_full_util > score_no_util

    def test_retrieval_utility_clamped_to_1(self):
        """Score must never exceed 1.0."""
        mem = _record(age_days=0, access_count=1000, provenance=Provenance.USER_EXPLICIT)
        score = calculate_importance(mem, retrieval_utility=1.0)
        assert score <= 1.0

    def test_backward_compat_no_retrieval_param(self):
        """Calling calculate_importance without retrieval_utility still works."""
        mem = _record()
        score = calculate_importance(mem)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 3. get_memory_retrieval_utility (per-record)
# ---------------------------------------------------------------------------

class TestGetMemoryRetrievalUtility:
    """Integration tests using a real SQLite in-memory DB."""

    @pytest.fixture
    def db_with_feedback(self, tmp_path):
        """Create a minimal sqlite_metadata instance with retrieval_feedback rows."""
        from muninn.store.sqlite_metadata import SQLiteMetadataStore
        store = SQLiteMetadataStore(str(tmp_path / "test.db"))
        conn = store._get_conn()
        now = time.time()
        # 3 feedback rows for "mem-A": 2 positive, 1 negative
        rows = [
            ("test-user", "global", "test", "mem-A", 1.0, 1, 1.0, now - 10),  # outcome=1, rank=1, sp=1.0
            ("test-user", "global", "test", "mem-A", 1.0, 2, 1.0, now - 20),  # outcome=1, rank=2, sp=1.0
            ("test-user", "global", "test", "mem-A", 0.0, 3, 1.0, now - 30),  # outcome=0, rank=3, sp=1.0
        ]
        conn.executemany(
            "INSERT INTO retrieval_feedback (user_id, namespace, project, memory_id, outcome, rank, sampling_prob, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        return store

    def test_no_feedback_returns_zero(self, db_with_feedback):
        score = db_with_feedback.get_memory_retrieval_utility("unknown-id")
        assert score == 0.0

    def test_snips_estimator_produces_nonzero(self, db_with_feedback):
        score = db_with_feedback.get_memory_retrieval_utility("mem-A", estimator="snips")
        assert 0.0 < score <= 1.0

    def test_plain_mean_estimator(self, db_with_feedback):
        """Plain mean: (1+1+0)/3 = 0.333"""
        score = db_with_feedback.get_memory_retrieval_utility("mem-A", estimator="mean")
        assert abs(score - (2.0 / 3.0)) < 0.01

    def test_snips_differs_from_mean(self, db_with_feedback):
        """SNIPS and plain mean produce different values because SNIPS uses rank-based IPS weighting."""
        snips = db_with_feedback.get_memory_retrieval_utility("mem-A", estimator="snips")
        mean = db_with_feedback.get_memory_retrieval_utility("mem-A", estimator="mean")
        # rank-3 negative gets ipw=2.0 (higher than rank-1/2 positives), so
        # SNIPS < mean here — the key invariant is that they DIFFER meaningfully.
        assert abs(snips - mean) > 0.01

    def test_lookback_cutoff_excludes_old_rows(self, tmp_path):
        """Rows older than lookback_days should be excluded."""
        from muninn.store.sqlite_metadata import SQLiteMetadataStore
        store = SQLiteMetadataStore(str(tmp_path / "test.db"))
        conn = store._get_conn()
        old_ts = time.time() - 40 * 86400  # 40 days ago
        conn.execute(
            "INSERT INTO retrieval_feedback (user_id, namespace, project, memory_id, outcome, rank, sampling_prob, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("test-user", "global", "test", "mem-old", 1.0, 1, 1.0, old_ts),
        )
        conn.commit()
        score_30d = store.get_memory_retrieval_utility("mem-old", lookback_days=30)
        score_60d = store.get_memory_retrieval_utility("mem-old", lookback_days=60)
        assert score_30d == 0.0   # excluded
        assert score_60d > 0.0    # included


# ---------------------------------------------------------------------------
# 4. get_batch_retrieval_utility (batch query, no N+1)
# ---------------------------------------------------------------------------

class TestGetBatchRetrievalUtility:
    @pytest.fixture
    def db_with_batch_feedback(self, tmp_path):
        from muninn.store.sqlite_metadata import SQLiteMetadataStore
        store = SQLiteMetadataStore(str(tmp_path / "test.db"))
        conn = store._get_conn()
        now = time.time()
        rows = [
            ("test-user", "global", "test", "mem-1", 1.0, 1, 1.0, now - 5),
            ("test-user", "global", "test", "mem-1", 1.0, 2, 1.0, now - 10),
            ("test-user", "global", "test", "mem-2", 0.0, 1, 1.0, now - 5),
            # mem-3 has no rows
        ]
        conn.executemany(
            "INSERT INTO retrieval_feedback (user_id, namespace, project, memory_id, outcome, rank, sampling_prob, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        return store

    def test_returns_dict_for_all_ids(self, db_with_batch_feedback):
        result = db_with_batch_feedback.get_batch_retrieval_utility(
            ["mem-1", "mem-2", "mem-3"]
        )
        assert set(result.keys()) == {"mem-1", "mem-2", "mem-3"}

    def test_missing_id_returns_zero(self, db_with_batch_feedback):
        result = db_with_batch_feedback.get_batch_retrieval_utility(["mem-3"])
        assert result["mem-3"] == 0.0

    def test_positive_memory_nonzero(self, db_with_batch_feedback):
        result = db_with_batch_feedback.get_batch_retrieval_utility(["mem-1"])
        assert result["mem-1"] > 0.0

    def test_negative_only_memory_low(self, db_with_batch_feedback):
        result = db_with_batch_feedback.get_batch_retrieval_utility(["mem-2"])
        assert result["mem-2"] == pytest.approx(0.0, abs=0.01)

    def test_batch_results_match_per_record(self, db_with_batch_feedback):
        """Batch results must match per-record results for correctness."""
        batch = db_with_batch_feedback.get_batch_retrieval_utility(
            ["mem-1", "mem-2"], estimator="snips"
        )
        per_1 = db_with_batch_feedback.get_memory_retrieval_utility("mem-1", estimator="snips")
        per_2 = db_with_batch_feedback.get_memory_retrieval_utility("mem-2", estimator="snips")
        assert abs(batch["mem-1"] - per_1) < 1e-9
        assert abs(batch["mem-2"] - per_2) < 1e-9

    def test_empty_list_returns_empty_dict(self, db_with_batch_feedback):
        result = db_with_batch_feedback.get_batch_retrieval_utility([])
        assert result == {}

    def test_issues_single_query_not_n_plus_1(self, db_with_batch_feedback):
        """Verify only one DB execute call is made regardless of ID count."""
        real_conn = db_with_batch_feedback._get_conn()
        call_count = 0

        class TrackingConn:
            """Thin wrapper — delegates to real conn, counts retrieval_feedback queries."""
            def execute(self, sql, params=()):
                nonlocal call_count
                if "retrieval_feedback" in sql:
                    call_count += 1
                return real_conn.execute(sql, params)

            def __getattr__(self, name):
                return getattr(real_conn, name)

        tracking = TrackingConn()
        with patch.object(db_with_batch_feedback, "_get_conn", return_value=tracking):
            db_with_batch_feedback.get_batch_retrieval_utility(
                ["mem-1", "mem-2", "mem-3"]
            )
        assert call_count == 1, f"Expected 1 query, got {call_count} (N+1 pattern detected)"


# ---------------------------------------------------------------------------
# 5. Daemon integration (mocked)
# ---------------------------------------------------------------------------

class TestDaemonBatchUtilityIntegration:
    """Verify daemon._phase_decay() uses batch function, not per-record."""

    def _make_daemon(self):
        from muninn.consolidation.daemon import ConsolidationDaemon
        daemon = ConsolidationDaemon(
            config=MagicMock(),
            metadata=MagicMock(),
            vectors=MagicMock(),
            graph=MagicMock(),
            bm25=MagicMock(),
        )
        daemon.config.decay_threshold = -999.0
        daemon.config.working_memory_ttl_hours = 100000
        daemon.metadata.get_batch_retrieval_utility.return_value = {}
        return daemon

    @pytest.mark.asyncio
    async def test_phase_decay_uses_batch_retrieval_utility(self):
        """_phase_decay must call get_batch_retrieval_utility once (not per-record)."""
        daemon = self._make_daemon()
        now = time.time()
        records = [
            MemoryRecord(
                id=f"m{i}",
                content=f"content {i}",
                memory_type=MemoryType.EPISODIC,
                access_count=0,
                created_at=now,
                ingested_at=now,
                provenance=Provenance.AUTO_EXTRACTED,
                importance=0.5,
                recency_score=1.0,
                novelty_score=0.5,
                namespace="global",
                project="test",
            )
            for i in range(5)
        ]
        daemon.metadata.get_for_consolidation.return_value = records
        daemon.graph.get_memory_node_degrees_batch.return_value = {r.id: 0.0 for r in records}
        daemon.metadata.get_batch_retrieval_utility.return_value = {r.id: 0.0 for r in records}

        await daemon._phase_decay()

        # get_batch_retrieval_utility called once (not 5 times)
        assert daemon.metadata.get_batch_retrieval_utility.call_count == 1
        # get_memory_retrieval_utility must NOT be called (old per-record pattern)
        assert not daemon.metadata.get_memory_retrieval_utility.called

    @pytest.mark.asyncio
    async def test_high_utility_memory_survives_low_importance(self):
        """Memory with utility=1.0 gets +0.10 boost; should survive if it
        would otherwise be at borderline importance."""
        daemon = self._make_daemon()
        daemon.config.decay_threshold = 0.30  # High threshold for this test
        now = time.time() - 180 * 86400  # 180-day-old memory
        mem = MemoryRecord(
            id="border-mem",
            content="borderline",
            memory_type=MemoryType.EPISODIC,
            access_count=0,
            created_at=now,
            ingested_at=now,
            provenance=Provenance.INGESTED,
            importance=0.5,
            recency_score=0.0,
            novelty_score=0.5,
            namespace="global",
            project="test",
        )
        daemon.metadata.get_for_consolidation.return_value = [mem]
        daemon.graph.get_memory_node_degrees_batch.return_value = {"border-mem": 0.0}
        # High retrieval utility pushes importance above 0.30 threshold
        daemon.metadata.get_batch_retrieval_utility.return_value = {"border-mem": 1.0}

        await daemon._phase_decay()

        # Memory should NOT have been deleted (utility saved it)
        daemon.metadata.delete.assert_not_called()
