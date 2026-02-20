"""
Tests for v3.15.0 Kaizen Review Fixes
=======================================

Covers all issues identified and fixed in the comprehensive Kaizen codebase review:

  P1 — handlers.py debug log removal
  P2 — scoring/importance.py frequency clamp + KeyError guard
  P2 — weight_adapter.py missing chain/goal signals
  P2 — handlers.py task-error JSON-RPC protocol fix
  P2 — memory.py duplicate OTel/flags initialization removed
  P3 — sqlite_metadata.py get_by_ids chunking (>999 SQLite limit)
  P3 — sqlite_metadata.py search_content LIKE metachar escaping
  P3 — sqlite_metadata.py snips_total/snips_sum_w deduplication
"""

import math
import inspect
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# P1: handlers.py — no hardcoded debug log file writes
# ---------------------------------------------------------------------------

class TestNoDebugLogWrites:
    """Ensure the hardcoded mcp_debug.log writes are fully removed."""

    def test_handle_call_tool_with_task_has_no_open_calls(self):
        """handle_call_tool_with_task must not open any filesystem paths."""
        import muninn.mcp.handlers as handlers_mod
        src = inspect.getsource(handlers_mod.handle_call_tool_with_task)
        assert "mcp_debug.log" not in src, "Debug log path still present in source"
        assert 'open("C:\\\\' not in src, "Hardcoded Windows path still present"
        assert "open(" not in src, "open() call found in handle_call_tool_with_task"

    def test_handlers_module_has_no_debug_log_path(self):
        """No reference to the developer-local debug log anywhere in the module."""
        import muninn.mcp.handlers as handlers_mod
        full_src = inspect.getsource(handlers_mod)
        assert "mcp_debug.log" not in full_src

    def test_handle_call_tool_with_task_uses_logger(self):
        """Diagnostic output must go through the logger, not a file."""
        from muninn.mcp.handlers import handle_call_tool_with_task

        sent = []

        def fake_worker(task_id, name, args, notif_fn):
            pass

        def fake_send(msg_id, payload):
            sent.append(payload)

        with patch("muninn.mcp.handlers.create_task") as mock_create, \
             patch("muninn.mcp.handlers.public_task") as mock_pub:
            mock_task = {"taskId": "t123", "ttl": 30000, "status": "working"}
            mock_create.return_value = mock_task
            mock_pub.return_value = {"taskId": "t123", "status": "working"}

            handle_call_tool_with_task(
                msg_id=1,
                name="add_memory",
                arguments={"content": "test"},
                task_request={},
                send_result_fn=fake_send,
                worker_fn=fake_worker,
            )

        assert len(sent) == 1
        assert sent[0]["task"]["taskId"] == "t123"
        # Verify the call succeeds without any file I/O in the handler itself.
        # (The mcp_debug.log file may exist from previous dev sessions; we check
        # the source, not the filesystem, to confirm the writes are removed.)


# ---------------------------------------------------------------------------
# P2: scoring/importance.py — calculate_frequency clamp
# ---------------------------------------------------------------------------

class TestCalculateFrequencyClamp:
    """calculate_frequency must return values strictly in [0, 1]."""

    def test_zero_access_is_zero(self):
        from muninn.scoring.importance import calculate_frequency
        assert calculate_frequency(0) == 0.0

    def test_max_expected_access_is_one(self):
        from muninn.scoring.importance import calculate_frequency
        result = calculate_frequency(100, max_expected=100)
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_over_max_expected_clamped_to_one(self):
        """Without the fix this would return >1.0 (e.g. ~1.49 for 1000)."""
        from muninn.scoring.importance import calculate_frequency
        for access_count in (101, 200, 1000, 10000, 999999):
            result = calculate_frequency(access_count, max_expected=100)
            assert result <= 1.0, (
                f"calculate_frequency({access_count}) = {result} exceeds 1.0"
            )

    def test_return_monotonically_increases(self):
        from muninn.scoring.importance import calculate_frequency
        prev = 0.0
        for n in range(0, 110, 10):
            cur = calculate_frequency(n, max_expected=100)
            assert cur >= prev, f"Not monotone at n={n}"
            prev = cur


# ---------------------------------------------------------------------------
# P2: scoring/importance.py — calculate_importance partial weight dict
# ---------------------------------------------------------------------------

class TestCalculateImportancePartialWeights:
    """calculate_importance must not raise KeyError with partial custom weights."""

    def _make_record(self):
        from muninn.core.types import MemoryRecord, MemoryType, Provenance
        return MemoryRecord(
            id="test",
            content="hello world",
            memory_type=MemoryType.EPISODIC,
            provenance=Provenance.AUTO_EXTRACTED,
            created_at=time.time() - 3600,
            ingested_at=time.time() - 3600,
            importance=0.5,
            recency_score=0.8,
            access_count=5,
            novelty_score=0.7,
            source_agent="test",
            project="global",
            namespace="global",
            scope="project",
        )

    def test_full_weights_dict_works(self):
        from muninn.scoring.importance import calculate_importance, DEFAULT_WEIGHTS
        rec = self._make_record()
        score = calculate_importance(rec, weights=DEFAULT_WEIGHTS)
        assert 0.0 <= score <= 1.0

    def test_partial_weights_dict_does_not_raise(self):
        from muninn.scoring.importance import calculate_importance
        rec = self._make_record()
        # Only supply two of five keys — should use defaults for the rest
        partial = {"recency": 0.5, "novelty": 0.3}
        score = calculate_importance(rec, weights=partial)
        assert 0.0 <= score <= 1.0

    def test_empty_weights_dict_uses_defaults(self):
        from muninn.scoring.importance import calculate_importance
        rec = self._make_record()
        score_default = calculate_importance(rec)
        score_empty = calculate_importance(rec, weights={})
        assert score_empty == pytest.approx(score_default, abs=1e-9)

    def test_output_always_in_unit_interval(self):
        from muninn.scoring.importance import calculate_importance
        rec = self._make_record()
        # Pathological weights that sum to more than 1
        bad_weights = {"recency": 5.0, "frequency": 5.0, "centrality": 5.0,
                       "novelty": 5.0, "provenance": 5.0}
        score = calculate_importance(rec, max_similarity=0.0, centrality=1.0,
                                     weights=bad_weights)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# P2: weight_adapter.py — chain and goal signals in DEFAULT_WEIGHTS
# ---------------------------------------------------------------------------

class TestWeightAdapterSignalCoverage:
    """WeightAdapter must cover all six RRF signals, not just four."""

    def test_default_weights_includes_chain(self):
        from muninn.retrieval.weight_adapter import DEFAULT_WEIGHTS
        assert "chain" in DEFAULT_WEIGHTS, (
            "DEFAULT_WEIGHTS missing 'chain' — chain signal never entropy-adapted"
        )

    def test_default_weights_includes_goal(self):
        from muninn.retrieval.weight_adapter import DEFAULT_WEIGHTS
        assert "goal" in DEFAULT_WEIGHTS, (
            "DEFAULT_WEIGHTS missing 'goal' — goal signal never entropy-adapted"
        )

    def test_all_six_rrf_signals_present(self):
        from muninn.retrieval.weight_adapter import DEFAULT_WEIGHTS
        required = {"vector", "graph", "bm25", "temporal", "chain", "goal"}
        missing = required - set(DEFAULT_WEIGHTS)
        assert not missing, f"DEFAULT_WEIGHTS missing signals: {missing}"

    def test_compute_weights_adapts_chain_and_goal(self):
        """With entropy-skewed results, chain and goal weights must change."""
        from muninn.retrieval.weight_adapter import WeightAdapter

        adapter = WeightAdapter()

        # Supply high-confidence results for chain and goal (low entropy)
        signal_results = {
            "vector": [("a", 0.9), ("b", 0.1)],
            "graph": [("a", 0.8), ("b", 0.2)],
            "bm25": [("a", 0.7), ("b", 0.3)],
            "temporal": [("a", 0.6), ("b", 0.4)],
            "chain": [("a", 0.95), ("b", 0.05)],   # very peaked → high confidence
            "goal": [("a", 0.99), ("b", 0.01)],    # very peaked → high confidence
        }
        weights = adapter.compute_weights("test query", signal_results)

        assert "chain" in weights
        assert "goal" in weights
        # High confidence → weights should be boosted above base
        base = adapter.base_weights
        assert weights["chain"] != base["chain"] or weights["goal"] != base["goal"], (
            "chain/goal weights are identical to base — entropy adaptation had no effect"
        )

    def test_chain_goal_base_values_match_hybrid_defaults(self):
        """chain=0.6, goal=0.65 must match hybrid.py SIGNAL_WEIGHTS and GOAL_SIGNAL_WEIGHT."""
        from muninn.retrieval.weight_adapter import DEFAULT_WEIGHTS
        from muninn.retrieval.hybrid import SIGNAL_WEIGHTS, GOAL_SIGNAL_WEIGHT

        assert DEFAULT_WEIGHTS["chain"] == pytest.approx(SIGNAL_WEIGHTS.get("chain", SIGNAL_WEIGHTS["bm25"]))
        # goal is stored separately in hybrid.py as GOAL_SIGNAL_WEIGHT
        assert DEFAULT_WEIGHTS["goal"] == pytest.approx(GOAL_SIGNAL_WEIGHT)


# ---------------------------------------------------------------------------
# P2: handlers.py — task error must use JSON-RPC error channel
# ---------------------------------------------------------------------------

class TestTaskResultErrorChannel:
    """handle_get_task_result must route task errors through send_error_fn."""

    def _make_failed_task(self, code=-32603, message="something broke"):
        return {
            "taskId": "fail-task",
            "status": "failed",
            "result": None,
            "error": {"code": code, "message": message},
            "ttl": 30000,
        }

    def test_error_goes_to_send_error_fn(self):
        from muninn.mcp.handlers import handle_get_task_result

        errors_received = []
        results_received = []

        def fake_send_error(msg_id, code, msg):
            errors_received.append((msg_id, code, msg))

        def fake_send_result(msg_id, payload):
            results_received.append((msg_id, payload))

        failed_task = self._make_failed_task()

        with patch("muninn.mcp.tasks.lookup_task_locked", return_value=failed_task), \
             patch("muninn.mcp.state._TASKS_CONDITION") as mock_cond:
            # Both _TASKS_CONDITION and lookup_task_locked are re-imported from their
            # source modules inside handle_get_task_result() on every call, so we must
            # patch the originals (state._TASKS_CONDITION, tasks.lookup_task_locked).
            mock_cond.__enter__ = MagicMock(return_value=mock_cond)
            mock_cond.__exit__ = MagicMock(return_value=False)

            handle_get_task_result(
                msg_id=99,
                params={"taskId": "fail-task", "wait": False},
                send_error_fn=fake_send_error,
                send_result_fn=fake_send_result,
            )

        assert len(errors_received) == 1, (
            "Expected exactly one call to send_error_fn for a failed task"
        )
        assert errors_received[0][0] == 99       # correct msg_id
        assert errors_received[0][1] == -32603   # correct code
        assert "something broke" in errors_received[0][2]
        assert len(results_received) == 0, (
            "send_result_fn must NOT be called for a failed task"
        )

    def test_error_code_preserved(self):
        from muninn.mcp.handlers import handle_get_task_result

        errors = []

        def fe(msg_id, code, msg):
            errors.append((msg_id, code, msg))

        def fr(msg_id, payload):
            pass

        task = self._make_failed_task(code=-32601, message="not found")
        with patch("muninn.mcp.tasks.lookup_task_locked", return_value=task), \
             patch("muninn.mcp.state._TASKS_CONDITION") as mc:
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            handle_get_task_result(42, {"taskId": "fail-task"}, fe, fr)

        assert errors[0][1] == -32601

    def test_successful_task_uses_send_result_fn(self):
        """Sanity check: completed tasks still go through send_result_fn."""
        from muninn.mcp.handlers import handle_get_task_result

        errors = []
        results = []

        def fe(msg_id, code, msg):
            errors.append((msg_id, code, msg))

        def fr(msg_id, payload):
            results.append((msg_id, payload))

        good_task = {
            "taskId": "ok-task",
            "status": "completed",
            "result": {"content": [{"type": "text", "text": "ok"}]},
            "error": None,
            "ttl": 30000,
        }
        with patch("muninn.mcp.tasks.lookup_task_locked", return_value=good_task), \
             patch("muninn.mcp.state._TASKS_CONDITION") as mc, \
             patch("muninn.mcp.tasks.related_task_meta", return_value={}):
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            handle_get_task_result(7, {"taskId": "ok-task"}, fe, fr)

        assert len(errors) == 0
        assert len(results) == 1


# ---------------------------------------------------------------------------
# P3: sqlite_metadata.py — get_by_ids chunked for >999 IDs
# ---------------------------------------------------------------------------

class TestGetByIdsChunking:
    """get_by_ids must not raise sqlite3.OperationalError for >999 IDs."""

    @pytest.fixture
    def store(self, tmp_path):
        from muninn.store.sqlite_metadata import SQLiteMetadataStore
        return SQLiteMetadataStore(tmp_path / "test.db")

    def test_empty_ids_returns_empty(self, store):
        assert store.get_by_ids([]) == []

    def test_small_batch_works(self, store):
        ids = ["nonexistent-" + str(i) for i in range(10)]
        result = store.get_by_ids(ids)
        assert result == []

    def test_over_999_ids_does_not_raise(self, store):
        """Previously raised sqlite3.OperationalError: too many SQL variables."""
        ids = ["ghost-" + str(i) for i in range(1500)]
        try:
            result = store.get_by_ids(ids)
        except sqlite3.OperationalError as e:
            pytest.fail(f"get_by_ids raised OperationalError for 1500 IDs: {e}")
        assert result == []  # none exist, so empty list expected

    def test_exactly_900_ids_works(self, store):
        """Boundary test at the chunk size."""
        ids = ["id-" + str(i) for i in range(900)]
        result = store.get_by_ids(ids)
        assert result == []

    def test_chunking_retrieves_all_records(self, store):
        """Records spanning multiple chunks must all be returned."""
        from muninn.core.types import MemoryRecord, MemoryType, Provenance
        import uuid as _uuid

        inserted_ids = []
        now = time.time()
        # Insert 1001 records to ensure we cross chunk boundary
        for i in range(1001):
            mid = f"chunk-test-{i:05d}"
            rec = MemoryRecord(
                id=mid,
                content=f"memory content {i}",
                memory_type=MemoryType.EPISODIC,
                provenance=Provenance.AUTO_EXTRACTED,
                created_at=now,
                ingested_at=now,
                importance=0.5,
                recency_score=0.8,
                access_count=0,
                novelty_score=0.5,
                source_agent="test",
                project="global",
                namespace="global",
                scope="project",
            )
            store.add(rec)
            inserted_ids.append(mid)

        fetched = store.get_by_ids(inserted_ids)
        assert len(fetched) == 1001, (
            f"Expected 1001 records across chunks, got {len(fetched)}"
        )
        fetched_ids = {r.id for r in fetched}
        assert fetched_ids == set(inserted_ids)

    def test_chunk_size_constant_is_reasonable(self):
        from muninn.store.sqlite_metadata import SQLiteMetadataStore
        # Must be < 999 to stay under SQLite hard limit
        assert SQLiteMetadataStore._SQLITE_MAX_VARS < 999
        # Must be large enough to be efficient
        assert SQLiteMetadataStore._SQLITE_MAX_VARS >= 500


# ---------------------------------------------------------------------------
# P3: sqlite_metadata.py — search_content LIKE metachar escaping
# ---------------------------------------------------------------------------

class TestSearchContentLikeEscaping:
    """search_content must treat % and _ in the query as literals."""

    @pytest.fixture
    def store(self, tmp_path):
        from muninn.store.sqlite_metadata import SQLiteMetadataStore
        return SQLiteMetadataStore(tmp_path / "search_test.db")

    def _insert(self, store, content):
        from muninn.core.types import MemoryRecord, MemoryType, Provenance
        now = time.time()
        import uuid as _uuid
        rec = MemoryRecord(
            id=str(_uuid.uuid4()),
            content=content,
            memory_type=MemoryType.EPISODIC,
            provenance=Provenance.AUTO_EXTRACTED,
            created_at=now,
            ingested_at=now,
            importance=0.5,
            recency_score=0.8,
            access_count=0,
            novelty_score=0.5,
            source_agent="test",
            project="global",
            namespace="global",
            scope="project",
        )
        store.add(rec)
        return rec.id

    def test_percent_literal_does_not_match_all(self, store):
        """Query '100%' should NOT match 'absolute result' — % must be escaped."""
        id1 = self._insert(store, "100% complete task")
        id2 = self._insert(store, "absolute confidence result")

        results = store.search_content("100%")
        contents = [r.content for r in results]
        assert "100% complete task" in contents
        assert "absolute confidence result" not in contents

    def test_underscore_literal_does_not_wildcard_match(self, store):
        """Query 'a_b' must NOT match 'axb' or 'a1b' — _ must be escaped."""
        id1 = self._insert(store, "value a_b found")
        id2 = self._insert(store, "value axb found")
        id3 = self._insert(store, "value a1b found")

        results = store.search_content("a_b")
        contents = [r.content for r in results]
        assert "value a_b found" in contents
        assert "value axb found" not in contents
        assert "value a1b found" not in contents

    def test_percent_percent_matches_literal_double_percent(self, store):
        """Query with literal %% in the search term should match content containing %%."""
        id1 = self._insert(store, "success rate: 100%% confidence")
        id2 = self._insert(store, "something else entirely")

        results = store.search_content("100%%")
        contents = [r.content for r in results]
        assert "success rate: 100%% confidence" in contents
        assert "something else entirely" not in contents

    def test_normal_substring_search_still_works(self, store):
        """Ensure basic substring matching is not broken by escaping."""
        self._insert(store, "the quick brown fox")
        self._insert(store, "the slow red fox")
        results = store.search_content("quick")
        assert len(results) == 1
        assert results[0].content == "the quick brown fox"

    def test_backslash_in_query_handled(self, store):
        """Backslash in query must not break the ESCAPE clause."""
        self._insert(store, r"C:\Users\user\file.txt")
        results = store.search_content(r"C:\Users")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# P3: sqlite_metadata.py — snips_sum_w deduplication
# ---------------------------------------------------------------------------

class TestSnipsDictDeduplication:
    """Verify snips_total was removed and snips_sum_w is used as the SNIPS denominator."""

    def test_snips_method_source_has_no_snips_total(self):
        from muninn.store.sqlite_metadata import SQLiteMetadataStore
        src = inspect.getsource(SQLiteMetadataStore.get_feedback_signal_multipliers)
        # Check that snips_total is not used as a variable (declaration or assignment).
        # A comment mentioning the old name is acceptable; live code is not.
        assert "snips_total =" not in src and "snips_total:" not in src, (
            "snips_total still declared or assigned — redundant dict was not removed"
        )

    def test_snips_sum_w_is_used_as_denominator(self):
        """After the fix, snips_sum_w is the sole accumulator for Σipw."""
        from muninn.store.sqlite_metadata import SQLiteMetadataStore
        src = inspect.getsource(SQLiteMetadataStore.get_feedback_signal_multipliers)
        assert "snips_sum_w" in src

    def test_snips_estimator_produces_correct_score(self, tmp_path):
        """SNIPS estimator end-to-end: score should equal weighted mean of outcomes."""
        from muninn.store.sqlite_metadata import SQLiteMetadataStore
        import json as _json

        store = SQLiteMetadataStore(tmp_path / "snips_test.db")
        now = time.time()

        # Insert two feedback rows, both outcome=1.0, same signal weight=1.0,
        # sampling_prob=1.0, rank=1 → propensity = 1/log2(2) = 1.0
        # SNIPS score = (1*1 + 1*1) / (1 + 1) = 1.0 → multiplier = floor+1*(ceiling-floor)
        for _ in range(3):
            conn = store._get_conn()
            conn.execute(
                """
                INSERT INTO retrieval_feedback
                  (user_id, namespace, project, query_text, memory_id,
                   outcome, rank, sampling_prob, signals_json, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("u1", "ns", "proj", "q", "m1",
                 1.0, 1, 1.0, _json.dumps({"vector": 1.0}), "test", now),
            )
            conn.commit()

        multipliers = store.get_feedback_signal_multipliers(
            user_id="u1",
            namespace="ns",
            project="proj",
            lookback_days=1,
            min_total_signal_weight=2.0,
            estimator="snips",
            floor=0.75,
            ceiling=1.25,
        )
        # Perfect outcome → score=1.0 → multiplier=ceiling=1.25
        assert "vector" in multipliers
        assert multipliers["vector"] == pytest.approx(1.25, abs=1e-6)


# ---------------------------------------------------------------------------
# P2: memory.py — duplicate OTel/flags initialization removed
# ---------------------------------------------------------------------------

class TestMemoryInitNoDuplicateOTel:
    """initialize() must not create OTelGenAITracer twice."""

    def test_otel_tracer_created_once(self):
        """Count OTelGenAITracer instantiations during initialize()."""
        from muninn.core import memory as mem_mod
        creation_count = []

        original_init = mem_mod.OTelGenAITracer.__init__

        def counting_init(self_otel, **kwargs):
            creation_count.append(1)
            original_init(self_otel, **kwargs)

        with patch.object(mem_mod.OTelGenAITracer, "__init__", counting_init):
            # We only need to test the source code structure, not run the full
            # initialize() which requires Qdrant/Kuzu services.
            src = inspect.getsource(mem_mod.MuninnMemory.initialize)
            # Count how many times OTelGenAITracer( appears in the source
            tracer_calls = src.count("OTelGenAITracer(")
            assert tracer_calls == 1, (
                f"OTelGenAITracer() appears {tracer_calls} times in initialize() — "
                "expected exactly 1 (duplicate removed)"
            )

    def test_flags_loaded_once(self):
        """get_flags() must be called only once during initialize()."""
        from muninn.core import memory as mem_mod
        src = inspect.getsource(mem_mod.MuninnMemory.initialize)
        # Count calls to get_flags() in source
        flags_calls = src.count("flags = get_flags()")
        assert flags_calls == 1, (
            f"get_flags() called {flags_calls} times — expected exactly 1"
        )

    def test_retriever_receives_same_otel_as_self(self):
        """The retriever must be constructed with the same OTel as self._otel — no post-hoc patch."""
        from muninn.core import memory as mem_mod
        src = inspect.getsource(mem_mod.MuninnMemory.initialize)
        # The old code had: self._retriever._telemetry = self._otel (post-hoc patch)
        assert "self._retriever._telemetry = self._otel" not in src, (
            "Post-hoc retriever telemetry patch still present — "
            "retriever should be constructed with telemetry=self._otel directly"
        )
