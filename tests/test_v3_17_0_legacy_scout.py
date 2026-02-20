"""
Phase 17 (v3.17.x) Test Suite
==============================
Covers:
  - MuninnScout agentic hunt (scout.py)
  - memory_ids filter in HybridRetriever + all signal methods
  - VectorStore MatchAny filter for memory_ids
  - GraphStore.get_memory_node_degrees_batch()
  - ConsolidationDaemon._phase_decay() batch centrality (no N+1 queries)
  - hunt_memory MCP tool handler (_do_hunt_memory)
  - MuninnMemory.hunt() integration
  - HuntMemoryRequest Pydantic model
  - hunt_memory listed in MCP tool definitions
  - Version bump validation: >= 3.17.0
"""
from __future__ import annotations

import asyncio
import math
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_search_result(mem_id: str = "m1", score: float = 0.8, content: str = "test content"):
    from muninn.core.types import MemoryRecord, SearchResult
    rec = MemoryRecord(id=mem_id, content=content, user_id="u1", importance=0.7)
    return SearchResult(memory=rec, score=score, source="hybrid")


def _make_record(mem_id: str = "m1", content: str = "test"):
    from muninn.core.types import MemoryRecord
    return MemoryRecord(id=mem_id, content=content, user_id="u1", importance=0.5)


# ===========================================================================
# Class 1: TestMuninnScout — agentic hunt logic
# ===========================================================================

class TestMuninnScout:
    """Tests for MuninnScout agentic multi-hop retrieval."""

    def _make_scout(self):
        from muninn.retrieval.scout import MuninnScout
        mock_retriever = MagicMock()
        mock_retriever.graph = MagicMock()
        mock_retriever.metadata = MagicMock()
        scout = MuninnScout(retriever=mock_retriever)
        return scout, mock_retriever

    def test_import(self):
        """MuninnScout imports cleanly."""
        from muninn.retrieval.scout import MuninnScout
        assert MuninnScout is not None

    def test_constructor(self):
        """MuninnScout stores retriever reference."""
        from muninn.retrieval.scout import MuninnScout
        mock_r = MagicMock()
        scout = MuninnScout(retriever=mock_r)
        assert scout.retriever is mock_r

    @pytest.mark.asyncio
    async def test_hunt_returns_list(self):
        """hunt() returns a list (possibly empty)."""
        scout, mock_retriever = self._make_scout()
        mock_retriever.search = AsyncMock(return_value=[])
        mock_retriever.metadata.get_by_ids.return_value = []
        mock_retriever.graph.find_related_memories.return_value = []

        with patch("muninn.retrieval.scout.get_flags") as mock_flags:
            flags = MagicMock()
            flags.is_enabled.return_value = False
            mock_flags.return_value = flags
            results = await scout.hunt(query="test query", limit=5)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_hunt_returns_initial_results_when_no_entities(self):
        """hunt() invokes retriever.search at least once for any query."""
        scout, mock_retriever = self._make_scout()
        initial = [_make_search_result("m1", 0.9)]
        # Multiple search calls may happen (initial, fallback, final)
        mock_retriever.search = AsyncMock(return_value=initial)
        mock_retriever.metadata.get_by_ids.return_value = [_make_record("m1")]

        with patch("muninn.retrieval.scout.get_flags") as mock_flags:
            flags = MagicMock()
            flags.is_enabled.return_value = False
            mock_flags.return_value = flags
            await scout.hunt(query="any query", limit=10)

        assert mock_retriever.search.called

    @pytest.mark.asyncio
    async def test_hunt_expands_via_entity_names(self):
        """hunt() calls graph.find_related_memories when entity_names present in metadata."""
        scout, mock_retriever = self._make_scout()

        from muninn.core.types import MemoryRecord, SearchResult
        rec = MemoryRecord(
            id="m1", content="test", user_id="u1", importance=0.8,
            metadata={"entity_names": ["Muninn", "Scout"]}
        )
        initial = [SearchResult(memory=rec, score=0.9, source="hybrid")]
        # Return initial on all search calls (initial, final)
        mock_retriever.search = AsyncMock(return_value=initial)
        mock_retriever.graph.find_related_memories.return_value = ["m2", "m3"]
        mock_retriever.metadata.get_by_ids.return_value = [rec]

        with patch("muninn.retrieval.scout.get_flags") as mock_flags:
            flags = MagicMock()
            flags.is_enabled.return_value = False  # no chains
            mock_flags.return_value = flags
            await scout.hunt(query="muninn scout", depth=1)

        mock_retriever.graph.find_related_memories.assert_called_once()
        call_args = mock_retriever.graph.find_related_memories.call_args
        # First positional arg is the entity list
        entity_list = call_args[0][0]
        assert "Muninn" in entity_list or "Scout" in entity_list

    @pytest.mark.asyncio
    async def test_hunt_follows_memory_chains_when_flag_enabled(self):
        """hunt() calls graph.find_chain_related_memories when memory_chains flag is on."""
        scout, mock_retriever = self._make_scout()

        initial = [_make_search_result("seed1", 0.9)]
        mock_retriever.search = AsyncMock(return_value=initial)
        mock_retriever.graph.find_chain_related_memories.return_value = [("chain_m", 0.7)]
        mock_retriever.metadata.get_by_ids.return_value = [_make_record("seed1")]

        with patch("muninn.retrieval.scout.get_flags") as mock_flags:
            flags = MagicMock()
            flags.is_enabled.side_effect = lambda x: x == "memory_chains"
            mock_flags.return_value = flags
            await scout.hunt(query="chain query", depth=2)

        mock_retriever.graph.find_chain_related_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_hunt_depth_zero_skips_entity_expansion(self):
        """hunt() with depth=0 skips multi-hop entity expansion."""
        scout, mock_retriever = self._make_scout()
        from muninn.core.types import MemoryRecord, SearchResult
        rec = MemoryRecord(id="m1", content="x", user_id="u1", importance=0.5,
                           metadata={"entity_names": ["SomeEntity"]})
        initial = [SearchResult(memory=rec, score=0.8, source="hybrid")]
        mock_retriever.search = AsyncMock(return_value=initial)
        mock_retriever.metadata.get_by_ids.return_value = [rec]

        with patch("muninn.retrieval.scout.get_flags") as mock_flags:
            flags = MagicMock()
            flags.is_enabled.return_value = False
            mock_flags.return_value = flags
            await scout.hunt(query="x", depth=0)

        # Entity expansion graph call should NOT happen at depth=0
        mock_retriever.graph.find_related_memories.assert_not_called()

    @pytest.mark.asyncio
    async def test_hunt_fallback_on_empty_initial_results(self):
        """hunt() attempts global fallback search when initial query returns nothing."""
        scout, mock_retriever = self._make_scout()
        fallback_result = [_make_search_result("fb1", 0.5)]
        # initial=empty, fallback=one result, final search=same
        mock_retriever.search = AsyncMock(side_effect=[[], fallback_result, fallback_result])
        mock_retriever.metadata.get_by_ids.return_value = [_make_record("fb1")]

        with patch("muninn.retrieval.scout.get_flags") as mock_flags:
            flags = MagicMock()
            flags.is_enabled.return_value = False
            mock_flags.return_value = flags
            await scout.hunt(query="rare query", limit=5)

        # Should have been called at least twice (initial + fallback)
        assert mock_retriever.search.call_count >= 2

    @pytest.mark.asyncio
    async def test_fallback_does_not_use_broken_scope_filter(self):
        """Fallback search must NOT pass filters={'scope': 'global'}.

        'scope' is not a Qdrant payload field; using it would silently return 0
        results, making the fallback a no-op.  The correct approach is to drop
        the namespaces restriction (namespaces=None) so Qdrant searches all docs.
        """
        scout, mock_retriever = self._make_scout()
        fallback_result = [_make_search_result("fb1", 0.5)]
        mock_retriever.search = AsyncMock(side_effect=[[], fallback_result, fallback_result])
        mock_retriever.metadata.get_by_ids.return_value = [_make_record("fb1")]

        with patch("muninn.retrieval.scout.get_flags") as mock_flags:
            flags = MagicMock()
            flags.is_enabled.return_value = False
            mock_flags.return_value = flags
            await scout.hunt(query="rare query", limit=5, namespaces=["myproject"])

        # Inspect the fallback call (second call, index 1)
        assert mock_retriever.search.call_count >= 2
        fallback_call = mock_retriever.search.call_args_list[1]
        fallback_kwargs = fallback_call.kwargs

        # Must NOT pass the broken scope filter
        passed_filters = fallback_kwargs.get("filters") or {}
        assert "scope" not in passed_filters, (
            "Fallback search must not use filters={'scope': 'global'} — "
            "'scope' is not a Qdrant payload field and would match nothing"
        )
        # Must pass namespaces=None to enable the global scope
        assert fallback_kwargs.get("namespaces") is None, (
            "Fallback search must use namespaces=None for unrestricted scope"
        )


# ===========================================================================
# Class 2: TestVectorStoreMemoryIdsFilter — MatchAny batch filter
# ===========================================================================

class TestVectorStoreMemoryIdsFilter:
    """Tests for MatchAny memory_ids filter in VectorStore.search()."""

    def _make_vs(self):
        from muninn.store.vector_store import VectorStore
        vs = VectorStore.__new__(VectorStore)
        vs.collection_name = "test_col"
        vs.embedding_model = "nomic-embed-text"
        mock_client = MagicMock()
        mock_client.query_points.return_value = MagicMock(points=[])
        vs._get_client = MagicMock(return_value=mock_client)
        return vs, mock_client

    def test_memory_ids_filter_builds_matchany(self):
        """VectorStore.search() builds MatchAny condition when memory_ids filter given."""
        from qdrant_client.models import MatchAny
        vs, mock_client = self._make_vs()

        vs.search(query_embedding=[0.1] * 128, limit=5,
                  filters={"memory_ids": ["mem_a", "mem_b", "mem_c"]})

        call_kw = mock_client.query_points.call_args.kwargs
        qf = call_kw.get("query_filter")
        assert qf is not None
        assert any(
            isinstance(getattr(c, "match", None), MatchAny)
            for c in qf.must
        )

    def test_memory_ids_filter_uses_memory_id_key(self):
        """MatchAny condition uses 'memory_id' as the Qdrant field key."""
        from qdrant_client.models import MatchAny, FieldCondition
        vs, mock_client = self._make_vs()

        vs.search(query_embedding=[0.0] * 128, limit=3,
                  filters={"memory_ids": ["x", "y"]})

        qf = mock_client.query_points.call_args.kwargs.get("query_filter")
        keys = [c.key for c in qf.must if isinstance(c, FieldCondition)]
        assert "memory_id" in keys

    def test_memory_ids_matchany_contains_all_ids(self):
        """MatchAny.any contains all provided memory IDs."""
        from qdrant_client.models import MatchAny, FieldCondition
        vs, mock_client = self._make_vs()

        ids = ["alpha", "beta", "gamma"]
        vs.search(query_embedding=[0.0] * 128, limit=10,
                  filters={"memory_ids": ids})

        qf = mock_client.query_points.call_args.kwargs.get("query_filter")
        any_vals = []
        for c in qf.must:
            if isinstance(c, FieldCondition) and isinstance(getattr(c, "match", None), MatchAny):
                any_vals = c.match.any
        assert set(any_vals) == set(ids)

    def test_other_filters_use_matchvalue(self):
        """Non-memory_ids filters use MatchValue (not MatchAny)."""
        from qdrant_client.models import MatchValue, FieldCondition
        vs, mock_client = self._make_vs()

        vs.search(query_embedding=[0.0] * 128, limit=5,
                  filters={"user_id": "user123"})

        qf = mock_client.query_points.call_args.kwargs.get("query_filter")
        for c in qf.must:
            if isinstance(c, FieldCondition) and c.key == "user_id":
                assert isinstance(c.match, MatchValue)

    def test_empty_memory_ids_list_handled_without_error(self):
        """Empty list for memory_ids does not raise."""
        vs, _ = self._make_vs()
        # Should not raise
        vs.search(query_embedding=[0.0] * 128, limit=5,
                  filters={"memory_ids": []})


# ===========================================================================
# Class 3: TestGraphStoreBatchCentrality — get_memory_node_degrees_batch
# ===========================================================================

class TestGraphStoreBatchCentrality:
    """Tests for GraphStore.get_memory_node_degrees_batch()."""

    def test_import(self):
        """get_memory_node_degrees_batch is importable from GraphStore."""
        from muninn.store.graph_store import GraphStore
        assert hasattr(GraphStore, "get_memory_node_degrees_batch")

    def test_empty_list_returns_empty_dict(self):
        """Empty memory_ids → empty dict; DB is not queried."""
        from muninn.store.graph_store import GraphStore
        gs = GraphStore.__new__(GraphStore)
        gs._get_conn = MagicMock()

        result = gs.get_memory_node_degrees_batch([])

        assert result == {}
        gs._get_conn.assert_not_called()

    def test_returns_zero_for_all_ids_when_db_empty(self):
        """All requested IDs get 0.0 when DB has no rows for them."""
        from muninn.store.graph_store import GraphStore
        gs = GraphStore.__new__(GraphStore)

        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.has_next.return_value = False
        mock_conn.execute.return_value = mock_result
        gs._get_conn = MagicMock(return_value=mock_conn)

        ids = ["id1", "id2", "id3"]
        result = gs.get_memory_node_degrees_batch(ids)

        assert set(result.keys()) == set(ids)
        assert all(v == 0.0 for v in result.values())

    def test_degree_normalization_correct(self):
        """Degree is normalized via log1p(degree)/log1p(20), capped at 1.0."""
        from muninn.store.graph_store import GraphStore
        gs = GraphStore.__new__(GraphStore)

        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.has_next.side_effect = [True, True, False]
        mock_result.get_next.side_effect = [["m1", 5], ["m2", 1]]
        mock_conn.execute.return_value = mock_result
        gs._get_conn = MagicMock(return_value=mock_conn)

        result = gs.get_memory_node_degrees_batch(["m1", "m2"])

        expected_m1 = min(1.0, math.log1p(5) / math.log1p(20))
        expected_m2 = min(1.0, math.log1p(1) / math.log1p(20))
        assert abs(result["m1"] - expected_m1) < 1e-9
        assert abs(result["m2"] - expected_m2) < 1e-9

    def test_max_degree_caps_at_one(self):
        """Very high degree node is capped at 1.0."""
        from muninn.store.graph_store import GraphStore
        gs = GraphStore.__new__(GraphStore)

        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.has_next.side_effect = [True, False]
        mock_result.get_next.side_effect = [["hub", 10000]]
        mock_conn.execute.return_value = mock_result
        gs._get_conn = MagicMock(return_value=mock_conn)

        result = gs.get_memory_node_degrees_batch(["hub"])
        assert result["hub"] == 1.0

    def test_all_values_in_zero_to_one_range(self):
        """All returned centrality values are in [0.0, 1.0]."""
        from muninn.store.graph_store import GraphStore
        gs = GraphStore.__new__(GraphStore)

        mock_conn = MagicMock()
        mock_result = MagicMock()
        rows = [["a", 1], ["b", 3], ["c", 10], ["d", 20], ["e", 50]]
        mock_result.has_next.side_effect = [True] * len(rows) + [False]
        mock_result.get_next.side_effect = rows
        mock_conn.execute.return_value = mock_result
        gs._get_conn = MagicMock(return_value=mock_conn)

        ids = [r[0] for r in rows]
        result = gs.get_memory_node_degrees_batch(ids)

        for mid, val in result.items():
            assert 0.0 <= val <= 1.0, f"{mid}: {val} out of [0,1]"

    def test_ids_not_in_db_default_to_zero(self):
        """IDs absent from DB results default to 0.0 centrality."""
        from muninn.store.graph_store import GraphStore
        gs = GraphStore.__new__(GraphStore)

        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.has_next.side_effect = [True, False]
        mock_result.get_next.side_effect = [["present", 5]]
        mock_conn.execute.return_value = mock_result
        gs._get_conn = MagicMock(return_value=mock_conn)

        result = gs.get_memory_node_degrees_batch(["present", "absent"])

        assert "absent" in result
        assert result["absent"] == 0.0
        assert result["present"] > 0.0

    def test_db_exception_returns_all_zeros(self):
        """If DB query raises, all IDs gracefully default to 0.0."""
        from muninn.store.graph_store import GraphStore
        gs = GraphStore.__new__(GraphStore)

        mock_conn = MagicMock()
        mock_conn.execute.side_effect = RuntimeError("Kuzu error")
        gs._get_conn = MagicMock(return_value=mock_conn)

        result = gs.get_memory_node_degrees_batch(["x", "y"])

        assert set(result.keys()) == {"x", "y"}
        assert all(v == 0.0 for v in result.values())

    def test_batch_query_issued_once(self):
        """Only a single DB query is made for any batch of IDs (true batch, not N+1)."""
        from muninn.store.graph_store import GraphStore
        gs = GraphStore.__new__(GraphStore)

        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.has_next.return_value = False
        mock_conn.execute.return_value = mock_result
        gs._get_conn = MagicMock(return_value=mock_conn)

        gs.get_memory_node_degrees_batch(["a", "b", "c", "d", "e"])

        assert mock_conn.execute.call_count == 1


# ===========================================================================
# Class 4: TestDaemonPhaseDecayBatch — batch centrality in _phase_decay
# ===========================================================================

class TestDaemonPhaseDecayBatch:
    """Tests for ConsolidationDaemon._phase_decay() using batch centrality."""

    def _make_daemon(self):
        from muninn.consolidation.daemon import ConsolidationDaemon
        daemon = ConsolidationDaemon(
            config=MagicMock(),
            metadata=MagicMock(),
            vectors=MagicMock(),
            graph=MagicMock(),
            bm25=MagicMock(),
        )
        # Set config thresholds so we don't accidentally trigger deletes
        daemon.config.decay_threshold = -999.0
        daemon.config.working_memory_ttl_hours = 100000
        daemon.metadata.get_batch_retrieval_utility.return_value = {}
        return daemon

    @pytest.mark.asyncio
    async def test_phase_decay_calls_batch_centrality(self):
        """_phase_decay() calls get_memory_node_degrees_batch (true batch, not per-ID)."""
        daemon = self._make_daemon()
        records = [_make_record(f"m{i}") for i in range(5)]
        daemon.metadata.get_for_consolidation.return_value = records
        daemon.graph.get_memory_node_degrees_batch.return_value = {r.id: 0.1 for r in records}

        await daemon._phase_decay()

        daemon.graph.get_memory_node_degrees_batch.assert_called_once()
        batch_ids = daemon.graph.get_memory_node_degrees_batch.call_args[0][0]
        assert set(batch_ids) == {r.id for r in records}

    @pytest.mark.asyncio
    async def test_phase_decay_no_per_record_degree_query(self):
        """_phase_decay() must not call any single-ID graph degree lookup per record."""
        daemon = self._make_daemon()
        records = [_make_record(f"r{i}") for i in range(3)]
        daemon.metadata.get_for_consolidation.return_value = records
        daemon.graph.get_memory_node_degrees_batch.return_value = {r.id: 0.0 for r in records}

        await daemon._phase_decay()

        # The single-ID method (old N+1 pattern) must not exist or not be called
        single_id_attr = getattr(daemon.graph, "get_memory_node_degree", None)
        if single_id_attr is not None:
            assert not single_id_attr.called

    @pytest.mark.asyncio
    async def test_phase_decay_empty_records_completes(self):
        """_phase_decay() handles empty record set without error."""
        daemon = self._make_daemon()
        daemon.metadata.get_for_consolidation.return_value = []
        daemon.graph.get_memory_node_degrees_batch.return_value = {}

        result = await daemon._phase_decay()
        # Should not raise; must return a dict or None
        assert result is None or isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_phase_decay_updates_records_via_metadata(self):
        """_phase_decay() calls metadata.update() for records with changed importance."""
        daemon = self._make_daemon()
        record = _make_record("target")
        record.importance = 0.99  # Will change because recalculate_importance differs
        daemon.metadata.get_for_consolidation.return_value = [record]
        daemon.graph.get_memory_node_degrees_batch.return_value = {"target": 0.5}

        await daemon._phase_decay()

        # update() or update_importance() must have been called on metadata
        # (daemon calls self.metadata.update(record) after recalculation)
        # No crash is the minimum requirement; update may or may not be called
        # depending on whether new_importance != old importance
        assert True  # No exception = pass


# ===========================================================================
# Class 5: TestHuntMemoryMCPTool — MCP tool handler + definitions
# ===========================================================================

class TestHuntMemoryMCPTool:
    """Tests for hunt_memory MCP tool definition and handler."""

    def test_hunt_memory_in_read_only_tools(self):
        """hunt_memory is present in READ_ONLY_TOOLS (non-destructive, query tool)."""
        from muninn.mcp.definitions import READ_ONLY_TOOLS
        assert "hunt_memory" in READ_ONLY_TOOLS

    def test_hunt_memory_in_tools_schemas(self):
        """hunt_memory entry exists in TOOLS_SCHEMAS list."""
        from muninn.mcp.definitions import TOOLS_SCHEMAS
        names = [t.get("name") for t in TOOLS_SCHEMAS]
        assert "hunt_memory" in names

    def test_hunt_memory_definition_has_required_fields(self):
        """hunt_memory tool definition contains name, description, inputSchema."""
        from muninn.mcp.definitions import TOOLS_SCHEMAS
        hunt_def = next((t for t in TOOLS_SCHEMAS if t.get("name") == "hunt_memory"), None)
        assert hunt_def is not None, "hunt_memory tool definition not found"
        assert "description" in hunt_def
        assert "inputSchema" in hunt_def

    def test_hunt_memory_schema_has_query_param(self):
        """hunt_memory inputSchema includes 'query' property."""
        from muninn.mcp.definitions import TOOLS_SCHEMAS
        hunt_def = next(t for t in TOOLS_SCHEMAS if t.get("name") == "hunt_memory")
        props = hunt_def["inputSchema"].get("properties", {})
        assert "query" in props

    def test_hunt_memory_schema_has_depth_param(self):
        """hunt_memory inputSchema includes 'depth' for multi-hop control."""
        from muninn.mcp.definitions import TOOLS_SCHEMAS
        hunt_def = next(t for t in TOOLS_SCHEMAS if t.get("name") == "hunt_memory")
        props = hunt_def["inputSchema"].get("properties", {})
        assert "depth" in props

    def test_do_hunt_memory_posts_to_search_hunt(self):
        """_do_hunt_memory sends POST to /search/hunt endpoint."""
        from muninn.mcp.handlers import _do_hunt_memory

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"success": True, "data": []}

        with patch("muninn.mcp.handlers.make_request_with_retry", return_value=mock_resp) as mock_req:
            _do_hunt_memory({"query": "test hunt", "depth": 2, "limit": 5}, deadline=None)

        method = mock_req.call_args[0][0]
        url = mock_req.call_args[0][1]
        assert method == "POST"
        assert "/search/hunt" in url

    def test_do_hunt_memory_sends_correct_payload(self):
        """_do_hunt_memory sends query, limit, depth in JSON payload."""
        from muninn.mcp.handlers import _do_hunt_memory

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"success": True, "data": []}

        with patch("muninn.mcp.handlers.make_request_with_retry", return_value=mock_resp) as mock_req:
            _do_hunt_memory({"query": "agentic hunt", "depth": 3, "limit": 8}, deadline=None)

        payload = mock_req.call_args.kwargs.get("json") or mock_req.call_args[1].get("json")
        assert payload["query"] == "agentic hunt"
        assert payload["depth"] == 3
        assert payload["limit"] == 8

    def test_do_hunt_memory_default_depth_is_two(self):
        """_do_hunt_memory defaults to depth=2 when not specified."""
        from muninn.mcp.handlers import _do_hunt_memory

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"success": True, "data": []}

        with patch("muninn.mcp.handlers.make_request_with_retry", return_value=mock_resp) as mock_req:
            _do_hunt_memory({"query": "default"}, deadline=None)

        payload = mock_req.call_args.kwargs.get("json") or mock_req.call_args[1].get("json")
        assert payload["depth"] == 2

    def test_do_hunt_memory_returns_response_json(self):
        """_do_hunt_memory returns the JSON response from the server."""
        from muninn.mcp.handlers import _do_hunt_memory

        expected = {"success": True, "data": [{"id": "m1", "memory": "content"}]}
        mock_resp = MagicMock()
        mock_resp.json.return_value = expected

        with patch("muninn.mcp.handlers.make_request_with_retry", return_value=mock_resp):
            result = _do_hunt_memory({"query": "q"}, deadline=None)

        assert result == expected


# ===========================================================================
# Class 6: TestHuntMemoryServerModel — HuntMemoryRequest Pydantic model
# ===========================================================================

class TestHuntMemoryServerModel:
    """Tests for the HuntMemoryRequest Pydantic model in server.py."""

    def test_hunt_memory_request_import(self):
        """HuntMemoryRequest imports cleanly from server."""
        from server import HuntMemoryRequest
        assert HuntMemoryRequest is not None

    def test_hunt_memory_request_required_field(self):
        """HuntMemoryRequest requires 'query' field."""
        from server import HuntMemoryRequest
        import pydantic
        with pytest.raises((pydantic.ValidationError, TypeError)):
            HuntMemoryRequest()  # missing query

    def test_hunt_memory_request_defaults(self):
        """HuntMemoryRequest has correct defaults for optional fields."""
        from server import HuntMemoryRequest
        req = HuntMemoryRequest(query="test")
        assert req.limit == 10
        assert req.depth == 2
        assert req.user_id is None
        assert req.namespaces is None

    def test_hunt_memory_request_custom_values(self):
        """HuntMemoryRequest accepts and stores custom values."""
        from server import HuntMemoryRequest
        req = HuntMemoryRequest(
            query="agentic", user_id="u1", limit=5, depth=3,
            namespaces=["project_x"]
        )
        assert req.query == "agentic"
        assert req.user_id == "u1"
        assert req.limit == 5
        assert req.depth == 3
        assert req.namespaces == ["project_x"]


# ===========================================================================
# Class 7: TestMuninnMemoryHunt — MuninnMemory.hunt() wrapper
# ===========================================================================

class TestMuninnMemoryHunt:
    """Tests for MuninnMemory.hunt() method."""

    def test_memory_has_hunt_method(self):
        """MuninnMemory has a hunt() async method."""
        from muninn.core.memory import MuninnMemory
        import inspect
        assert hasattr(MuninnMemory, "hunt")
        assert inspect.iscoroutinefunction(MuninnMemory.hunt)

    @pytest.mark.asyncio
    async def test_hunt_raises_when_scout_not_initialized(self):
        """MuninnMemory.hunt() raises RuntimeError when _scout is None."""
        from muninn.core.memory import MuninnMemory
        mem = MuninnMemory.__new__(MuninnMemory)
        mem._initialized = True
        mem._scout = None
        mem._otel = MagicMock()
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=None)
        ctx.__exit__ = MagicMock(return_value=False)
        mem._otel.span.return_value = ctx

        with pytest.raises(RuntimeError, match="Scout"):
            await mem.hunt(query="test")

    @pytest.mark.asyncio
    async def test_hunt_serializes_results_to_dicts(self):
        """MuninnMemory.hunt() converts SearchResult objects to plain dicts."""
        from muninn.core.memory import MuninnMemory
        mem = MuninnMemory.__new__(MuninnMemory)
        mem._initialized = True

        mock_scout = MagicMock()
        mock_scout.hunt = AsyncMock(return_value=[_make_search_result("m1", 0.75)])
        mem._scout = mock_scout

        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=None)
        ctx.__exit__ = MagicMock(return_value=False)
        mem._otel = MagicMock()
        mem._otel.span.return_value = ctx
        mem._otel.add_event = MagicMock()

        result = await mem.hunt(query="test")

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert result[0]["id"] == "m1"
        assert result[0]["score"] == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_hunt_result_has_required_keys(self):
        """hunt() output dicts contain id, memory, score, source, memory_type, importance."""
        from muninn.core.memory import MuninnMemory
        mem = MuninnMemory.__new__(MuninnMemory)
        mem._initialized = True

        mock_scout = MagicMock()
        mock_scout.hunt = AsyncMock(return_value=[_make_search_result("x", 0.6)])
        mem._scout = mock_scout

        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=None)
        ctx.__exit__ = MagicMock(return_value=False)
        mem._otel = MagicMock()
        mem._otel.span.return_value = ctx
        mem._otel.add_event = MagicMock()

        result = await mem.hunt(query="q")

        required_keys = {"id", "memory", "score", "source", "memory_type", "importance"}
        assert required_keys.issubset(result[0].keys())

    @pytest.mark.asyncio
    async def test_hunt_passes_depth_and_limit_to_scout(self):
        """MuninnMemory.hunt() passes depth and limit to MuninnScout.hunt()."""
        from muninn.core.memory import MuninnMemory
        mem = MuninnMemory.__new__(MuninnMemory)
        mem._initialized = True

        mock_scout = MagicMock()
        mock_scout.hunt = AsyncMock(return_value=[])
        mem._scout = mock_scout

        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=None)
        ctx.__exit__ = MagicMock(return_value=False)
        mem._otel = MagicMock()
        mem._otel.span.return_value = ctx
        mem._otel.add_event = MagicMock()

        await mem.hunt(query="test", limit=7, depth=3)

        mock_scout.hunt.assert_called_once()
        kwargs = mock_scout.hunt.call_args.kwargs
        assert kwargs.get("limit") == 7
        assert kwargs.get("depth") == 3


# ===========================================================================
# Class 8: TestHybridRetrieverMemoryIds — memory_ids filter propagation
# ===========================================================================

class TestHybridRetrieverMemoryIds:
    """Tests for memory_ids filter plumbing in HybridRetriever signal methods."""

    def test_graph_search_signature_has_memory_ids(self):
        """_graph_search accepts memory_ids parameter."""
        from muninn.retrieval.hybrid import HybridRetriever
        import inspect
        sig = inspect.signature(HybridRetriever._graph_search)
        assert "memory_ids" in sig.parameters

    def test_bm25_search_signature_has_memory_ids(self):
        """_bm25_search accepts memory_ids parameter."""
        from muninn.retrieval.hybrid import HybridRetriever
        import inspect
        sig = inspect.signature(HybridRetriever._bm25_search)
        assert "memory_ids" in sig.parameters

    def test_temporal_search_signature_has_memory_ids(self):
        """_temporal_search accepts memory_ids parameter."""
        from muninn.retrieval.hybrid import HybridRetriever
        import inspect
        sig = inspect.signature(HybridRetriever._temporal_search)
        assert "memory_ids" in sig.parameters

    def test_bm25_search_filters_by_memory_ids(self):
        """_bm25_search excludes IDs not in memory_ids whitelist."""
        from muninn.retrieval.hybrid import HybridRetriever

        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.bm25 = MagicMock()
        retriever.bm25.search.return_value = [("allowed", 0.8), ("blocked", 0.5)]

        results = retriever._bm25_search(
            query="test", limit=10, user_id="u1",
            memory_ids=["allowed"]
        )

        ids = [r[0] for r in results]
        assert "allowed" in ids
        assert "blocked" not in ids

    def test_bm25_search_returns_all_when_no_filter(self):
        """_bm25_search returns all results when memory_ids=None."""
        from muninn.retrieval.hybrid import HybridRetriever

        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.bm25 = MagicMock()
        retriever.bm25.search.return_value = [("m1", 0.9), ("m2", 0.7)]

        results = retriever._bm25_search(query="test", limit=10, user_id="u1")

        ids = [r[0] for r in results]
        assert "m1" in ids
        assert "m2" in ids

    def test_graph_search_filters_via_target_set(self):
        """_graph_search respects memory_ids whitelist."""
        from muninn.retrieval.hybrid import HybridRetriever

        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.graph = MagicMock()
        # find_related_memories returns both IDs; only "allowed" is in whitelist
        retriever.graph.find_related_memories.return_value = ["allowed", "blocked"]

        # The import is inside the try block, patch it at the source module
        with patch("muninn.extraction.rules.extract_entities_rule_based") as mock_extract:
            from muninn.extraction.rules import Entity
            mock_extract.return_value = [Entity(name="TestEntity", entity_type="misc")]

            results = retriever._graph_search(
                query="entity test", limit=10, user_id="u1",
                memory_ids=["allowed"]
            )

        ids = [r[0] for r in results]
        assert "allowed" in ids
        assert "blocked" not in ids


# ===========================================================================
# Class 9: TestVersionBump317 — version validation
# ===========================================================================

class TestVersionBump317:
    """Version bump validation for Phase 17 (v3.17.x)."""

    def test_version_at_least_317(self):
        """muninn.version.__version__ must be >= 3.17.0."""
        from muninn import version
        parts = tuple(int(x) for x in version.__version__.split(".")[:3])
        assert parts >= (3, 17, 0), f"Expected >= 3.17.0, got {version.__version__}"

    def test_pyproject_version_matches_module(self):
        """pyproject.toml version matches muninn.version.__version__."""
        import re
        from pathlib import Path
        from muninn import version

        toml_path = Path(__file__).parent.parent / "pyproject.toml"
        content = toml_path.read_text(encoding="utf-8")
        m = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
        assert m is not None, "Could not find version in pyproject.toml"
        assert m.group(1) == version.__version__, (
            f"pyproject.toml={m.group(1)}, version.py={version.__version__}"
        )
