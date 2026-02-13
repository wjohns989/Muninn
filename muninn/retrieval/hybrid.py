"""
Muninn Hybrid Retrieval Engine
-------------------------------
Multi-signal retrieval combining 4 strategies with
Reciprocal Rank Fusion (RRF) and cross-encoder reranking.

Signals:
1. Vector Search (Qdrant HNSW, cosine similarity)
2. Graph Traversal (Kuzu entity-linked memories)
3. Temporal Filtering (SQLite bi-temporal queries)
4. BM25 Keyword (in-memory inverted index)

v3.1.0: Added explainable recall traces (per-signal attribution).

Inspired by:
- ColBERT late interaction for fine-grained matching
- Reciprocal Rank Fusion for multi-signal combination
- Kalman filtering for adaptive signal weighting
"""

import logging
import time
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict

from muninn.core.types import MemoryRecord, SearchResult
from muninn.core.recall_trace import (
    RecallTrace, create_signal_contribution,
)
from muninn.core.feature_flags import get_flags
from muninn.store.sqlite_metadata import SQLiteMetadataStore
from muninn.store.vector_store import VectorStore
from muninn.store.graph_store import GraphStore
from muninn.retrieval.bm25 import BM25Index
from muninn.retrieval.reranker import Reranker
from muninn.retrieval.weight_adapter import WeightAdapter

logger = logging.getLogger("Muninn.Retrieval")

# RRF constant (standard value from literature)
RRF_K = 60

# Signal weights for RRF contribution (used when adaptive_weights is OFF)
SIGNAL_WEIGHTS = {
    "vector": 1.0,
    "graph": 1.0,
    "bm25": 0.8,
    "temporal": 0.5,
}

# Module-level WeightAdapter instance (lazy-initialized)
_weight_adapter: WeightAdapter | None = None


def _get_weight_adapter() -> WeightAdapter:
    """Get or create the module-level WeightAdapter singleton."""
    global _weight_adapter
    if _weight_adapter is None:
        _weight_adapter = WeightAdapter(base_weights=SIGNAL_WEIGHTS)
    return _weight_adapter


class HybridRetriever:
    """
    Multi-signal retrieval engine with Reciprocal Rank Fusion.

    Combines vector similarity, graph traversal, BM25 keyword matching,
    and temporal filtering into a unified retrieval pipeline.
    """

    def __init__(
        self,
        metadata_store: SQLiteMetadataStore,
        vector_store: VectorStore,
        graph_store: GraphStore,
        bm25_index: BM25Index,
        reranker: Optional[Reranker] = None,
        embed_fn=None,
    ):
        self.metadata = metadata_store
        self.vectors = vector_store
        self.graph = graph_store
        self.bm25 = bm25_index
        self.reranker = reranker
        self._embed_fn = embed_fn  # async or sync function: text → List[float]

    async def search(
        self,
        query: str,
        limit: int = 10,
        user_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True,
        namespaces: Optional[List[str]] = None,
        explain: bool = False,
    ) -> List[SearchResult]:
        """
        Execute multi-signal hybrid search with RRF fusion.

        Args:
            query: Search query text.
            limit: Maximum results to return.
            user_id: Optional user filter.
            filters: Optional metadata filters.
            rerank: Whether to apply cross-encoder reranking.
            namespaces: Optional namespace filter list.
            explain: Whether to generate recall traces (v3.1.0).

        Returns:
            List of SearchResult sorted by relevance.
        """
        t0 = time.time()

        # Check if explainable recall is enabled via feature flags
        flags = get_flags()
        generate_traces = explain and flags.is_enabled("explainable_recall")

        # Generate query embedding
        query_embedding = await self._get_embedding(query)

        # --- Parallel retrieval across all signals ---
        # In practice these are all fast local operations so we run them
        # sequentially to avoid asyncio overhead. If any store becomes
        # remote, wrap in asyncio.gather().

        vector_results = self._vector_search(query_embedding, limit * 3, filters)
        graph_results = self._graph_search(query, limit * 2)
        bm25_results = self._bm25_search(query, limit * 2)
        temporal_results = self._temporal_search(filters, namespaces, limit * 2)

        # --- Compute signal weights (adaptive or fixed) ---
        use_adaptive = flags.is_enabled("adaptive_weights")
        if use_adaptive:
            adapter = _get_weight_adapter()
            signal_results_for_entropy = {
                "vector": vector_results,
                "graph": graph_results,
                "bm25": bm25_results,
                "temporal": temporal_results,
            }
            active_weights = adapter.compute_weights(query, signal_results_for_entropy)
        else:
            active_weights = SIGNAL_WEIGHTS

        # --- Reciprocal Rank Fusion (with optional trace tracking) ---
        if generate_traces:
            fused_scores, traces = self._rrf_fusion_with_traces(
                vector_results=vector_results,
                graph_results=graph_results,
                bm25_results=bm25_results,
                temporal_results=temporal_results,
                weights=active_weights,
            )
        else:
            fused_scores = self._rrf_fusion(
                vector_results=vector_results,
                graph_results=graph_results,
                bm25_results=bm25_results,
                temporal_results=temporal_results,
                weights=active_weights,
            )
            traces = {}

        if not fused_scores:
            return []

        # --- Apply importance weighting ---
        weighted = self._apply_importance_weighting(fused_scores, traces)

        # --- Sort and select candidates ---
        candidates = sorted(weighted.items(), key=lambda x: x[1], reverse=True)
        candidate_ids = [c[0] for c in candidates[:limit * 2]]

        # --- Fetch full records ---
        records = self.metadata.get_by_ids(candidate_ids)
        record_map = {r.id: r for r in records}

        # --- Reranking ---
        if rerank and self.reranker and self.reranker.is_available and records:
            results = self._rerank_candidates(
                query, candidates[:limit * 2], record_map, limit, traces
            )
        else:
            results = self._build_results(candidates[:limit], record_map, traces)

        # --- Record access for accessed memories ---
        for r in results:
            self.metadata.record_access(r.memory.id)

        elapsed = time.time() - t0
        logger.debug("Hybrid search completed: %d results in %.3fs", len(results), elapsed)

        return results

    def _vector_search(
        self,
        query_embedding: List[float],
        limit: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, int]]:
        """Vector similarity search. Returns list of (id, rank)."""
        try:
            qdrant_filters = None
            if filters:
                qdrant_filters = self._build_qdrant_filters(filters)

            results = self.vectors.search(
                query_vector=query_embedding,
                limit=limit,
                filter_conditions=qdrant_filters,
            )
            # results are (id, score) tuples from Qdrant
            return [(str(r[0]), rank) for rank, r in enumerate(results)]
        except Exception as e:
            logger.warning("Vector search failed: %s", e)
            return []

    def _graph_search(self, query: str, limit: int) -> List[Tuple[str, int]]:
        """Graph-based entity search. Returns list of (id, rank)."""
        try:
            # Extract potential entity names from query
            # Simple heuristic: capitalized words and known patterns
            words = query.split()
            entity_candidates = [w for w in words if len(w) > 2]

            all_memory_ids = set()
            for entity_name in entity_candidates[:5]:  # Cap to avoid expensive traversals
                related = self.graph.find_related_memories(entity_name)
                all_memory_ids.update(related)

            ranked = list(all_memory_ids)[:limit]
            return [(mid, rank) for rank, mid in enumerate(ranked)]
        except Exception as e:
            logger.warning("Graph search failed: %s", e)
            return []

    def _bm25_search(self, query: str, limit: int) -> List[Tuple[str, int]]:
        """BM25 keyword search. Returns list of (id, rank)."""
        try:
            results = self.bm25.search(query, limit=limit)
            return [(doc_id, rank) for rank, (doc_id, _score) in enumerate(results)]
        except Exception as e:
            logger.warning("BM25 search failed: %s", e)
            return []

    def _temporal_search(
        self,
        filters: Optional[Dict[str, Any]],
        namespaces: Optional[List[str]],
        limit: int,
    ) -> List[Tuple[str, int]]:
        """Temporal/metadata search via SQLite. Returns list of (id, rank)."""
        try:
            # Use metadata store's general query with recency ordering
            records = self.metadata.get_all(
                limit=limit,
                namespace=namespaces[0] if namespaces else None,
            )
            # Already ordered by importance + recency in metadata store
            return [(r.id, rank) for rank, r in enumerate(records)]
        except Exception as e:
            logger.warning("Temporal search failed: %s", e)
            return []

    def _rrf_fusion(
        self,
        vector_results: List[Tuple[str, int]],
        graph_results: List[Tuple[str, int]],
        bm25_results: List[Tuple[str, int]],
        temporal_results: List[Tuple[str, int]],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Reciprocal Rank Fusion across all retrieval signals.

        RRF score = sum(weight / (k + rank + 1)) for each signal.

        Args:
            weights: Signal weights dict. When adaptive_weights is ON,
                     these are computed per-query by WeightAdapter.
                     Falls back to SIGNAL_WEIGHTS if not provided.
        """
        w = weights or SIGNAL_WEIGHTS
        scores: Dict[str, float] = defaultdict(float)

        signal_data = [
            (vector_results, w.get("vector", 1.0)),
            (graph_results, w.get("graph", 1.0)),
            (bm25_results, w.get("bm25", 0.8)),
            (temporal_results, w.get("temporal", 0.5)),
        ]

        for results, weight in signal_data:
            for doc_id, rank in results:
                scores[doc_id] += weight / (RRF_K + rank + 1)

        return dict(scores)

    def _rrf_fusion_with_traces(
        self,
        vector_results: List[Tuple[str, int]],
        graph_results: List[Tuple[str, int]],
        bm25_results: List[Tuple[str, int]],
        temporal_results: List[Tuple[str, int]],
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, RecallTrace]]:
        """
        RRF fusion with per-signal attribution tracking.

        Returns both the fused scores AND RecallTrace objects for each
        document, enabling explainable recall.

        v3.1.0: Unique differentiator — no competitor provides this.
        v3.2.0: Supports adaptive per-query weights from WeightAdapter.
        """
        w = weights or SIGNAL_WEIGHTS
        scores: Dict[str, float] = defaultdict(float)
        traces: Dict[str, RecallTrace] = {}

        signal_data = [
            ("vector", vector_results, w.get("vector", 1.0)),
            ("graph", graph_results, w.get("graph", 1.0)),
            ("bm25", bm25_results, w.get("bm25", 0.8)),
            ("temporal", temporal_results, w.get("temporal", 0.5)),
        ]

        for signal_name, results, weight in signal_data:
            for doc_id, rank in results:
                rrf_contrib = weight / (RRF_K + rank + 1)
                scores[doc_id] += rrf_contrib

                # Initialize trace if needed
                if doc_id not in traces:
                    traces[doc_id] = RecallTrace(memory_id=doc_id)

                # Record this signal's contribution
                # raw_score is the rank for now; vector/bm25 store actual
                # scores but graph/temporal only provide ranks
                traces[doc_id].signals.append(
                    create_signal_contribution(
                        signal=signal_name,
                        raw_score=float(rank),  # Rank as proxy for raw score
                        rank=rank,
                        rrf_contribution=rrf_contrib,
                        weight=weight,
                    )
                )

        return dict(scores), traces

    def _apply_importance_weighting(
        self,
        rrf_scores: Dict[str, float],
        traces: Optional[Dict[str, RecallTrace]] = None,
    ) -> Dict[str, float]:
        """
        Apply importance-based boosting to RRF scores.

        Final = rrf_score * (0.7 + 0.3 * importance)
        This means importance can boost by up to 30%.
        """
        weighted = {}
        for mem_id, rrf_score in rrf_scores.items():
            record = self.metadata.get(mem_id)
            if record:
                importance = record.importance
                boost_factor = 0.7 + 0.3 * importance
                weighted[mem_id] = rrf_score * boost_factor

                # Track importance boost in trace
                if traces and mem_id in traces:
                    traces[mem_id].importance_boost = 0.3 * importance
            else:
                weighted[mem_id] = rrf_score * 0.7

        return weighted

    def _rerank_candidates(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        record_map: Dict[str, MemoryRecord],
        limit: int,
        traces: Optional[Dict[str, RecallTrace]] = None,
    ) -> List[SearchResult]:
        """Apply cross-encoder reranking to candidate set."""
        doc_ids = []
        doc_texts = []
        for mem_id, _score in candidates:
            if mem_id in record_map:
                doc_ids.append(mem_id)
                doc_texts.append(record_map[mem_id].content)

        if not doc_texts:
            return []

        reranked = self.reranker.rerank(
            query=query,
            documents=doc_texts,
            doc_ids=doc_ids,
            limit=limit,
        )

        results = []
        for doc_id, score in reranked:
            if doc_id in record_map:
                trace = None
                if traces and doc_id in traces:
                    traces[doc_id].rerank_score = score
                    traces[doc_id].finalize()
                    trace = traces[doc_id]

                results.append(SearchResult(
                    memory=record_map[doc_id],
                    score=score,
                    source="hybrid+rerank",
                    trace=trace,
                ))

        return results

    def _build_results(
        self,
        candidates: List[Tuple[str, float]],
        record_map: Dict[str, MemoryRecord],
        traces: Optional[Dict[str, RecallTrace]] = None,
    ) -> List[SearchResult]:
        """Build SearchResult list from scored candidates."""
        results = []
        for mem_id, score in candidates:
            if mem_id in record_map:
                trace = None
                if traces and mem_id in traces:
                    traces[mem_id].finalize()
                    trace = traces[mem_id]

                results.append(SearchResult(
                    memory=record_map[mem_id],
                    score=score,
                    source="hybrid",
                    trace=trace,
                ))
        return results

    async def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using the configured embed function."""
        if self._embed_fn is None:
            raise RuntimeError("No embedding function configured")
        result = self._embed_fn(text)
        # Handle both sync and async embed functions
        if hasattr(result, "__await__"):
            return await result
        return result

    def _build_qdrant_filters(self, filters: Dict[str, Any]) -> Optional[Dict]:
        """Convert generic filters to Qdrant filter format."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        conditions = []
        for key, value in filters.items():
            if value is not None:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

        if conditions:
            return Filter(must=conditions)
        return None
