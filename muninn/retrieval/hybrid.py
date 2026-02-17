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
5. Goal Relevance (goal-vector similarity prior)

v3.1.0: Added explainable recall traces (per-signal attribution).

Inspired by:
- ColBERT late interaction for fine-grained matching
- Reciprocal Rank Fusion for multi-signal combination
- Kalman filtering for adaptive signal weighting
"""

import logging
import time
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict

from muninn.core.types import MemoryRecord, SearchResult
import numpy as np
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
from muninn.chains import MemoryChainRetriever
from muninn.advanced.colbert import ColBERTScorer
from muninn.observability import OTelGenAITracer

# Optional Qdrant imports for late interaction
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

logger = logging.getLogger("Muninn.Retrieval")

# RRF constant (standard value from literature)
RRF_K = 60

# Signal weights for RRF contribution (used when adaptive_weights is OFF)
SIGNAL_WEIGHTS = {
    "vector": 1.0,
    "graph": 1.0,
    "bm25": 0.8,
    "temporal": 0.5,
    "chain": 0.6,
}
GOAL_SIGNAL_WEIGHT = 0.65

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
        colbert_indexer: Optional[Any] = None,
        embed_fn=None,
        telemetry: Optional[OTelGenAITracer] = None,
        chain_signal_weight: float = 0.6,
        chain_expansion_limit: int = 20,
        chain_max_seed_memories: int = 6,
    ):
        self.metadata = metadata_store
        self.vectors = vector_store
        self.graph = graph_store
        self.bm25 = bm25_index
        self.reranker = reranker
        self._embed_fn = embed_fn  # async or sync function: text → List[float]
        self._telemetry = telemetry or OTelGenAITracer(enabled=False)
        self._chain_signal_weight = max(0.0, float(chain_signal_weight))
        self._chain_expansion_limit = max(0, int(chain_expansion_limit))
        self._chain_retriever = MemoryChainRetriever(
            graph_store=self.graph,
            max_seed_memories=max(1, int(chain_max_seed_memories)),
        )
        
        # ColBERT (Phase 6)
        flags = get_flags()
        self._colbert_enabled = flags.is_enabled("colbert")
        self._colbert_indexer = colbert_indexer
        self._colbert_scorer = ColBERTScorer()

    async def search(
        self,
        query: str,
        limit: int = 10,
        user_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True,
        namespaces: Optional[List[str]] = None,
        explain: bool = False,
        goal_embedding: Optional[List[float]] = None,
        goal_signal_weight: float = GOAL_SIGNAL_WEIGHT,
        feedback_signal_multipliers: Optional[Dict[str, float]] = None,
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
        with self._telemetry.span(
            "muninn.retrieval.search",
            {
                "gen_ai.operation.name": "retrieval.search",
                "gen_ai.system": "muninn",
                "muninn.limit": limit,
                "muninn.user_id": user_id,
            },
        ):
            # Check if explainable recall is enabled via feature flags
            flags = get_flags()
            generate_traces = explain and flags.is_enabled("explainable_recall")

            # Generate query embedding
            query_embedding = await self._get_embedding(query)

            # Scope filters are always applied to vector search. We also enforce
            # user/namespace constraints again after metadata fetch.
            effective_filters = dict(filters or {})
            if user_id and "user_id" not in effective_filters:
                effective_filters["user_id"] = user_id
            if namespaces and len(namespaces) == 1 and "namespace" not in effective_filters:
                effective_filters["namespace"] = namespaces[0]

        # --- Parallel retrieval across all signals ---
        # We use asyncio.to_thread to run synchronous store operations in separate threads,
        # avoiding blocking the event loop and allowing potential I/O parallelism.

            (
                vector_results,
                graph_results,
                bm25_results,
                goal_results,
                temporal_results,
            ) = await asyncio.gather(
                asyncio.to_thread(self._vector_search, query_embedding, limit * 3, effective_filters),
                asyncio.to_thread(self._graph_search, query, limit * 2, user_id=user_id, namespaces=namespaces),
                asyncio.to_thread(self._bm25_search, query, limit * 2, user_id=user_id, namespaces=namespaces),
                asyncio.to_thread(self._goal_search, goal_embedding, limit * 2, effective_filters),
                asyncio.to_thread(
                    self._temporal_search,
                    filters=effective_filters,
                    namespaces=namespaces,
                    user_id=user_id,
                    limit=limit * 2,
                ),
            )

            # Chain search depends on other signals, so it runs after
            # We run it in a thread as well for consistency/non-blocking
            chain_results = await asyncio.to_thread(
                self._chain_search,
                vector_results=vector_results,
                graph_results=graph_results,
                bm25_results=bm25_results,
                goal_results=goal_results,
                temporal_results=temporal_results,
                limit=min(limit * 2, self._chain_expansion_limit),
                enabled=flags.is_enabled("memory_chains"),
            )

        # --- Compute signal weights (adaptive or fixed) ---
            use_adaptive = flags.is_enabled("adaptive_weights")
            if use_adaptive:
                adapter = _get_weight_adapter()
                signal_results_for_entropy = {
                    "vector": vector_results,
                    "graph": graph_results,
                    "bm25": bm25_results,
                    "temporal": temporal_results,
                    "chain": chain_results,
                }
                active_weights = adapter.compute_weights(
                    query,
                    signal_results_for_entropy,
                    feedback_multipliers=feedback_signal_multipliers,
                )
            else:
                active_weights = SIGNAL_WEIGHTS

        # --- Reciprocal Rank Fusion (with optional trace tracking) ---
            if generate_traces:
                fused_scores, traces = self._rrf_fusion_with_traces(
                    vector_results=vector_results,
                    graph_results=graph_results,
                    bm25_results=bm25_results,
                    goal_results=goal_results,
                    temporal_results=temporal_results,
                    chain_results=chain_results,
                    weights=active_weights,
                    goal_signal_weight=goal_signal_weight,
                    chain_signal_weight=self._chain_signal_weight,
                )
            else:
                fused_scores = self._rrf_fusion(
                    vector_results=vector_results,
                    graph_results=graph_results,
                    bm25_results=bm25_results,
                    goal_results=goal_results,
                    temporal_results=temporal_results,
                    chain_results=chain_results,
                    weights=active_weights,
                    goal_signal_weight=goal_signal_weight,
                    chain_signal_weight=self._chain_signal_weight,
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
            record_map = {
                r.id: r
                for r in records
                if self._record_matches_constraints(
                    record=r,
                    user_id=user_id,
                    namespaces=namespaces,
                    filters=effective_filters,
                )
            }

        # --- Reranking ---
            if rerank and records:
                # ColBERT late-interaction reranking (Phase 6)
                if self._colbert_enabled and self._colbert_indexer:
                    results = await self._colbert_rerank(
                        query, candidates[:limit * 2], record_map, limit, traces
                    )
                # Standard Cross-Encoder fallback
                elif self.reranker and self.reranker.is_available:
                    results = self._rerank_candidates(
                        query, candidates[:limit * 2], record_map, limit, traces
                    )
                else:
                    results = self._build_results(candidates[:limit], record_map, traces)
            else:
                results = self._build_results(candidates[:limit], record_map, traces)

        # --- Record access for accessed memories (Batch Optimized) ---
            if results:
                accessed_ids = [r.memory.id for r in results]
                self.metadata.record_access_batch(accessed_ids)

            elapsed = time.time() - t0
            logger.debug("Hybrid search completed: %d results in %.3fs", len(results), elapsed)
            self._telemetry.add_event(
                "muninn.retrieval.result",
                {
                    "muninn.result_count": len(results),
                    "muninn.elapsed_ms": round(elapsed * 1000.0, 2),
                },
            )
            return results

    def _vector_search(
        self,
        query_embedding: List[float],
        limit: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        """Vector similarity search. Returns list of (id, score)."""
        try:
            results = self.vectors.search(
                query_embedding=query_embedding,
                limit=limit,
                filters=filters,
            )
            # results are (id, score) tuples from Qdrant
            return [(str(r[0]), float(r[1])) for r in results]
        except Exception as e:
            logger.warning("Vector search failed: %s", e)
            return []

    def _graph_search(
        self, 
        query: str, 
        limit: int,
        user_id: Optional[str] = None,
        namespaces: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """Graph-based entity search. Returns list of (id, score) (scoped)."""
        try:
            # Extract potential entity names from query
            words = query.split()
            entity_candidates = [w for w in words if len(w) > 2]

            if not entity_candidates:
                return []

            # We take the first namespace if provided, or default to global
            ns = namespaces[0] if namespaces and len(namespaces) == 1 else "global"
            uid = user_id or "global"

            scores: Dict[str, float] = defaultdict(float)
            for idx, entity_name in enumerate(entity_candidates[:5]):
                related = self.graph.find_related_memories(
                    [entity_name], 
                    limit=limit,
                    user_id=uid,
                    namespace=ns
                )
                if not related:
                    continue
                query_entity_weight = 1.0 / (idx + 1.0)
                for mem_id in related:
                    scores[mem_id] += query_entity_weight

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
            return ranked
        except Exception as e:
            logger.warning("Graph search failed: %s", e)
            return []

    def _bm25_search(
        self,
        query: str,
        limit: int,
        user_id: Optional[str] = None,
        namespaces: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """BM25 keyword search. Returns list of (id, score)."""
        try:
            results = self.bm25.search(
                query,
                limit=limit,
                user_id=user_id,
                namespaces=namespaces,
            )
            return [(doc_id, float(score)) for doc_id, score in results]
        except Exception as e:
            logger.warning("BM25 search failed: %s", e)
            return []

    def _goal_search(
        self,
        goal_embedding: Optional[List[float]],
        limit: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        """Goal prior search against memory vectors. Returns list of (id, score)."""
        if not goal_embedding:
            return []
        try:
            results = self.vectors.search(
                query_embedding=goal_embedding,
                limit=limit,
                filters=filters,
            )
            return [(str(r[0]), float(r[1])) for r in results]
        except Exception as e:
            logger.warning("Goal search failed: %s", e)
            return []

    def _temporal_search(
        self,
        filters: Optional[Dict[str, Any]],
        namespaces: Optional[List[str]],
        user_id: Optional[str],
        limit: int,
    ) -> List[Tuple[str, float]]:
        """Temporal/metadata search via SQLite. Returns list of (id, score)."""
        try:
            project = filters.get("project") if filters else None
            # Pull more than needed when filtering across multiple namespaces.
            records = self.metadata.get_all(
                limit=limit * 2,
                project=project,
                namespace=namespaces[0] if namespaces and len(namespaces) == 1 else None,
                user_id=user_id,
            )
            if namespaces:
                ns = set(namespaces)
                records = [r for r in records if r.namespace in ns]

            # Blend recency and intrinsic importance into temporal score.
            now = time.time()
            scored: List[Tuple[str, float]] = []
            for r in records:
                age_seconds = max(0.0, now - r.created_at)
                recency = 1.0 / (1.0 + (age_seconds / 86400.0))
                temporal_score = 0.7 * recency + 0.3 * float(r.importance)
                scored.append((r.id, temporal_score))

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:limit]
        except Exception as e:
            logger.warning("Temporal search failed: %s", e)
            return []

    def _rrf_fusion(
        self,
        vector_results: List[Tuple[str, float]],
        graph_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]],
        goal_results: List[Tuple[str, float]],
        temporal_results: List[Tuple[str, float]],
        chain_results: List[Tuple[str, float]],
        weights: Optional[Dict[str, float]] = None,
        goal_signal_weight: float = GOAL_SIGNAL_WEIGHT,
        chain_signal_weight: float = 0.6,
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
            (goal_results, w.get("goal", goal_signal_weight)),
            (temporal_results, w.get("temporal", 0.5)),
            (chain_results, w.get("chain", chain_signal_weight)),
        ]

        for results, weight in signal_data:
            for rank, (doc_id, _raw_score) in enumerate(results):
                scores[doc_id] += weight / (RRF_K + rank + 1)

        return dict(scores)

    def _rrf_fusion_with_traces(
        self,
        vector_results: List[Tuple[str, float]],
        graph_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]],
        goal_results: List[Tuple[str, float]],
        temporal_results: List[Tuple[str, float]],
        chain_results: List[Tuple[str, float]],
        weights: Optional[Dict[str, float]] = None,
        goal_signal_weight: float = GOAL_SIGNAL_WEIGHT,
        chain_signal_weight: float = 0.6,
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
            ("goal", goal_results, w.get("goal", goal_signal_weight)),
            ("temporal", temporal_results, w.get("temporal", 0.5)),
            ("chain", chain_results, w.get("chain", chain_signal_weight)),
        ]

        for signal_name, results, weight in signal_data:
            for rank, (doc_id, raw_score) in enumerate(results):
                rrf_contrib = weight / (RRF_K + rank + 1)
                scores[doc_id] += rrf_contrib

                # Initialize trace if needed
                if doc_id not in traces:
                    traces[doc_id] = RecallTrace(memory_id=doc_id)

                # Record this signal's contribution
                traces[doc_id].signals.append(
                    create_signal_contribution(
                        signal=signal_name,
                        raw_score=float(raw_score),
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
        mem_ids = list(rrf_scores.keys())
        
        # Batch fetch all records to avoid N+1 queries (P0 Performance Fix)
        records = self.metadata.get_by_ids(mem_ids)
        record_cache = {r.id: r for r in records}

        for mem_id, rrf_score in rrf_scores.items():
            record = record_cache.get(mem_id)
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

    async def _colbert_rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        record_map: Dict[str, MemoryRecord],
        limit: int,
        traces: Optional[Dict[str, RecallTrace]] = None,
    ) -> List[SearchResult]:
        """
        Apply ColBERT late-interaction scoring to a candidate set.
        """
        flags = get_flags()
        if not flags.is_enabled("colbert") or not self._colbert_indexer or not self._colbert_indexer.encoder.is_available:
            return self._rerank_candidates(query, candidates, record_map, limit, traces)

        # 1. Encode query at token level
        query_vectors = self._colbert_indexer.encoder.encode(query)
        if query_vectors.size == 0:
            return self._build_results(candidates[:limit], record_map, traces)

        # 2. Get relevant centroids for the query (PLAID Phase 1) if enabled
        relevant_centroids = None
        if flags.is_enabled("colbert_plaid"):
            # union of top-8 centroids for each query token
            relevant_centroids = self._colbert_indexer.get_query_centroids(query_vectors, top_k=8)
        
        # 3. Security/Scoping: Resolve target namespace and user for strict isolation
        # We use search metrics context or default to record_map items
        target_records = list(record_map.values())
        target_user = target_records[0].user_id if target_records else None
        target_namespace = target_records[0].namespace if target_records else None
        
        scored_candidates = []
        for mem_id, _fused_score in candidates:
            if mem_id not in record_map:
                continue
            
            # 4. Retrieve ONLY document tokens matching the relevant centroids
            # This is the core PLAID-Lite optimization
            client = self.vectors._get_client()
            # Filter by memory_id AND user/namespace for strict isolation
            must_conditions = [
                FieldCondition(key="memory_id", match=MatchValue(value=mem_id))
            ]
            if target_user:
                must_conditions.append(FieldCondition(key="user_id", match=MatchValue(value=target_user)))
            if target_namespace:
                must_conditions.append(FieldCondition(key="namespace", match=MatchValue(value=target_namespace)))
                
            if relevant_centroids:
                must_conditions.append(
                    FieldCondition(key="centroid_id", match=MatchAny(any=relevant_centroids))
                )

            results = client.scroll(
                collection_name=self._colbert_indexer.collection_name,
                scroll_filter=Filter(must=must_conditions),
                limit=512,
                with_vectors=True
            )
            
            points = results[0]
            if not points:
                # Fallback: maybe centroids didn't match, or not indexed yet?
                # Or we can just score it as 0
                scored_candidates.append((mem_id, 0.0))
                continue
                
            doc_vectors = np.array([p.vector for p in points])
                
            # 4. Compute MaxSim score on subset
            score = self._colbert_scorer.maxsim_score(query_vectors, doc_vectors)
            scored_candidates.append((mem_id, score))

        # Sort by ColBERT score
        ranked = sorted(scored_candidates, key=lambda x: x[1], reverse=True)[:limit]
        
        # Build results
        results = []
        for doc_id, score in ranked:
            trace = None
            if traces and doc_id in traces:
                traces[doc_id].rerank_score = score
                traces[doc_id].finalize()
                trace = traces[doc_id]

            results.append(SearchResult(
                memory=record_map[doc_id],
                score=score,
                source="hybrid+colbert",
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

    def _record_matches_constraints(
        self,
        record: MemoryRecord,
        user_id: Optional[str],
        namespaces: Optional[List[str]],
        filters: Optional[Dict[str, Any]],
    ) -> bool:
        """Apply final in-memory scope checks to prevent cross-user leakage."""
        metadata = record.metadata or {}

        if user_id and metadata.get("user_id") != user_id:
            return False

        if namespaces and record.namespace not in namespaces:
            return False

        if filters:
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

    def _chain_search(
        self,
        *,
        vector_results: List[Tuple[str, float]],
        graph_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]],
        goal_results: List[Tuple[str, float]],
        temporal_results: List[Tuple[str, float]],
        limit: int,
        enabled: bool,
    ) -> List[Tuple[str, float]]:
        """Expand candidates using memory-chain graph links."""
        if not enabled or limit <= 0:
            return []
        try:
            seed_ranked: List[Tuple[str, float]] = []
            for signal_results in (
                vector_results,
                graph_results,
                bm25_results,
                goal_results,
                temporal_results,
            ):
                if not signal_results:
                    continue
                seed_ranked.extend(signal_results[:6])
                if len(seed_ranked) >= (self._chain_retriever.max_seed_memories * 3):
                    break
            if not seed_ranked:
                return []
            return self._chain_retriever.expand_from_ranked_results(
                ranked_results=seed_ranked,
                limit=limit,
            )
        except Exception as e:
            logger.warning("Chain search failed: %s", e)
            return []

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