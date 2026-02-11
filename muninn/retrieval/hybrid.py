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
from muninn.store.sqlite_metadata import SQLiteMetadataStore
from muninn.store.vector_store import VectorStore
from muninn.store.graph_store import GraphStore
from muninn.retrieval.bm25 import BM25Index
from muninn.retrieval.reranker import Reranker

logger = logging.getLogger("Muninn.Retrieval")

# RRF constant (standard value from literature)
RRF_K = 60

# Signal weights for RRF contribution
SIGNAL_WEIGHTS = {
    "vector": 1.0,
    "graph": 1.0,
    "bm25": 0.8,
    "temporal": 0.5,
}


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

        Returns:
            List of SearchResult sorted by relevance.
        """
        t0 = time.time()

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

        # --- Reciprocal Rank Fusion ---
        fused_scores = self._rrf_fusion(
            vector_results=vector_results,
            graph_results=graph_results,
            bm25_results=bm25_results,
            temporal_results=temporal_results,
        )

        if not fused_scores:
            return []

        # --- Apply importance weighting ---
        weighted = self._apply_importance_weighting(fused_scores)

        # --- Sort and select candidates ---
        candidates = sorted(weighted.items(), key=lambda x: x[1], reverse=True)
        candidate_ids = [c[0] for c in candidates[:limit * 2]]

        # --- Fetch full records ---
        records = self.metadata.get_by_ids(candidate_ids)
        record_map = {r.id: r for r in records}

        # --- Reranking ---
        if rerank and self.reranker and self.reranker.is_available and records:
            results = self._rerank_candidates(query, candidates[:limit * 2], record_map, limit)
        else:
            results = self._build_results(candidates[:limit], record_map)

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
    ) -> Dict[str, float]:
        """
        Reciprocal Rank Fusion across all retrieval signals.

        RRF score = sum(weight / (k + rank + 1)) for each signal.
        """
        scores: Dict[str, float] = defaultdict(float)

        signal_data = [
            (vector_results, SIGNAL_WEIGHTS["vector"]),
            (graph_results, SIGNAL_WEIGHTS["graph"]),
            (bm25_results, SIGNAL_WEIGHTS["bm25"]),
            (temporal_results, SIGNAL_WEIGHTS["temporal"]),
        ]

        for results, weight in signal_data:
            for doc_id, rank in results:
                scores[doc_id] += weight / (RRF_K + rank + 1)

        return dict(scores)

    def _apply_importance_weighting(self, rrf_scores: Dict[str, float]) -> Dict[str, float]:
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
                weighted[mem_id] = rrf_score * (0.7 + 0.3 * importance)
            else:
                weighted[mem_id] = rrf_score * 0.7

        return weighted

    def _rerank_candidates(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        record_map: Dict[str, MemoryRecord],
        limit: int,
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
                results.append(SearchResult(
                    memory=record_map[doc_id],
                    score=score,
                    source="hybrid+rerank",
                ))

        return results

    def _build_results(
        self,
        candidates: List[Tuple[str, float]],
        record_map: Dict[str, MemoryRecord],
    ) -> List[SearchResult]:
        """Build SearchResult list from scored candidates."""
        results = []
        for mem_id, score in candidates:
            if mem_id in record_map:
                results.append(SearchResult(
                    memory=record_map[mem_id],
                    score=score,
                    source="hybrid",
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
