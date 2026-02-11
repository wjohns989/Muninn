"""
Muninn Reranker
---------------
Cross-encoder reranking using Jina Tiny or FastEmbed rerankers.
Provides precision reranking of retrieval candidates.

Falls back gracefully if no reranker model is available.
"""

import logging
from typing import List, Tuple, Optional

logger = logging.getLogger("Muninn.Reranker")


class Reranker:
    """
    Cross-encoder reranker for precision scoring of retrieval candidates.

    Supports:
    - fastembed TextCrossEncoder (preferred, local, fast)
    - Disabled mode (passthrough) if no model available

    The reranker takes a query and a list of documents, and returns
    relevance scores that are more accurate than bi-encoder similarity
    at the cost of being non-parallelizable (O(n) forward passes).
    """

    def __init__(self, model_name: str = "jinaai/jina-reranker-v1-tiny-en"):
        self._model = None
        self._model_name = model_name
        self._available = False
        self._init_model()

    def _init_model(self) -> None:
        """Try to initialize the cross-encoder model."""
        try:
            from fastembed import TextCrossEncoder
            self._model = TextCrossEncoder(model_name=self._model_name)
            self._available = True
            logger.info("Reranker initialized: %s", self._model_name)
        except ImportError:
            logger.warning("fastembed not installed — reranker disabled")
            self._available = False
        except Exception as e:
            logger.warning("Reranker init failed: %s — disabled", e)
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def rerank(
        self,
        query: str,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Rerank documents by cross-encoder relevance to query.

        Args:
            query: The search query.
            documents: List of document texts to score.
            doc_ids: Optional list of document IDs (parallel to documents).
                     If None, indices are used as IDs.
            limit: Maximum results to return.

        Returns:
            List of (doc_id, relevance_score) sorted descending by score.
        """
        if not documents:
            return []

        if doc_ids is None:
            doc_ids = [str(i) for i in range(len(documents))]

        if not self._available:
            # Passthrough: return in original order with synthetic scores
            return [(doc_ids[i], 1.0 - i * 0.01) for i in range(min(limit, len(documents)))]

        try:
            # fastembed TextCrossEncoder expects list of (query, doc) pairs
            pairs = [(query, doc) for doc in documents]
            scores = list(self._model.rerank(query, documents))

            # Pair scores with doc_ids
            scored = list(zip(doc_ids, scores))
            # Sort by score descending
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:limit]
        except Exception as e:
            logger.error("Reranking failed: %s — returning unranked", e)
            return [(doc_ids[i], 1.0 - i * 0.01) for i in range(min(limit, len(documents)))]
