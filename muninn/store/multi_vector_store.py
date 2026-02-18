"""
Muninn Multi-Vector Store
--------------------------
Qdrant-backed storage for ColBERT native multi-vector (MaxSim) retrieval.

Unlike the token-level ColBERT index (one Qdrant point per token), this module
stores *all* token vectors for a document as a **single Qdrant point** using
Qdrant's ``MultiVectorConfig`` (available from qdrant-client ≥ 1.8 /
Qdrant server ≥ 1.10).

Benefits over the per-token approach:
- Drastically fewer points → smaller index, faster scrolls
- Native MaxSim computation on the Qdrant side during ``query_points``
  (avoids Python-level MaxSim loops for many candidates)
- A single payload lookup per document instead of N token lookups

Graceful fallback: if ``MultiVectorConfig`` / ``MultiVectorComparator`` are not
importable (older qdrant-client), the collection falls back to plain dense
vectors using the document centroid (mean of token vectors) and MaxSim is
approximated by cosine similarity.
"""

from __future__ import annotations

import logging
import math
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger("Muninn.MultiVectorStore")

# ---------------------------------------------------------------------------
# Multi-vector capability probe
# ---------------------------------------------------------------------------

try:
    from qdrant_client.models import MultiVectorComparator, MultiVectorConfig

    _MULTIVEC_AVAILABLE = True
    logger.debug("qdrant MultiVectorConfig available — native MaxSim enabled")
except ImportError:
    _MULTIVEC_AVAILABLE = False
    logger.warning(
        "qdrant_client MultiVectorConfig not available (upgrade to ≥1.8 for native MaxSim). "
        "Falling back to centroid-vector approximation."
    )
    MultiVectorComparator = None  # type: ignore[assignment,misc]
    MultiVectorConfig = None      # type: ignore[assignment,misc]

DEFAULT_COLLECTION = "muninn_colbert_multivec"
DEFAULT_DIM = 128


class MultiVectorStore:
    """
    Manages per-document multi-vector (token-level) storage in Qdrant for
    ColBERT late-interaction (MaxSim) retrieval.

    Thread-safety: A single ``QdrantClient`` is shared; qdrant-client's client
    object is not thread-safe for *concurrent writes* to the same collection
    from multiple threads, but individual operations (upsert, query) are safe
    in practice when Qdrant is accessed through the local file-based client.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str = DEFAULT_COLLECTION,
        dim: int = DEFAULT_DIM,
    ) -> None:
        self._client = qdrant_client
        self.collection_name = collection_name
        self.dim = dim
        self._native_multivec = _MULTIVEC_AVAILABLE
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        if self._initialized:
            return

        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection_name not in existing:
            self._create_collection()
            logger.info(
                "Created multi-vector collection '%s' (dim=%d, native=%s)",
                self.collection_name,
                self.dim,
                self._native_multivec,
            )
        else:
            logger.debug("Multi-vector collection '%s' exists", self.collection_name)

        self._initialized = True

    def _create_collection(self) -> None:
        if self._native_multivec:
            vec_cfg = VectorParams(
                size=self.dim,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM,
                ),
            )
        else:
            # Fallback: plain dense collection using centroid vector
            vec_cfg = VectorParams(
                size=self.dim,
                distance=Distance.COSINE,
                on_disk=True,
            )

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vec_cfg,
        )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def upsert(
        self,
        memory_id: str,
        token_vectors: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store (or overwrite) all token vectors for *memory_id*.

        Args:
            memory_id: Unique identifier for the memory.
            token_vectors: shape ``(num_tokens, dim)``; should be L2-normalised.
            metadata: Optional extra payload fields (e.g. ``user_id``,
                      ``namespace``, ``created_at``).

        Returns:
            The Qdrant point ID (deterministic UUID derived from memory_id).
        """
        self.ensure_collection()

        if token_vectors.ndim != 2 or token_vectors.shape[0] == 0:
            raise ValueError(
                f"token_vectors must be 2D with at least one row, got shape {token_vectors.shape}"
            )

        point_id = self._point_id(memory_id)
        payload: Dict[str, Any] = {"memory_id": memory_id}
        if metadata:
            payload.update(metadata)

        if self._native_multivec:
            # Each row is one token vector; Qdrant stores them as a list of vectors
            vector: Any = token_vectors.tolist()
        else:
            # Centroid approximation: mean-pool then re-normalise
            centroid = token_vectors.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            vector = centroid.tolist()

        self._client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        return point_id

    def delete(self, memory_id: str) -> bool:
        """Remove the multi-vector point for *memory_id*."""
        self.ensure_collection()
        point_id = self._point_id(memory_id)
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=[point_id],
        )
        return True

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def search(
        self,
        query_vectors: np.ndarray,
        limit: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search for the most relevant documents using MaxSim (or cosine fallback).

        Args:
            query_vectors: shape ``(num_query_tokens, dim)``; L2-normalised.
            limit: Maximum number of results.
            score_threshold: Minimum score to include.
            filters: Optional payload field equality filters (key→value).

        Returns:
            List of ``(memory_id, score)`` ordered by descending score.
        """
        self.ensure_collection()

        if query_vectors.ndim != 2 or query_vectors.shape[0] == 0:
            return []

        query_filter = self._build_filter(filters)

        if self._native_multivec:
            # Native MaxSim: pass the full query token matrix
            query_vec: Any = query_vectors.tolist()
        else:
            # Centroid approximation for query as well
            centroid = query_vectors.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            query_vec = centroid.tolist()

        try:
            results = self._client.query_points(
                collection_name=self.collection_name,
                query=query_vec,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
            ).points
        except Exception as exc:
            logger.warning("MultiVectorStore search failed: %s", exc)
            return []

        return [
            (hit.payload.get("memory_id", ""), hit.score)
            for hit in results
            if hit.payload and "memory_id" in hit.payload
        ]

    def get_vectors(self, memory_id: str) -> Optional[np.ndarray]:
        """
        Retrieve stored token vectors for *memory_id*.

        Returns:
            ``np.ndarray`` of shape ``(num_tokens, dim)``, or ``None`` if not found.
        """
        self.ensure_collection()
        point_id = self._point_id(memory_id)
        try:
            results = self._client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_vectors=True,
            )
        except Exception as exc:
            logger.warning("MultiVectorStore retrieve failed for %s: %s", memory_id, exc)
            return None

        if not results:
            return None

        vec = results[0].vector
        if vec is None:
            return None

        if isinstance(vec, dict):
            # Named-vector format (shouldn't occur for this collection but be safe)
            vec = next(iter(vec.values()))

        arr = np.array(vec, dtype=float)
        if arr.ndim == 1:
            # Centroid fallback stored as flat vector — return as (1, dim)
            return arr.reshape(1, -1)
        return arr  # (num_tokens, dim)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    @property
    def is_native_multivec(self) -> bool:
        """True if native MaxSim is used; False if centroid fallback is active."""
        return self._native_multivec

    def count(self) -> int:
        """Return the number of documents (points) indexed."""
        self.ensure_collection()
        info = self._client.get_collection(self.collection_name)
        return info.points_count or 0

    def delete_collection(self) -> None:
        """Drop and recreate the collection (used in tests / migrations)."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection_name in existing:
            self._client.delete_collection(self.collection_name)
        self._initialized = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _point_id(memory_id: str) -> str:
        """Deterministic UUID-v5 point ID derived from memory_id."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"multivec:{memory_id}"))

    @staticmethod
    def _build_filter(filters: Optional[Dict[str, str]]) -> Optional[Filter]:
        if not filters:
            return None
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
            if v is not None
        ]
        return Filter(must=conditions) if conditions else None


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_multi_vector_store(
    qdrant_path: str,
    collection_name: str = DEFAULT_COLLECTION,
    dim: int = DEFAULT_DIM,
) -> MultiVectorStore:
    """
    Convenience factory: open (or create) a local Qdrant database and wrap it
    with a :class:`MultiVectorStore`.

    Args:
        qdrant_path: Filesystem path for the local Qdrant database.
        collection_name: Name of the multi-vector collection.
        dim: Token-embedding dimension (must match the ColBERT encoder).

    Returns:
        An initialised :class:`MultiVectorStore`.
    """
    client = QdrantClient(path=qdrant_path)
    store = MultiVectorStore(client, collection_name=collection_name, dim=dim)
    store.ensure_collection()
    return store
