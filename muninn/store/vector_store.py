"""
Muninn Vector Store
-------------------
Qdrant-based vector storage for embedding-based memory retrieval.
Wraps qdrant-client with Muninn-specific operations.
"""

import logging
import uuid
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

logger = logging.getLogger("Muninn.Vector")

DEFAULT_COLLECTION = "muninn_memories"
DEFAULT_DIMS = 768


class VectorStore:
    """Manages vector embeddings in local Qdrant for similarity search."""

    def __init__(
        self,
        data_path,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_dims: int = DEFAULT_DIMS,
    ):
        self.data_path = Path(data_path) if not isinstance(data_path, Path) else data_path
        self.collection_name = collection_name
        self.embedding_dims = embedding_dims
        self._client: Optional[QdrantClient] = None
        self._initialize()

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(path=str(self.data_path))
        return self._client

    def _initialize(self):
        client = self._get_client()
        collections = [c.name for c in client.get_collections().collections]
        if self.collection_name not in collections:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dims,
                    distance=Distance.COSINE,
                    on_disk=True,
                ),
            )
            logger.info(f"Created vector collection '{self.collection_name}' ({self.embedding_dims} dims)")
        else:
            logger.info(f"Vector collection '{self.collection_name}' exists")

    def upsert(self, memory_id: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Insert or update a vector point. Returns the point ID."""
        client = self._get_client()
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, memory_id))
        payload = {"memory_id": memory_id}
        if metadata:
            payload.update(metadata)

        client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )
        return point_id

    def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.
        Returns list of (memory_id, score) tuples.
        """
        client = self._get_client()
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            query_filter = Filter(must=conditions)

        # v1.16+ uses query_points instead of search
        results = client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
        ).points

        return [
            (hit.payload.get("memory_id", ""), hit.score)
            for hit in results
            if hit.payload and "memory_id" in hit.payload
        ]

    def get_vector(self, memory_id: str) -> Optional[List[float]]:
        """
        Retrieve the embedding vector for a specific memory_id.
        Used for background integrity auditing.
        """
        try:
            client = self._get_client()
            results = client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id],
                with_vectors=True
            )
            if results and results[0].vector:
                # Qdrant return vectors as a dict if named, or list if unnamed
                vec = results[0].vector
                if isinstance(vec, dict):
                    return vec.get("default", list(vec.values())[0])
                return vec
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve vector for {memory_id}: {e}")
            return None

    def set_payload(self, memory_id: str, payload: Dict[str, Any]) -> bool:
        """Update payload fields for an existing vector point by memory ID."""
        if not payload:
            return False
        client = self._get_client()
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, memory_id))
        client.set_payload(
            collection_name=self.collection_name,
            payload=payload,
            points=[point_id],
        )
        return True

    def delete(self, memory_id: str) -> bool:
        """Delete a vector by memory_id."""
        client = self._get_client()
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, memory_id))
        client.delete(
            collection_name=self.collection_name,
            points_selector=[point_id],
        )
        return True

    def delete_all(self) -> bool:
        """Delete and recreate the collection."""
        client = self._get_client()
        client.delete_collection(self.collection_name)
        self._initialize()
        return True

    def count(self) -> int:
        client = self._get_client()
        info = client.get_collection(self.collection_name)
        return info.points_count or 0

    async def update_collection_quantization(
        self,
        quantization_config: Optional[Any] = None
    ) -> bool:
        """
        Dynamically update collection quantization settings.
        Supported by Qdrant (PATCH /collections/{name}).
        """
        try:
            client = self._get_client()
            client.update_collection(
                collection_name=self.collection_name,
                quantization_config=quantization_config
            )
            logger.info(f"Updated quantization for collection '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to update quantization for '{self.collection_name}': {e}")
            return False

    def close(self):
        if self._client:
            self._client.close()
            self._client = None
