"""
Muninn Conflict Resolver (v3.2.0)
----------------------------------
Executes resolution strategies determined by ConflictDetector.

Resolution strategies:
  - SUPERSEDE: Archive old memory, store new
  - MERGE: Combine into unified memory
  - FLAG_FOR_REVIEW: Return conflict info to caller (no automatic action)
  - KEEP_EXISTING: Discard new memory
"""

import logging
import time
from typing import Optional, Dict, Any

from muninn.core.types import MemoryRecord
from muninn.conflict.detector import ConflictResult, ConflictResolution

logger = logging.getLogger("Muninn.ConflictResolver")

SUPERSEDED_IMPORTANCE_FACTOR = 0.1
VECTOR_CONTENT_PREVIEW_LIMIT = 500
GRAPH_SUMMARY_LIMIT = 200


class ConflictResolver:
    """
    Executes conflict resolution strategies.

    Works with the metadata store and vector store to perform the actual
    resolution actions (archive, merge, flag, discard).
    """

    def __init__(self, metadata_store, vector_store, graph_store, bm25_index, embed_fn=None):
        """
        Args:
            metadata_store: SQLiteMetadataStore instance.
            vector_store: VectorStore instance.
            graph_store: GraphStore instance.
            bm25_index: BM25Index instance.
        """
        self.metadata = metadata_store
        self.vectors = vector_store
        self.graph = graph_store
        self.bm25 = bm25_index
        self.embed_fn = embed_fn

    def resolve(
        self,
        conflict: ConflictResult,
        new_record: Optional[MemoryRecord] = None,
        new_embedding: Optional[list] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the suggested resolution strategy for a conflict.

        Args:
            conflict: The conflict detection result.
            new_record: The new MemoryRecord (needed for SUPERSEDE/MERGE).
            new_embedding: The new memory's embedding vector.
            user_id: Owner identifier for scoped vector metadata updates.

        Returns:
            Dict with resolution outcome details.
        """
        strategy = conflict.suggested_resolution

        if strategy == ConflictResolution.SUPERSEDE:
            return self._resolve_supersede(conflict, new_record, new_embedding)
        elif strategy == ConflictResolution.MERGE:
            return self._resolve_merge(conflict, new_record, new_embedding, user_id)
        elif strategy == ConflictResolution.KEEP_EXISTING:
            return self._resolve_keep_existing(conflict)
        elif strategy == ConflictResolution.FLAG_FOR_REVIEW:
            return self._resolve_flag(conflict)
        else:
            logger.warning("Unknown resolution strategy: %s", strategy)
            return self._resolve_flag(conflict)

    def _resolve_supersede(
        self,
        conflict: ConflictResult,
        new_record: Optional[MemoryRecord],
        new_embedding: Optional[list],
    ) -> Dict[str, Any]:
        """
        SUPERSEDE: Archive old memory, new memory takes its place.

        The old memory is not deleted but marked as superseded in its metadata.
        """
        old_id = conflict.existing_memory_id

        # Mark old memory as superseded
        old_record = self.metadata.get(old_id)
        if old_record:
            old_metadata = old_record.metadata.copy()
            old_metadata["superseded_by"] = new_record.id if new_record else "unknown"
            old_metadata["superseded_at"] = time.time()
            old_metadata["supersede_reason"] = (
                f"Contradiction score {conflict.contradiction_score:.2f}"
            )
            self.metadata.update(
                old_id,
                importance=old_record.importance * SUPERSEDED_IMPORTANCE_FACTOR,  # Drastically reduce importance
                metadata=old_metadata,
            )

        logger.info(
            "SUPERSEDE: Old memory %s superseded (importance reduced)",
            old_id,
        )

        return {
            "resolution": "supersede",
            "superseded_memory_id": old_id,
            "action": "Old memory importance reduced; new memory will be stored normally",
        }

    def _resolve_merge(
        self,
        conflict: ConflictResult,
        new_record: Optional[MemoryRecord],
        new_embedding: Optional[list],
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """
        MERGE: Combine conflicting memories into a unified version.

        Strategy: Append a temporal qualifier to the existing memory noting
        the contradiction and both perspectives.
        """
        old_id = conflict.existing_memory_id
        old_record = self.metadata.get(old_id)

        if not old_record:
            return self._resolve_flag(conflict)

        # Create merged content with temporal context
        merged_content = (
            f"{old_record.content} "
            f"[Updated: {conflict.new_content}]"
        )

        # Update the existing memory with merged content
        old_metadata = old_record.metadata.copy()
        old_metadata["merged_at"] = time.time()
        old_metadata["merge_source"] = "conflict_resolution"
        old_metadata["contradiction_score"] = conflict.contradiction_score

        self.metadata.update(
            old_id,
            content=merged_content,
            metadata=old_metadata,
        )

        # Update BM25 index with new content
        self.bm25.add(old_id, merged_content)

        # Refresh vector and graph indexes so merged content is immediately retrievable
        merged_user_id = user_id or old_record.metadata.get("user_id", "global_user")
        if self.embed_fn:
            merged_embedding = self.embed_fn(merged_content)
            self.vectors.upsert(
                memory_id=old_id,
                embedding=merged_embedding,
                metadata={
                    "content": merged_content[:VECTOR_CONTENT_PREVIEW_LIMIT],
                    "memory_type": old_record.memory_type.value,
                    "namespace": old_record.namespace,
                    "importance": old_record.importance,
                    "user_id": merged_user_id,
                },
            )
        self.graph.add_memory_node(
            old_id, merged_content[:GRAPH_SUMMARY_LIMIT],
            user_id=merged_user_id,
            namespace=old_record.namespace,
        )

        logger.info("MERGE: Memory %s merged with new content", old_id)

        return {
            "resolution": "merge",
            "merged_memory_id": old_id,
            "merged_content": merged_content[:GRAPH_SUMMARY_LIMIT],
            "action": "Existing memory updated with merged content; new memory will not be stored separately",
            "skip_new_storage": True,
        }

    def _resolve_keep_existing(
        self,
        conflict: ConflictResult,
    ) -> Dict[str, Any]:
        """
        KEEP_EXISTING: Discard the new memory, keep the existing one.
        """
        logger.info(
            "KEEP_EXISTING: Discarding new memory, keeping %s",
            conflict.existing_memory_id,
        )

        return {
            "resolution": "keep_existing",
            "kept_memory_id": conflict.existing_memory_id,
            "action": "New memory discarded; existing memory retained",
            "skip_new_storage": True,
        }

    def _resolve_flag(
        self,
        conflict: ConflictResult,
    ) -> Dict[str, Any]:
        """
        FLAG_FOR_REVIEW: Return conflict information to the caller.
        No automatic action is taken — the user/agent decides.
        """
        logger.info(
            "FLAG_FOR_REVIEW: Conflict between new content and memory %s "
            "(contradiction=%.2f)",
            conflict.existing_memory_id,
            conflict.contradiction_score,
        )

        return {
            "resolution": "flag_for_review",
            "conflict": conflict.model_dump(),
            "action": "Conflict flagged for review; new memory stored with conflict metadata",
            "skip_new_storage": False,
        }
