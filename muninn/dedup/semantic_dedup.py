"""
Muninn Semantic Deduplication (v3.2.0)
---------------------------------------
Embedding-based near-duplicate detection at ingestion time.

Uses existing vector infrastructure (Qdrant cosine similarity) to detect
when a new memory is semantically equivalent to an existing one.

Threshold hierarchy:
  - 0.95+ : Near-duplicate (ingestion-time dedup)
  - 0.92  : Merge candidate (consolidation-time merge)
  - 0.80  : Similar but distinct (no action)

Strategies:
  - SKIP:            Discard new memory (near-identical content)
  - UPDATE_EXISTING: Merge new info into existing memory
  - LINK:            Store both, mark as semantically related

Dependencies: None — uses existing fastembed embeddings and Qdrant search
"""

import logging
from enum import Enum
from typing import Optional, List, Tuple, Dict, Any
from pydantic import BaseModel, Field

from muninn.core.types import MemoryRecord

logger = logging.getLogger("Muninn.Dedup")


class DedupStrategy(str, Enum):
    """Resolution strategy for detected duplicates."""
    SKIP = "skip"                       # Discard new, keep existing
    UPDATE_EXISTING = "update_existing"  # Merge new info into existing
    LINK = "link"                       # Store both, mark as related


class DedupResult(BaseModel):
    """Result of a deduplication check."""
    is_duplicate: bool = False
    existing_memory_id: Optional[str] = None
    similarity: float = 0.0
    strategy: DedupStrategy = DedupStrategy.SKIP
    explanation: str = ""


class SemanticDedup:
    """
    Embedding-based near-duplicate detection for memory ingestion.

    Checks whether a new memory's embedding is within `threshold` cosine
    similarity of any existing memory. If so, returns a DedupResult with
    the recommended strategy.

    The threshold (default 0.95) is intentionally higher than consolidation's
    merge_similarity (0.92) to avoid false positives at ingestion time while
    still catching true duplicates.
    """

    def __init__(
        self,
        threshold: float = 0.95,
        content_overlap_threshold: float = 0.8,
    ):
        """
        Args:
            threshold: Cosine similarity threshold for duplicate detection.
                       Must be in (0.0, 1.0]. Values below 0.90 risk false positives.
            content_overlap_threshold: Token-level overlap threshold for
                                       content comparison (0.0-1.0).
        """
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0.0, 1.0], got {threshold}")
        if not 0.0 <= content_overlap_threshold <= 1.0:
            raise ValueError(
                f"content_overlap_threshold must be in [0.0, 1.0], got {content_overlap_threshold}"
            )

        self.threshold = threshold
        self.content_overlap_threshold = content_overlap_threshold
        logger.info(
            "SemanticDedup initialized: threshold=%.3f, content_overlap=%.2f",
            threshold,
            content_overlap_threshold,
        )

    def check_duplicate(
        self,
        embedding: List[float],
        content: str,
        vector_store,
        metadata_store,
        exclude_ids: Optional[List[str]] = None,
    ) -> Optional[DedupResult]:
        """
        Check if an embedding is a near-duplicate of existing memories.

        Args:
            embedding: The new memory's embedding vector.
            content: The new memory's text content.
            vector_store: VectorStore instance for similarity search.
            metadata_store: SQLiteMetadataStore instance for content retrieval.
            exclude_ids: Memory IDs to exclude from comparison (e.g., self).

        Returns:
            DedupResult if duplicate found, None otherwise.
        """
        exclude_ids = set(exclude_ids or [])

        try:
            # Search for high-similarity candidates
            matches = vector_store.search(
                query_embedding=embedding,
                limit=5,
                score_threshold=self.threshold,
            )
        except Exception as e:
            logger.warning("Dedup vector search failed: %s", e)
            return None

        if not matches:
            return None

        for memory_id, score in matches:
            if memory_id in exclude_ids:
                continue

            if score < self.threshold:
                continue

            # Fetch existing record for content comparison
            existing = metadata_store.get(memory_id)
            if existing is None:
                continue

            # Compute token-level content overlap
            overlap = self._content_overlap(content, existing.content)

            if overlap >= self.content_overlap_threshold:
                # True duplicate — determine strategy
                strategy = self._determine_strategy(
                    new_content=content,
                    existing_record=existing,
                    similarity=score,
                    overlap=overlap,
                )

                return DedupResult(
                    is_duplicate=True,
                    existing_memory_id=memory_id,
                    similarity=score,
                    strategy=strategy,
                    explanation=(
                        f"Semantic duplicate detected: "
                        f"cosine={score:.3f}, token_overlap={overlap:.2f}. "
                        f"Existing memory: {existing.content[:80]}..."
                    ),
                )

        return None

    def _determine_strategy(
        self,
        new_content: str,
        existing_record: MemoryRecord,
        similarity: float,
        overlap: float,
    ) -> DedupStrategy:
        """
        Decide resolution strategy based on content analysis.

        Logic:
          - Near-identical (similarity >= 0.98 AND overlap >= 0.95): SKIP
          - New content has additional information: UPDATE_EXISTING
          - Otherwise: SKIP (default safe action)
        """
        if similarity >= 0.98 and overlap >= 0.95:
            # Almost identical — skip entirely
            return DedupStrategy.SKIP

        # Check if new content adds information
        new_tokens = set(new_content.lower().split())
        existing_tokens = set(existing_record.content.lower().split())
        novel_tokens = new_tokens - existing_tokens

        # If new content has >20% novel tokens, update existing
        if len(novel_tokens) > len(new_tokens) * 0.2:
            return DedupStrategy.UPDATE_EXISTING

        return DedupStrategy.SKIP

    @staticmethod
    def _content_overlap(content_a: str, content_b: str) -> float:
        """
        Compute token-level Jaccard similarity between two texts.

        Returns:
            Float in [0.0, 1.0] where 1.0 = identical token sets.
        """
        tokens_a = set(content_a.lower().split())
        tokens_b = set(content_b.lower().split())

        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b

        return len(intersection) / len(union) if union else 0.0

    def merge_content(
        self,
        new_content: str,
        existing_content: str,
    ) -> str:
        """
        Merge new content into existing content by appending novel sentences.

        Used when strategy is UPDATE_EXISTING to enrich the existing memory
        without losing the original content.

        Returns:
            Merged content string.
        """
        existing_sentences = set(
            s.strip()
            for s in existing_content.replace(".", ".\n").split("\n")
            if s.strip()
        )
        new_sentences = [
            s.strip()
            for s in new_content.replace(".", ".\n").split("\n")
            if s.strip()
        ]

        # Add sentences from new content that aren't in existing
        novel = []
        for sentence in new_sentences:
            if sentence not in existing_sentences:
                # Check fuzzy match (>80% token overlap = same sentence)
                is_duplicate = False
                for existing_s in existing_sentences:
                    if self._content_overlap(sentence, existing_s) > 0.8:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    novel.append(sentence)

        if not novel:
            return existing_content

        merged = existing_content.rstrip()
        if not merged.endswith("."):
            merged += "."
        merged += " " + ". ".join(novel)
        if not merged.endswith("."):
            merged += "."

        return merged
