"""
Muninn Memory Merge
-------------------
Near-duplicate detection and merging for episodic memories.

Inspired by Minimum Description Length (information theory):
- If two memories can be described more compactly as one,
  they should be merged.

Threshold: cosine_similarity > 0.92 indicates near-duplicate.
"""

import logging
import time
from typing import List, Tuple, Optional

from muninn.core.types import MemoryRecord, MemoryType

logger = logging.getLogger("Muninn.Consolidation.Merge")


async def find_merge_candidates(
    records: List[MemoryRecord],
    vector_search_fn,
    similarity_threshold: float = 0.92,
) -> List[Tuple[str, str, float]]:
    """
    Find pairs of near-duplicate memories above the similarity threshold.

    Args:
        records: List of episodic memory records to check.
        vector_search_fn: Function(vector_id) → List[(id, score)] to find similar vectors.
        similarity_threshold: Minimum cosine similarity to consider a merge.

    Returns:
        List of (memory_id_1, memory_id_2, similarity) tuples.
    """
    candidates = []
    seen_pairs = set()

    for record in records:
        if record.memory_type != MemoryType.EPISODIC:
            continue
        if record.vector_id is None:
            continue

        # Search for similar vectors with isolation parameters
        try:
            record_user_id = (record.metadata or {}).get("user_id")
            similar = vector_search_fn(
                record.vector_id,
                user_id=record_user_id,
                namespace=record.namespace
            )
            # v3.22.1 Fix: Support async search function results
            if hasattr(similar, "__await__"):
                similar = await similar
            for other_id, score in similar:
                if other_id == record.id:
                    continue
                if score < similarity_threshold:
                    continue

                # Ensure we don't double-count pairs
                pair_key = tuple(sorted([record.id, other_id]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                candidates.append((record.id, other_id, score))
        except Exception as e:
            logger.warning("Merge candidate search failed for %s: %s", record.id, e)

    logger.info("Found %d merge candidates (threshold=%.2f)", len(candidates), similarity_threshold)
    return candidates


def merge_memories(
    primary: MemoryRecord,
    secondary: MemoryRecord,
) -> MemoryRecord:
    """
    Merge two near-duplicate memories into one.

    Strategy:
    - Keep the higher-importance record as primary
    - Combine content (if meaningfully different)
    - Preserve the earliest created_at
    - Sum access counts
    - Take the higher importance score
    - Link secondary → primary via parent_id

    Args:
        primary: The primary (surviving) memory record.
        secondary: The secondary (to be absorbed) memory record.

    Returns:
        Updated primary record with merged information.
    """
    # Determine which is actually higher importance
    if secondary.importance > primary.importance:
        primary, secondary = secondary, primary

    # Combine content if they differ meaningfully
    if secondary.content.strip() not in primary.content:
        # Append unique information from secondary
        combined = f"{primary.content}\n---\n{secondary.content}"
        # Cap at reasonable length
        if len(combined) <= 2000:
            primary.content = combined

    # Merge temporal data
    primary.created_at = min(primary.created_at, secondary.created_at)
    primary.last_accessed = max(
        primary.last_accessed or 0,
        secondary.last_accessed or 0,
    ) or None

    # Combine access counts
    primary.access_count += secondary.access_count

    # Merge metadata
    for key, value in secondary.metadata.items():
        if key not in primary.metadata:
            primary.metadata[key] = value

    # Increment consolidation generation
    primary.consolidation_gen = max(primary.consolidation_gen, secondary.consolidation_gen) + 1
    primary.consolidated = True

    logger.debug("Merged memory %s ← %s (gen=%d)",
                 primary.id, secondary.id, primary.consolidation_gen)

    return primary
