"""
Muninn Memory Promotion
-----------------------
Memory type promotion based on access patterns and stability.

Promotion rules inspired by Complementary Learning Systems (CLS):
- Episodic → Semantic: Frequently accessed memories crystallize into general knowledge
- Semantic → Procedural: Stable patterns become actionable procedures

Thresholds:
- Episodic → Semantic: access_count >= 5
- Semantic → Procedural: access_count >= 15 + consolidation_gen >= 2
"""

import logging
from typing import List, Tuple

from muninn.core.types import MemoryRecord, MemoryType

logger = logging.getLogger("Muninn.Consolidation.Promote")

# Promotion thresholds
EPISODIC_TO_SEMANTIC_ACCESS = 5
SEMANTIC_TO_PROCEDURAL_ACCESS = 15
SEMANTIC_TO_PROCEDURAL_GEN = 2


def find_promotion_candidates(records: List[MemoryRecord]) -> List[Tuple[str, MemoryType]]:
    """
    Identify memories eligible for type promotion.

    Returns:
        List of (memory_id, new_memory_type) tuples.
    """
    promotions = []

    for record in records:
        new_type = _check_promotion(record)
        if new_type and new_type != record.memory_type:
            promotions.append((record.id, new_type))

    logger.info("Found %d promotion candidates", len(promotions))
    return promotions


def _check_promotion(record: MemoryRecord) -> MemoryType:
    """
    Check if a single memory qualifies for promotion.

    Promotion flow:
    WORKING → (TTL expiry, handled by decay) → deleted
    EPISODIC → SEMANTIC when accessed frequently
    SEMANTIC → PROCEDURAL when stable and well-consolidated
    PROCEDURAL → (terminal state, no further promotion)
    """
    if record.memory_type == MemoryType.EPISODIC:
        if record.access_count >= EPISODIC_TO_SEMANTIC_ACCESS:
            return MemoryType.SEMANTIC

    elif record.memory_type == MemoryType.SEMANTIC:
        if (record.access_count >= SEMANTIC_TO_PROCEDURAL_ACCESS
                and record.consolidation_gen >= SEMANTIC_TO_PROCEDURAL_GEN):
            return MemoryType.PROCEDURAL

    return record.memory_type


def promote_memory(record: MemoryRecord, new_type: MemoryType) -> MemoryRecord:
    """
    Promote a memory to a higher type.

    Side effects:
    - Updates memory_type
    - Marks as consolidated
    - Increments consolidation generation
    - Boosts importance slightly (promoted memories are more valuable)
    """
    old_type = record.memory_type
    record.memory_type = new_type
    record.consolidated = True
    record.consolidation_gen += 1

    # Importance boost for promotion (capped at 1.0)
    promotion_boost = 0.05
    record.importance = min(1.0, record.importance + promotion_boost)

    logger.info("Promoted memory %s: %s -> %s (importance=%.3f)",
                record.id, old_type.value, new_type.value, record.importance)

    return record
