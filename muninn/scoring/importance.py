"""
Muninn Importance Scoring
-------------------------
Multi-factor importance calculation inspired by:
- Synaptic Tagging & Capture (neuroscience)
- Danger Theory (immune systems)
- PageRank (graph centrality)
- Minimum Description Length (information theory)
"""

import math
import time
import logging
from typing import Optional, List, Callable

from muninn.core.types import MemoryRecord, Provenance

logger = logging.getLogger("Muninn.Scoring")

# Default weights (tunable)
DEFAULT_WEIGHTS = {
    "recency": 0.25,
    "frequency": 0.15,
    "centrality": 0.20,
    "novelty": 0.25,
    "provenance": 0.15,
}

# Recency half-life in days
RECENCY_HALF_LIFE = 7.0

# Provenance weights (user-explicit memories are most valuable)
PROVENANCE_WEIGHTS = {
    Provenance.USER_EXPLICIT: 1.0,
    Provenance.ASSISTANT_CONFIRMED: 0.8,
    Provenance.AUTO_EXTRACTED: 0.5,
    Provenance.INGESTED: 0.3,
}


def calculate_recency(created_at: float, half_life_days: float = RECENCY_HALF_LIFE) -> float:
    """Exponential decay: half-life determines how fast memories fade."""
    age_days = (time.time() - created_at) / 86400.0
    if age_days < 0:
        return 1.0
    return math.exp(-0.693 * age_days / half_life_days)


def calculate_frequency(access_count: int, max_expected: int = 100) -> float:
    """Log-scaled access frequency, normalized to [0, 1]."""
    return math.log1p(access_count) / math.log1p(max_expected)


def calculate_novelty(similarity_to_existing: float) -> float:
    """
    Novelty = 1 - max_similarity.
    High novelty means the memory is unique; low means it's redundant.
    """
    return max(0.0, min(1.0, 1.0 - similarity_to_existing))


def calculate_provenance_weight(provenance: Provenance) -> float:
    """Weight based on how the memory was created."""
    return PROVENANCE_WEIGHTS.get(provenance, 0.5)


def calculate_importance(
    memory: MemoryRecord,
    max_similarity: float = 0.0,
    centrality: float = 0.0,
    weights: Optional[dict] = None,
) -> float:
    """
    Composite importance score combining multiple signals.

    Args:
        memory: The memory record to score
        max_similarity: Maximum cosine similarity to existing semantic memories
        centrality: Graph degree centrality for entities in this memory
        weights: Optional custom weights dict

    Returns:
        importance score in [0.0, 1.0]
    """
    w = weights or DEFAULT_WEIGHTS

    recency = calculate_recency(memory.created_at)
    frequency = calculate_frequency(memory.access_count)
    novelty = calculate_novelty(max_similarity)
    provenance = calculate_provenance_weight(memory.provenance)

    importance = (
        w["recency"] * recency +
        w["frequency"] * frequency +
        w["centrality"] * centrality +
        w["novelty"] * novelty +
        w["provenance"] * provenance
    )

    return min(1.0, max(0.0, importance))


def batch_update_importance(
    memories: List[MemoryRecord],
    get_centrality: Callable[[str], float],
    get_max_similarity: Callable[[str], float],
) -> List[tuple]:
    """
    Batch importance recalculation for consolidation cycles.

    Returns list of (memory_id, new_importance) tuples.
    """
    updates = []
    for mem in memories:
        centrality = get_centrality(mem.id)
        max_sim = get_max_similarity(mem.id)
        new_importance = calculate_importance(mem, max_similarity=max_sim, centrality=centrality)
        updates.append((mem.id, new_importance))
    return updates
