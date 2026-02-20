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
    "novelty": 0.15,
    "provenance": 0.15,
    "retrieval": 0.10,
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
    """Log-scaled access frequency, normalized to [0, 1].

    Clamped so that access_count > max_expected never pushes the value above 1.0.
    Without the clamp, the raw ratio exceeds 1.0 for power-accessed memories and
    breaks the [0,1] component contract assumed by the weighted sum in
    calculate_importance().
    """
    return min(1.0, math.log1p(access_count) / math.log1p(max_expected))


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
    retrieval_utility: float = 0.0,
    weights: Optional[dict] = None,
) -> float:
    """
    Composite importance score combining multiple signals.

    Args:
        memory: The memory record to score
        max_similarity: Maximum cosine similarity to existing semantic memories
        centrality: Graph degree centrality for entities in this memory
        retrieval_utility: SNIPS feedback-derived retrieval utility [0.0, 1.0]
        weights: Optional custom weights dict

    Returns:
        importance score in [0.0, 1.0]
    """
    w = weights or DEFAULT_WEIGHTS

    recency = calculate_recency(memory.created_at)
    frequency = calculate_frequency(memory.access_count)
    novelty = calculate_novelty(max_similarity)
    provenance = calculate_provenance_weight(memory.provenance)

    # Use .get() with DEFAULT_WEIGHTS fallback so callers can supply partial
    # custom weight dicts without raising KeyError.
    importance = (
        w.get("recency", DEFAULT_WEIGHTS["recency"]) * recency
        + w.get("frequency", DEFAULT_WEIGHTS["frequency"]) * frequency
        + w.get("centrality", DEFAULT_WEIGHTS["centrality"]) * centrality
        + w.get("novelty", DEFAULT_WEIGHTS["novelty"]) * novelty
        + w.get("provenance", DEFAULT_WEIGHTS["provenance"]) * provenance
        + w.get("retrieval", DEFAULT_WEIGHTS["retrieval"]) * retrieval_utility
    )

    return min(1.0, max(0.0, importance))


def batch_update_importance(
    memories: List[MemoryRecord],
    get_centrality: Callable[[str], float],
    get_max_similarity: Callable[[str], float],
    get_retrieval_utility: Optional[Callable[[str], float]] = None,
) -> List[tuple]:
    """
    Batch importance recalculation for consolidation cycles.

    Returns list of (memory_id, new_importance) tuples.
    """
    updates = []
    for mem in memories:
        centrality = get_centrality(mem.id)
        max_sim = get_max_similarity(mem.id)
        ret_util = get_retrieval_utility(mem.id) if get_retrieval_utility else 0.0
        new_importance = calculate_importance(
            mem, 
            max_similarity=max_sim, 
            centrality=centrality,
            retrieval_utility=ret_util
        )
        updates.append((mem.id, new_importance))
    return updates
