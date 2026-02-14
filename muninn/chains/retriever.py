"""
Memory-chain retrieval expansion helpers.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


class MemoryChainRetriever:
    """
    Expands retrieval candidate sets with chain-neighbor memories.
    """

    def __init__(
        self,
        *,
        graph_store,
        max_seed_memories: int = 6,
    ):
        self.graph_store = graph_store
        self.max_seed_memories = max(1, int(max_seed_memories))

    def expand_from_ranked_results(
        self,
        ranked_results: Sequence[Tuple[str, float]],
        *,
        limit: int = 20,
    ) -> List[Tuple[str, float]]:
        if not ranked_results or self.graph_store is None:
            return []
        seed_ids: List[str] = []
        for memory_id, _score in ranked_results:
            if memory_id in seed_ids:
                continue
            seed_ids.append(memory_id)
            if len(seed_ids) >= self.max_seed_memories:
                break
        if not seed_ids:
            return []
        return self.graph_store.find_chain_related_memories(seed_ids, limit=limit)

