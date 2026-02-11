# Lazy imports — ConsolidationDaemon transitively pulls vector/graph stores
from muninn.consolidation.merge import merge_memories, find_merge_candidates
from muninn.consolidation.promote import find_promotion_candidates, promote_memory

__all__ = [
    "ConsolidationDaemon",
    "merge_memories", "find_merge_candidates",
    "find_promotion_candidates", "promote_memory",
]


def __getattr__(name):
    if name == "ConsolidationDaemon":
        from muninn.consolidation.daemon import ConsolidationDaemon
        return ConsolidationDaemon
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
