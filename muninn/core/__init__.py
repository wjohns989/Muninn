# Lazy imports to avoid pulling heavy dependencies on simple type imports
from muninn.core.types import MemoryRecord, MemoryType, SearchResult

__all__ = ["MuninnMemory", "MemoryRecord", "MemoryType", "SearchResult"]


def __getattr__(name):
    if name == "MuninnMemory":
        from muninn.core.memory import MuninnMemory
        return MuninnMemory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
