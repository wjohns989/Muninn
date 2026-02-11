# Lazy imports — VectorStore needs qdrant_client, GraphStore needs kuzu
from muninn.store.sqlite_metadata import SQLiteMetadataStore

__all__ = ["SQLiteMetadataStore", "VectorStore", "GraphStore"]


def __getattr__(name):
    if name == "VectorStore":
        from muninn.store.vector_store import VectorStore
        return VectorStore
    if name == "GraphStore":
        from muninn.store.graph_store import GraphStore
        return GraphStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
