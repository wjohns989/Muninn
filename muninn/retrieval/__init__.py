# Lazy imports — HybridRetriever pulls in qdrant_client transitively
from muninn.retrieval.bm25 import BM25Index
from muninn.retrieval.reranker import Reranker

__all__ = ["HybridRetriever", "BM25Index", "Reranker"]


def __getattr__(name):
    if name == "HybridRetriever":
        from muninn.retrieval.hybrid import HybridRetriever
        return HybridRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
