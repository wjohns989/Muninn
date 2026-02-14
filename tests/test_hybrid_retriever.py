from unittest.mock import MagicMock

from muninn.retrieval.hybrid import HybridRetriever


def _retriever():
    return HybridRetriever(
        metadata_store=MagicMock(),
        vector_store=MagicMock(),
        graph_store=MagicMock(),
        bm25_index=MagicMock(),
        reranker=None,
        embed_fn=lambda q: [0.1, 0.2],
    )


def test_vector_search_calls_vector_store_with_supported_signature():
    retriever = _retriever()
    retriever.vectors.search.return_value = [("mem-1", 0.99), ("mem-2", 0.95)]

    results = retriever._vector_search(
        query_embedding=[0.2, 0.3],
        limit=3,
        filters={"namespace": "project-a", "user_id": "user-1"},
    )

    retriever.vectors.search.assert_called_once_with(
        query_embedding=[0.2, 0.3],
        limit=3,
        filters={"namespace": "project-a", "user_id": "user-1"},
    )
    assert results == [("mem-1", 0), ("mem-2", 1)]
