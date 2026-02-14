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
    assert results == [("mem-1", 0.99), ("mem-2", 0.95)]


def test_graph_search_passes_entity_list_and_scores_results():
    retriever = _retriever()
    retriever.graph.find_related_memories.return_value = ["mem-1", "mem-2"]

    results = retriever._graph_search("python api memory", limit=3)

    # Ensure graph store receives entity list (not a raw string)
    first_call = retriever.graph.find_related_memories.call_args_list[0]
    assert isinstance(first_call.args[0], list)
    assert first_call.args[0] == ["python"]
    assert all(isinstance(score, float) for _, score in results)


def test_goal_search_calls_vector_store_with_goal_embedding():
    retriever = _retriever()
    retriever.vectors.search.return_value = [("mem-9", 0.77)]

    results = retriever._goal_search(
        goal_embedding=[0.9, 0.1],
        limit=2,
        filters={"project": "muninn_mcp", "namespace": "global", "user_id": "global_user"},
    )

    retriever.vectors.search.assert_called_once_with(
        query_embedding=[0.9, 0.1],
        limit=2,
        filters={"project": "muninn_mcp", "namespace": "global", "user_id": "global_user"},
    )
    assert results == [("mem-9", 0.77)]


def test_rrf_includes_goal_signal_weight():
    retriever = _retriever()

    scores = retriever._rrf_fusion(
        vector_results=[],
        graph_results=[],
        bm25_results=[],
        goal_results=[("mem-1", 0.91)],
        temporal_results=[],
        goal_signal_weight=0.5,
    )

    assert "mem-1" in scores
    assert scores["mem-1"] > 0
