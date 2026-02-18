import pytest
import numpy as np
import asyncio
from unittest.mock import MagicMock, patch

# v3.6.2 Robust Test Suite
# Avoids complex imports that cause collection errors in heterogeneous environments

@pytest.mark.asyncio
async def test_colbert_logic_fix():
    """Verify ColBERT reranking logic (config and numpy usage)."""
    from muninn.retrieval.hybrid import HybridRetriever

    # Mock dependencies
    mock_vectors = MagicMock()
    mock_metadata = MagicMock()

    retriever = HybridRetriever(
        metadata_store=mock_metadata,
        vector_store=mock_vectors,
        graph_store=MagicMock(),
        bm25_index=MagicMock()
    )

    # Set up indexer mock
    mock_indexer = MagicMock()
    mock_indexer.encoder.is_available = True
    mock_indexer.encoder.encode.return_value = np.array([[0.1]*128])
    mock_indexer.collection_name = "test_colbert_tokens"
    retriever._colbert_indexer = mock_indexer

    # Mock the scorer instance directly (already created in __init__)
    mock_scorer = MagicMock()
    mock_scorer.maxsim_score.return_value = 0.9
    retriever._colbert_scorer = mock_scorer

    # Mock scroll response with realistic point data
    mock_point = MagicMock()
    mock_point.vector = [0.2] * 128
    mock_client = MagicMock()
    mock_client.scroll.return_value = ([mock_point], None)
    mock_vectors._get_client.return_value = mock_client

    with patch("muninn.retrieval.hybrid.get_flags") as mock_get_flags:
        flags = MagicMock()
        flags.is_enabled.side_effect = lambda x: x == "colbert"  # colbert on, plaid off
        mock_get_flags.return_value = flags

        from muninn.core.types import MemoryRecord
        candidates = [("mem1", 0.5)]
        record_map = {"mem1": MemoryRecord(id="mem1", content="test", user_id="u1", importance=1.0)}

        results = await retriever._colbert_rerank("query", candidates, record_map, limit=1)
        assert len(results) == 1
        assert results[0].score == 0.9

@pytest.mark.asyncio
async def test_integrity_user_scoping():
    """Verify that integrity audit enforces user scoping in semantic search."""
    from muninn.consolidation.daemon import ConsolidationDaemon
    from muninn.core.types import MemoryRecord

    daemon = ConsolidationDaemon(
        config=MagicMock(),
        metadata=MagicMock(),
        vectors=MagicMock(),
        graph=MagicMock(),
        bm25=MagicMock(),
    )
    daemon._conflict_detector = MagicMock()
    daemon._conflict_detector.is_available = True

    # Mock records with user_id in metadata (as stored in SQLite)
    record1 = MemoryRecord(
        id="rec1", content="record from user1", importance=1.0,
        metadata={"user_id": "user1"},
    )
    daemon.metadata.get_for_consolidation.return_value = [record1]
    daemon.vectors.get_vectors.return_value = {"rec1": [0.1]*128}

    # Mock search to capture filter
    daemon.vectors.search.return_value = []
    # Mock get_random for the fallback path
    daemon.metadata.get_random.return_value = []

    await daemon._phase_integrity()

    # Verify search was called with filters= (correct VectorStore.search API) for user1
    args, kwargs = daemon.vectors.search.call_args
    assert "filters" in kwargs
    assert kwargs["filters"] is not None
    # Check filter content string (since it's a Qdrant model)
    filter_repr = repr(kwargs["filters"])
    assert "user1" in filter_repr

def test_server_auth_token_enforcement():
    """Verify that sensitive endpoints in server.py have Depends(verify_token)."""
    from server import app

    sensitive_paths = ["/goal/set", "/profile/user/set", "/ingest", "/consolidation/run"]

    found_paths = []
    for route in app.routes:
        if hasattr(route, "path") and route.path in sensitive_paths:
            # Check for verify_token in dependencies
            has_auth = any("verify_token" in str(getattr(d, "dependency", d)) for d in route.dependencies)
            if has_auth:
                found_paths.append(route.path)

    for path in sensitive_paths:
        assert path in found_paths, f"Endpoint {path} is missing verify_token dependency"
