import pytest
import numpy as np
import asyncio
from unittest.mock import MagicMock, patch

# v3.6.2 Robust Test Suite
# Avoids complex imports that cause collection errors in heterogeneous environments

@pytest.mark.asyncio
async def test_colbert_logic_fix():
    """Verify ColBERT reranking logic (config and numpy usage)."""
    # Create a local mock for HybridRetriever to test the specific logic
    from muninn.retrieval.hybrid import HybridRetriever
    
    # Mock dependencies
    mock_vectors = MagicMock()
    mock_metadata = MagicMock()
    mock_config = MagicMock()
    
    # Combined fix for initialization errors
    retriever = HybridRetriever(
        metadata_store=mock_metadata, 
        vector_store=mock_vectors, 
        graph_store=MagicMock(), 
        bm25_index=MagicMock()
    )
    
    # ... existing indexer mock ...
    mock_indexer = MagicMock()
    mock_indexer.encoder.is_available = True
    mock_indexer.encoder.encode.return_value = np.array([[0.1]*128])
    retriever._colbert_indexer = mock_indexer
    
    # Mock Scorer and get_flags
    with patch("muninn.retrieval.hybrid.ColBERTScorer") as mock_scorer_cls, \
         patch("muninn.retrieval.hybrid.get_flags") as mock_get_flags:
        
        mock_scorer = mock_scorer_cls.return_value
        mock_scorer.maxsim_score.return_value = 0.9
        
        flags = MagicMock()
        flags.is_enabled.side_effect = lambda x: True
        mock_get_flags.return_value = flags
        
        # Test rerank
        from muninn.core.types import MemoryRecord
        candidates = [("mem1", 0.5)]
        record_map = {"mem1": MemoryRecord(id="mem1", content="test", user_id="u1", importance=1.0)}
        
        # This will trigger the fixed _colbert_rerank logic
        results = await retriever._colbert_rerank("query", candidates, record_map, limit=1)
        assert len(results) == 1
        assert results[0].score == 0.9

@pytest.mark.asyncio
async def test_integrity_user_scoping():
    """Verify that integrity audit enforces user scoping in semantic search."""
    # Mock ConsolidationDaemon to avoid heavy init
    from muninn.consolidation.daemon import ConsolidationDaemon
    from muninn.core.types import MemoryRecord
    
    daemon = ConsolidationDaemon(
        metadata_store=MagicMock(), 
        vector_store=MagicMock(),
        conflict_detector=MagicMock(),
        conflict_resolver=MagicMock(),
        config=MagicMock()
    )
    daemon._conflict_detector.is_available = True
    
    # Mock records with different users
    record1 = MemoryRecord(id="rec1", content="record from user1", user_id="user1", importance=1.0)
    daemon.metadata.get_for_consolidation.return_value = [record1]
    daemon.vectors.get_vectors.return_value = {"rec1": [0.1]*128}
    
    # Mock search to capture filter
    daemon.vectors.search.return_value = []
    
    await daemon._phase_integrity()
    
    # Verify search was called with a filter for user1
    args, kwargs = daemon.vectors.search.call_args
    assert "filter" in kwargs
    assert kwargs["filter"] is not None
    # Check filter content string (since it's a Qdrant model)
    filter_repr = repr(kwargs["filter"])
    assert "user1" in filter_repr

def test_server_auth_token_enforcement():
    """Verify that sensitive endpoints in server.py have Depends(verify_token)."""
    # Import app locally to avoid global side effects if possible
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
