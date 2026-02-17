import pytest
import time
import asyncio
from unittest.mock import MagicMock
from muninn.retrieval.hybrid import HybridRetriever
from muninn.core.types import MemoryRecord, MemoryType, Provenance

@pytest.mark.asyncio
async def test_hybrid_importance_weighting_performance():
    # Mock metadata store to count queries
    metadata = MagicMock()
    metadata.get_by_ids.side_effect = lambda ids: [
        MemoryRecord(id=mid, content="test", importance=0.8) for mid in ids
    ]
    
    retriever = HybridRetriever(
        metadata_store=metadata,
        vector_store=MagicMock(),
        graph_store=MagicMock(),
        bm25_index=MagicMock()
    )
    
    # 100 results to trigger weighting
    rrf_scores = {f"mem-{i}": 0.1 for i in range(100)}
    
    start_time = time.time()
    weighted = retriever._apply_importance_weighting(rrf_scores)
    end_time = time.time()
    
    # Verify exactly ONE call to get_by_ids instead of 100 calls to get
    assert metadata.get_by_ids.call_count == 1
    assert len(weighted) == 100
    print(f"Performance test PASSED: 1 query for 100 records in {end_time - start_time:.4f}s")

if __name__ == "__main__":
    asyncio.run(test_hybrid_importance_weighting_performance())