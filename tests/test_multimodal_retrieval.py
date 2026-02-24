"""
Tests for Multimodal Retrieval Filtering (Phase 20).
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from muninn.core.types import MemoryRecord, MediaType, MemoryType, SearchResult
from muninn.retrieval.hybrid import HybridRetriever

@pytest.fixture
def mock_stores():
    return {
        "metadata": MagicMock(),
        "vector": MagicMock(),
        "graph": MagicMock(),
        "bm25": MagicMock(),
    }

@pytest.fixture
def hybrid_retriever(mock_stores):
    return HybridRetriever(
        metadata_store=mock_stores["metadata"],
        vector_store=mock_stores["vector"],
        graph_store=mock_stores["graph_store"] if "graph_store" in mock_stores else mock_stores["graph"],
        bm25_index=mock_stores["bm25"],
        embed_fn=AsyncMock(return_value=[0.1] * 768)
    )

@pytest.mark.asyncio
async def test_search_with_media_type_filter(hybrid_retriever, mock_stores):
    # Setup mock data
    mem_text = MemoryRecord(id="m1", content="Text memory", media_type=MediaType.TEXT)
    mem_image = MemoryRecord(id="m2", content="Image memory", media_type=MediaType.IMAGE)
    
    mock_stores["vector"].search.return_value = [("m1", 0.9), ("m2", 0.8)]
    mock_stores["metadata"].get_by_ids.return_value = [mem_text, mem_image]
    
    # Search for images only
    results = await hybrid_retriever.search(
        query="test",
        media_type="image",
        rerank=False
    )
    
    assert len(results) == 1
    assert results[0].memory.id == "m2"
    assert results[0].memory.media_type == MediaType.IMAGE

@pytest.mark.asyncio
async def test_search_bm25_media_type_filtering(hybrid_retriever, mock_stores):
    # Setup mock BM25 results
    mock_stores["bm25"].search.return_value = [("m1", 1.0), ("m2", 0.5)]
    
    mem_text = MemoryRecord(id="m1", content="Text memory", media_type=MediaType.TEXT)
    mem_image = MemoryRecord(id="m2", content="Image memory", media_type=MediaType.IMAGE)
    mock_stores["metadata"].get_by_ids.return_value = [mem_text, mem_image]
    
    # We need to mock other signals to return empty so BM25 dominates
    mock_stores["vector"].search.return_value = []
    mock_stores["graph"].find_related_memories.return_value = []
    mock_stores["metadata"].get_all.return_value = []
    
    # Search for text only
    results = await hybrid_retriever.search(
        query="test",
        media_type="text",
        rerank=False
    )
    
    assert len(results) == 1
    assert results[0].memory.id == "m1"
    assert results[0].memory.media_type == MediaType.TEXT
