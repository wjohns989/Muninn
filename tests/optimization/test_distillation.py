"""
Tests for Distillation Daemon (Phase 25)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from muninn.optimization.distillation import DistillationDaemon

@pytest.fixture
def mock_memory():
    m = MagicMock()
    # Mock components
    m._extraction = MagicMock()
    m._extraction.client = AsyncMock()
    m.add = AsyncMock()
    m.update = AsyncMock()
    return m

@pytest.fixture
def daemon(mock_memory):
    d = DistillationDaemon(mock_memory)
    # Mock the clustering engine to avoid DB calls
    d.cluster_engine = MagicMock()
    d.cluster_engine.find_episodic_clusters = AsyncMock()
    return d

@pytest.mark.asyncio
async def test_run_cycle_success(daemon, mock_memory):
    # Setup: 1 cluster found
    cluster = {
        "id": "c1",
        "memory_ids": ["m1", "m2"],
        "memories": [{"content": "log1"}, {"content": "log2"}],
        "topic": "test topic",
        "namespace": "ns1",
        "project": "p1"
    }
    daemon.cluster_engine.find_episodic_clusters.return_value = [cluster]
    
    # Mock LLM response
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="Synthesized Manual"))]
    mock_memory._extraction.client.chat.completions.create.return_value = mock_completion
    
    # Run
    result = await daemon.run_cycle()
    
    assert result["success"] is True
    assert result["processed"] == 1
    
    # Verify semantic memory added
    mock_memory.add.assert_called_once()
    args = mock_memory.add.call_args
    assert args.kwargs["content"] == "Synthesized Manual"
    assert args.kwargs["metadata"]["source_cluster"] == "c1"
    assert args.kwargs["namespace"] == "ns1"
    
    # Verify archiving
    assert mock_memory.update.call_count == 2 # m1, m2
    
@pytest.mark.asyncio
async def test_run_cycle_no_clusters(daemon):
    daemon.cluster_engine.find_episodic_clusters.return_value = []
    result = await daemon.run_cycle()
    assert result["processed"] == 0
