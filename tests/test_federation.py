"""
Tests for Cross-Agent Federation logic.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from muninn.advanced.cross_agent import FederationManager
from muninn.core.types import MemoryRecord, MemoryType, Provenance

@pytest.fixture
def mock_memory():
    m = AsyncMock()
    m._metadata = MagicMock()
    return m

@pytest.mark.asyncio
async def test_manifest_generation(mock_memory):
    # Setup mock records
    r1 = MemoryRecord(id="m1", content="Memory 1", created_at=100.0)
    r2 = MemoryRecord(id="m2", content="Memory 2", created_at=200.0)
    # FederationManager queries the underlying metadata store directly
    # (see generate_manifest), so ensure the mock object's internal
    # ``_metadata.get_all`` returns our records.
    mock_memory._metadata.get_all.return_value = [r1, r2]
    
    fed = FederationManager(mock_memory)
    manifest = await fed.generate_manifest(project="test_proj")
    
    assert manifest["count"] == 2
    assert len(manifest["ids"]) == 2
    assert manifest["ids"][0][0] == "m1"
    assert manifest["ids"][1][0] == "m2"

@pytest.mark.asyncio
async def test_delta_calculation(mock_memory):
    fed = FederationManager(mock_memory)
    
    # Local has m1, m2
    local = {
        "ids": [("m1", "hash1"), ("m2", "hash2")]
    }
    
    # Remote has m2, m3
    remote = {
        "ids": [("m2", "hash2"), ("m3", "hash3")]
    }
    
    delta = await fed.calculate_delta(local, remote)
    
    # Local is missing m3 (present in remote but not local)
    assert "m3" in delta["missing"]
    assert "m2" not in delta["missing"]
    
    # Local offers m1 (present in local but not remote)
    assert "m1" in delta["offer"]

@pytest.mark.asyncio
async def test_apply_bundle(mock_memory):
    fed = FederationManager(mock_memory)
    
    bundle = {
        "memories": [
            {"id": "new1", "content": "New content", "metadata": {"user_id": "u1"}}
        ]
    }
    
    applied = await fed.apply_sync_bundle(bundle)
    
    assert applied == 1
    mock_memory.add.assert_called_once()
    call_kwargs = mock_memory.add.call_args.kwargs
    assert call_kwargs["content"] == "New content"
    assert call_kwargs["provenance"] == "federated_sync"
