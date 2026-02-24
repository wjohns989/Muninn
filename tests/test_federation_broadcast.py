import pytest
import asyncio
from unittest.mock import MagicMock, patch
from muninn.core.memory import MuninnMemory
from muninn.core.config import MuninnConfig
from muninn.advanced.cross_agent import FederationManager

@pytest.fixture
def mock_memory():
    # Use a simpler mock without strict spec to allow private attribute access
    memory = MagicMock()
    memory.config = MuninnConfig.from_env()
    memory.config.federation.enabled = True
    memory.config.federation.peers = ["http://peer1:42069", "http://peer2:42069"]
    memory.config.federation.sync_on_add = True
    memory.config.server.auth_token = "test-token"
    return memory

@pytest.mark.asyncio
async def test_federation_broadcast_memory(mock_memory):
    fed_manager = FederationManager(mock_memory)
    
    # Mock record retrieval
    mock_record = MagicMock()
    mock_record.id = "mem1"
    mock_record.content = "test content"
    mock_record.metadata = {"user_id": "user1"}
    mock_record.media_type.value = "text"
    mock_record.memory_type.value = "episodic"
    mock_record.created_at = 123456789.0
    
    mock_memory._metadata.get_by_ids.return_value = [mock_record]
    
    # Mock bundle creation (tested elsewhere, but needed for flow)
    with patch.object(fed_manager, 'create_sync_bundle', return_value={"memories": []}) as mock_create:
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_post.return_value = MagicMock(status_code=200, json=lambda: {"success": True})
            
            result = await fed_manager.broadcast_memory("mem1")
            
            assert result["status"] == "completed"
            assert len(result["results"]) == 2
            assert "http://peer1:42069" in result["results"]
            assert "http://peer2:42069" in result["results"]
            
            # Verify bundle was created for the right user
            mock_create.assert_called_once_with(["mem1"], user_id="user1")
            
            # Verify HTTP calls were made with auth header
            assert mock_post.call_count == 2
            args, kwargs = mock_post.call_args
            assert kwargs["headers"]["Authorization"] == "Bearer test-token"

@pytest.mark.asyncio
async def test_federation_disabled(mock_memory):
    mock_memory.config.federation.enabled = False
    fed_manager = FederationManager(mock_memory)
    
    result = await fed_manager.broadcast_memory("mem1")
    assert result["status"] == "skipped"
    assert result["reason"] == "federation_disabled_or_no_peers"