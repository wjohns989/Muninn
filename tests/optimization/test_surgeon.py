"""
Tests for Memory Surgeon (Phase 25)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from muninn.optimization.surgeon import MemorySurgeon

@pytest.fixture
def mock_memory():
    m = MagicMock()
    m._metadata.get_by_ids = MagicMock()
    m.update = AsyncMock()
    return m

@pytest.mark.asyncio
async def test_correct_memory_success(mock_memory):
    surgeon = MemorySurgeon(mock_memory)
    
    # Mock existing record
    record = MagicMock()
    record.content = "Original Content"
    mock_memory._metadata.get_by_ids.return_value = [record]
    
    # Run
    success = await surgeon.correct_memory("mem-1", "This is wrong.")
    
    assert success is True
    mock_memory.update.assert_called_once()
    args = mock_memory.update.call_args
    assert args.args[0] == "mem-1"
    assert "Original Content" in args.kwargs["data"]
    assert "[CORRECTION]: This is wrong." in args.kwargs["data"]

@pytest.mark.asyncio
async def test_correct_memory_not_found(mock_memory):
    surgeon = MemorySurgeon(mock_memory)
    mock_memory._metadata.get_by_ids.return_value = []
    
    success = await surgeon.correct_memory("mem-1", "fix")
    assert success is False
    mock_memory.update.assert_not_called()
