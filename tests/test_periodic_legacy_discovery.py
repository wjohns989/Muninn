"""
Tests for background legacy scan scheduling (Phase 19).
"""

import asyncio
import time
from unittest.mock import MagicMock, AsyncMock

import pytest
from muninn.ingestion.legacy_scheduler import LegacyDiscoveryScheduler


@pytest.mark.asyncio
async def test_legacy_scheduler_trigger_once():
    """Verify trigger_once calls memory engine and syncs with cache."""
    # Mock memory engine
    memory = MagicMock()
    memory.discover_legacy_sources = AsyncMock(return_value={
        "sources": [
            {"source_id": "src1", "provider": "p1", "path": "/path1", "category": "c1"},
            {"source_id": "src2", "provider": "p2", "path": "/path2", "category": "c2"},
        ]
    })
    
    # Mock metadata store
    memory._metadata = MagicMock()
    memory._metadata.sync_legacy_sources_cache.return_value = {
        "new": 1, "updated": 1, "total": 2
    }
    memory._metadata.get_legacy_sources_stats.return_value = {
        "total_cached": 2, "new_last_24h": 1, "active_non_ignored": 2
    }

    scheduler = LegacyDiscoveryScheduler(memory, interval_seconds=10.0)
    
    result = await scheduler.trigger_once()
    
    assert result["new"] == 1
    assert result["total"] == 2
    memory.discover_legacy_sources.assert_called_once()
    memory._metadata.sync_legacy_sources_cache.assert_called_once_with(
        memory.discover_legacy_sources.return_value["sources"]
    )
    
    status = scheduler.status
    assert status["running"] is False
    assert status["last_run_at"] is not None
    assert status["cache_stats"]["total_cached"] == 2


@pytest.mark.asyncio
async def test_legacy_scheduler_loop(monkeypatch):
    """Verify scheduler loop runs periodically."""
    # Speed up the initial startup sleep and the loop sleep
    real_sleep = asyncio.sleep
    async def mock_sleep(seconds):
        await real_sleep(min(seconds, 0.01))
    monkeypatch.setattr(asyncio, "sleep", mock_sleep)

    memory = MagicMock()
    memory.discover_legacy_sources = AsyncMock(return_value={"sources": []})
    memory._metadata = MagicMock()
    memory._metadata.sync_legacy_sources_cache.return_value = {"new": 0, "total": 0}
    
    # Fast loop for testing
    scheduler = LegacyDiscoveryScheduler(memory, interval_seconds=0.1)
    
    # Use monkeypatch to avoid actual long sleeps in _run_loop if needed,
    # but here we just want to see it start.
    
    await scheduler.start()
    assert scheduler.status["running"] is True
    
    # Let it run for a bit
    # Use real sleep to ensure the background task has time to execute
    await real_sleep(0.1)
    
    assert memory.discover_legacy_sources.call_count >= 1
    
    await scheduler.stop()
    assert scheduler.status["running"] is False


@pytest.mark.asyncio
async def test_legacy_scheduler_error_handling():
    """Verify scheduler handles exceptions gracefully."""
    memory = MagicMock()
    memory.discover_legacy_sources = AsyncMock(side_effect=RuntimeError("scan failed"))
    
    scheduler = LegacyDiscoveryScheduler(memory)
    result = await scheduler.trigger_once()
    
    assert result["success"] is False
    assert "scan failed" in result["error"]
    assert scheduler._last_run_at is None  # Not updated on failure