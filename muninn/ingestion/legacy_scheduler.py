"""
Background legacy scan scheduler for Muninn.
Periodically runs discovery pass to find new AI data sources.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("Muninn.Ingestion.Legacy")


class LegacyDiscoveryScheduler:
    """
    Async scheduler for legacy source discovery scans.

    Maintains a persistent cache of discovered sources in SQLite to detect
    new arrivals since the last scan.
    """

    def __init__(self, memory: Any, interval_seconds: float = 3600.0):
        self._memory = memory
        self._interval = interval_seconds
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_run_at: Optional[float] = None
        self._last_sync_result: Optional[Dict[str, int]] = None

    @property
    def status(self) -> Dict[str, Any]:
        """Return runtime status of the scheduler."""
        stats = {}
        if self._memory and hasattr(self._memory, "_metadata") and self._memory._metadata:
            try:
                stats = self._memory._metadata.get_legacy_sources_stats()
            except Exception:
                pass

        return {
            "running": self._running,
            "last_run_at": self._last_run_at,
            "last_sync_result": self._last_sync_result,
            "interval_seconds": self._interval,
            "cache_stats": stats,
        }

    async def start(self) -> bool:
        """Start the periodic discovery loop."""
        if self._running:
            return False
        self._running = True
        self._task = asyncio.create_task(
            self._run_loop(), name="muninn-legacy-discovery"
        )
        logger.info(
            "Legacy discovery scheduler started (interval=%.1fs)", self._interval
        )
        return True

    async def stop(self) -> bool:
        """Stop the periodic discovery loop."""
        if not self._running:
            return False
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Legacy discovery scheduler stopped")
        return True

    async def trigger_once(self) -> Dict[str, Any]:
        """Run discovery scan immediately and update cache."""
        logger.info("Starting background legacy source scan...")
        t0 = time.time()
        try:
            # 1. Discover using the memory engine's existing logic
            discovery_result = await self._memory.discover_legacy_sources()
            sources = discovery_result.get("sources", [])

            # 2. Sync with persistent cache in the metadata store
            sync_result = await asyncio.to_thread(
                self._memory._metadata.sync_legacy_sources_cache, sources
            )

            self._last_run_at = time.time()
            self._last_sync_result = sync_result

            elapsed = self._last_run_at - t0
            new_count = sync_result.get("new", 0)
            if new_count > 0:
                logger.info(
                    "Legacy scan complete (%.2fs): found %d NEW sources (%d total)",
                    elapsed,
                    new_count,
                    sync_result.get("total", 0),
                )
            else:
                logger.info(
                    "Legacy scan complete (%.2fs): no new sources found (%d total)",
                    elapsed,
                    sync_result.get("total", 0),
                )

            return sync_result
        except Exception as e:
            logger.error("Legacy discovery scan failed: %s", e, exc_info=True)
            return {"success": False, "error": str(e)}

    async def _run_loop(self) -> None:
        """Internal loop for periodic execution."""
        # Initial scan on startup (after a short delay to let server settle)
        await asyncio.sleep(5.0)
        try:
            await self.trigger_once()
        except Exception:
            pass

        while self._running:
            try:
                await asyncio.sleep(self._interval)
                if not self._running:
                    break
                await self.trigger_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Legacy discovery scan loop error: %s", e)
                # Avoid tight loop on repeated failure
                await asyncio.sleep(60.0)