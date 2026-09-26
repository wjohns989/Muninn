"""Keeps local AI history safe and imported: vault sync on a timer, then incremental import.

- Vault sync runs at startup and every ``MUNINN_HISTORY_SYNC_MINUTES`` (default 30),
  well inside the shortest app retention window, so nothing an app cleans up is lost.
- Import is dry-run until you run it once with apply; after that, each sync also
  imports new turns (live threads keep flowing in, including what compaction drops).
  ``MUNINN_HISTORY_AUTO_IMPORT=1`` or ``0`` forces this on or off.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from muninn.history.importer import import_history, read_thread
from muninn.history.locations import history_homes, history_sources
from muninn.history.vault import HistoryVault

if TYPE_CHECKING:
    from muninn.core.memory import MuninnMemory

logger = logging.getLogger("Muninn.history")
AUTO_IMPORT_META = "history_auto_import"
# Stop fires after every reply; import a live thread at most this often from it.
CAPTURE_DEBOUNCE_SECONDS = 120.0


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _minutes() -> float:
    try:
        return max(1.0, float(os.environ.get("MUNINN_HISTORY_SYNC_MINUTES", "30")))
    except ValueError:
        return 30.0


class HistoryService:
    def __init__(self, memory: "MuninnMemory", vault_root: Path, home: Optional[Path] = None,
                 interval_minutes: Optional[float] = None):
        self.memory = memory
        self.home = home
        self.vault = HistoryVault(vault_root, home=home)
        self.interval = (interval_minutes or _minutes()) * 60.0
        self._task: Optional[asyncio.Task] = None
        self._job: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self.last_sync: Optional[Dict[str, Any]] = None
        self.last_import: Optional[Dict[str, Any]] = None
        self.last_analysis: Optional[Dict[str, Any]] = None
        self.progress: Dict[str, Any] = {}
        self._captured: Dict[str, float] = {}
        self._background: set = set()

    # --- settings ---------------------------------------------------------------

    def auto_import_enabled(self) -> bool:
        forced = os.environ.get("MUNINN_HISTORY_AUTO_IMPORT", "").strip().lower()
        if forced in ("1", "true", "yes", "on"):
            return True
        if forced in ("0", "false", "no", "off"):
            return False
        return self.memory._metadata.get_meta(AUTO_IMPORT_META) == "1"

    def retention_warnings(self) -> List[str]:
        warnings = []
        for source in (s for home in history_homes(self.home) for s in history_sources(home)):
            if not source.exists:
                continue
            days = source.retention.get("days")
            if source.provider == "claude_code" and isinstance(days, (int, float)) and days < 3650:
                warnings.append(
                    f"Claude Code deletes transcripts older than {days:g} days (cleanupPeriodDays). Muninn's vault "
                    f"keeps copies (synced every {self.interval / 60:g} min); raise cleanupPeriodDays in "
                    f"{source.home / 'settings.json'} to keep them in Claude Code as well."
                )
            if source.provider == "gemini_cli" and source.retention.get("value"):
                warnings.append("Gemini CLI session retention is on; the vault keeps copies of expired sessions.")
        return warnings

    # --- operations ---------------------------------------------------------------

    async def sync(self, extra_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        async with self._lock:
            report = await asyncio.to_thread(self.vault.sync, extra_paths)
        report["at"] = time.time()
        self.last_sync = report
        return report

    async def run_import(self, *, apply: bool, providers: Optional[List[str]] = None,
                         since: Optional[float] = None) -> Dict[str, Any]:
        async with self._lock:
            self.progress = {"running": True, "apply": apply, "started_at": time.time()}
            try:
                report = await import_history(self.memory, self.vault, apply=apply, providers=providers,
                                              since=since, progress=self.progress)
            finally:
                self.progress["running"] = False
        if apply:
            report["finished_at"] = time.time()
            self.last_import = report
            if not providers and not since:
                await asyncio.to_thread(self.memory._metadata.set_meta, AUTO_IMPORT_META, "1")
        return report

    async def run_analysis(self, *, apply: bool, **options: Any) -> Dict[str, Any]:
        from muninn.history.insights import analyze_threads

        if not apply:
            return await analyze_threads(self.memory, self.vault, apply=False, **options)
        self.progress = {"running": True, "analysis": True, "started_at": time.time()}
        try:
            report = await analyze_threads(self.memory, self.vault, apply=True, progress=self.progress, **options)
        finally:
            self.progress["running"] = False
        report["finished_at"] = time.time()
        self.last_analysis = report
        return report

    def start_analysis(self, **options: Any) -> bool:
        if self._job and not self._job.done():
            return False
        self._job = asyncio.create_task(self._guarded(self.run_analysis(apply=True, **options)))
        return True

    def start_import(self, *, providers: Optional[List[str]] = None, since: Optional[float] = None) -> bool:
        """Apply in the background (large histories take a while); poll status()."""
        if self._job and not self._job.done():
            return False
        self._job = asyncio.create_task(self._guarded(self.run_import(apply=True, providers=providers, since=since)))
        return True

    async def _guarded(self, coro) -> None:
        try:
            await coro
        except Exception as exc:
            logger.exception("History import failed")
            self.last_import = {"error": str(exc), "finished_at": time.time()}

    async def capture(self, path: str, provider: str, *, force: bool = False) -> Dict[str, Any]:
        """Vault and import one transcript now (called by agent hooks)."""
        key = str(Path(path).resolve())
        now = time.time()
        if not force and now - self._captured.get(key, 0.0) < CAPTURE_DEBOUNCE_SECONDS:
            return {"skipped": "debounced"}
        self._captured[key] = now
        async with self._lock:
            outcome = await asyncio.to_thread(self.vault.capture, Path(path), provider)
            if outcome == "missing":
                return {"captured": False}
            report = await import_history(self.memory, self.vault, apply=True, sources=[key])
        return {"captured": True, "vault": outcome, "turn_memories": report["turn_memories"],
                "compaction_memories": report["compaction_memories"]}

    def capture_later(self, path: str, provider: str, *, force: bool = False) -> None:
        """Hooks must answer fast (Codex allows 1 s at session end): capture in the background."""
        task = asyncio.create_task(self._guarded(self.capture(path, provider, force=force)))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def thread(self, thread_key: str, offset: int = 0, limit: int = 50) -> Dict[str, Any]:
        return await read_thread(self.memory, thread_key, offset, limit)

    def status(self) -> Dict[str, Any]:
        sources = [
            {"provider": s.provider, "home": str(s.home), "found": s.exists, "relocated_by": s.relocated_by,
             "retention": s.retention}
            for home in history_homes(self.home) for s in history_sources(home)
        ]
        return {
            "vault": self.vault.status(),
            "sources": sources,
            "sync_interval_minutes": self.interval / 60,
            "auto_import": self.auto_import_enabled(),
            "last_sync": self.last_sync,
            "last_import": self.last_import,
            "last_analysis": self.last_analysis,
            "auto_analyze": _flag("MUNINN_INSIGHTS_AUTO"),
            "import_progress": self.progress,
            "warnings": self.retention_warnings(),
        }

    # --- background loop ------------------------------------------------------------

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        for task in (self._task, self._job, *self._background):
            if task and not task.done():
                task.cancel()
        self._task = None
        self.vault.close()

    async def _loop(self) -> None:
        while True:
            try:
                await self.sync()
                if self.auto_import_enabled() and not (self._job and not self._job.done()):
                    await self.run_import(apply=True)
                    if _flag("MUNINN_INSIGHTS_AUTO"):
                        await self.run_analysis(apply=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("History sync failed")
            await asyncio.sleep(self.interval)
