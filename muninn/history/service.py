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
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from muninn.history.importer import import_history, read_thread
from muninn.history.locations import app_data_dirs, export_candidates, history_homes, history_sources
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.vault import HistoryVault, require_legacy_history_disabled, strict_history_mode

if TYPE_CHECKING:
    from muninn.core.memory import MuninnMemory

logger = logging.getLogger("Muninn.history")
AUTO_IMPORT_META = "history_auto_import"
# Stop fires after every reply; import a live thread at most this often from it.
CAPTURE_DEBOUNCE_SECONDS = 120.0
STRICT_CAPTURE_DEBOUNCE_SECONDS = 600.0
_CLAUDE_SESSION_NAME = re.compile(
    r"^(?:[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}|agent-[0-9a-f-]+)\.jsonl$", re.I)
_CODEX_ROLLOUT_NAME = re.compile(
    r"^rollout-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-"
    r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
    r"(?:_[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})?\.jsonl(?:\.zst)?$", re.I)


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _minutes() -> float:
    try:
        return max(1.0, float(os.environ.get("MUNINN_HISTORY_SYNC_MINUTES", "30")))
    except ValueError:
        return 30.0


class HistoryService:
    def __init__(self, memory: "MuninnMemory", vault_root: Path, home: Optional[Path] = None,
                 interval_minutes: Optional[float] = None,
                 secure_archive_root: Optional[Path] = None):
        self.memory = memory
        self.home = home
        self.vault = None if strict_history_mode() else HistoryVault(vault_root, home=home, allow_plaintext=True)
        self.secure_archive_root = Path(secure_archive_root or os.environ.get("MUNINN_HISTORY_ARCHIVE_DIR")
                                        or (Path(vault_root).parent / "history_secure_archive"))
        self.secure_archive: Optional[SecureHistoryArchive] = None
        self.secure_archive_error: Optional[str] = None
        if strict_history_mode():
            self._open_secure_archive()
        self.interval = (interval_minutes or _minutes()) * 60.0
        self._task: Optional[asyncio.Task] = None
        self._job: Optional[asyncio.Task] = None
        self._vault_lock = asyncio.Lock()
        self._import_lock = asyncio.Lock()
        self._run_lock = asyncio.Lock()
        self.last_sync: Optional[Dict[str, Any]] = None
        self.last_import: Optional[Dict[str, Any]] = None
        self.last_analysis: Optional[Dict[str, Any]] = None
        self.progress: Dict[str, Any] = {}
        self._captured: Dict[str, float] = {}
        self._background: set = set()
        self._auto_task: Optional[asyncio.Task] = None
        self._analysis_lock = asyncio.Lock()
        self.last_auto_route: Optional[Dict[str, Any]] = None

    # --- settings ---------------------------------------------------------------

    def _open_secure_archive(self) -> None:
        if self.secure_archive is not None:
            return
        if not self.secure_archive_root.exists():
            self.secure_archive_error = "Encrypted archive not initialized"
            return
        try:
            self.secure_archive = SecureHistoryArchive(self.secure_archive_root)
            self.secure_archive_error = None
        except Exception as exc:
            self.secure_archive_error = f"Encrypted archive unavailable: {type(exc).__name__}"
            logger.error(self.secure_archive_error)

    def _require_secure_archive(self) -> SecureHistoryArchive:
        self._open_secure_archive()
        if self.secure_archive is None:
            raise RuntimeError("strict history mode requires an initialized, unlocked encrypted archive")
        return self.secure_archive

    def _validate_capture_source(self, path: str, provider: str) -> Path:
        if provider not in ("codex", "claude_code"):
            raise ValueError("Strict hook capture only accepts configured chat providers")
        source_path = Path(path).resolve(strict=True)
        for home in history_homes(self.home):
            for configured in history_sources(home):
                if configured.provider != provider or not source_path.is_relative_to(configured.home.resolve()):
                    continue
                relative = source_path.relative_to(configured.home.resolve())
                parts = relative.parts
                if provider == "claude_code" and (
                    (len(parts) >= 3 and parts[0] == "projects"
                     and _CLAUDE_SESSION_NAME.fullmatch(source_path.name) is not None)
                    or parts == ("history.jsonl",)
                ):
                    return source_path
                if provider == "codex" and (
                    (((len(parts) >= 3 and parts[0] == "sessions")
                      or (len(parts) >= 2 and parts[0] == "archived_sessions"))
                     and _CODEX_ROLLOUT_NAME.fullmatch(source_path.name) is not None)
                    or parts == ("history.jsonl",)
                ):
                    return source_path
        raise ValueError("Transcript is outside configured provider history roots")

    def _validate_sync_source(self, path: Path, provider: str, kind: str, home: Path) -> Path:
        resolved = path.resolve(strict=True)
        if provider in ("codex", "claude_code"):
            validated = self._validate_capture_source(str(resolved), provider)
            relative_name = validated.name
            if kind == "prompt_history" and relative_name == "history.jsonl":
                return validated
            if kind == "transcript" and relative_name != "history.jsonl":
                return validated
        elif provider == "gemini_cli":
            for configured in history_sources(home):
                if configured.provider != provider or not resolved.is_relative_to(configured.home.resolve()):
                    continue
                parts = resolved.relative_to(configured.home.resolve()).parts
                if (kind == "transcript" and len(parts) == 4 and parts[0] == "tmp"
                        and parts[2] == "chats" and resolved.suffix in (".json", ".jsonl")):
                    return resolved
        elif provider == "claude_desktop" and kind == "desktop_session":
            for directory in app_data_dirs(home):
                sessions = (directory / "Claude" / "claude-code-sessions").resolve()
                if resolved.is_relative_to(sessions) and resolved.suffix == ".json":
                    return resolved
        elif provider == "export" and kind == "export":
            for folder in (home / "Downloads", home / "Documents"):
                if not resolved.is_relative_to(folder.resolve()):
                    continue
                parts = resolved.relative_to(folder.resolve()).parts
                if (len(parts) == 1 and (resolved.name == "conversations.json" or resolved.suffix == ".zip")):
                    return resolved
                if len(parts) == 2 and parts[-1] == "conversations.json":
                    return resolved
        raise ValueError("Secure history candidate is outside the provider allowlist")

    def auto_import_enabled(self) -> bool:
        if strict_history_mode():
            return False
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
        require_legacy_history_disabled()
        async with self._vault_lock:
            report = await asyncio.to_thread(self.vault.sync, extra_paths)
        report["at"] = time.time()
        self.last_sync = report
        return report

    async def secure_sync(self, *, dry_run: bool = False) -> Dict[str, Any]:
        """Manual, copy-only encrypted history sync; never indexes or invokes a model."""
        if not strict_history_mode():
            raise RuntimeError("Secure sync requires strict history mode")
        archive = self._require_secure_archive()
        items: List[tuple[Path, str, str]] = []
        skipped_state_db = 0
        seen: set[Path] = set()
        for home in history_homes(self.home):
            for source in history_sources(home):
                if not source.exists:
                    continue
                for item in HistoryVault._source_files(source):
                    path = item["path"].resolve()
                    if path in seen:
                        continue
                    seen.add(path)
                    if item["kind"] == "state_db":
                        # Raw live SQLite files omit WAL state. An encrypted online
                        # snapshot path must be built before these can be captured.
                        skipped_state_db += 1
                        continue
                    validated = self._validate_sync_source(path, item["provider"], item["kind"], home)
                    items.append((validated, item["provider"], item["kind"]))
            for directory in app_data_dirs(home):
                sessions = directory / "Claude" / "claude-code-sessions"
                if sessions.is_dir():
                    for path in sessions.glob("**/*.json"):
                        resolved = path.resolve()
                        if resolved not in seen and path.is_file():
                            seen.add(resolved)
                            validated = self._validate_sync_source(resolved, "claude_desktop", "desktop_session", home)
                            items.append((validated, "claude_desktop", "desktop_session"))
            for path in export_candidates(home):
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    validated = self._validate_sync_source(resolved, "export", "export", home)
                    items.append((validated, "export", "export"))
        if dry_run:
            by_provider: Dict[str, int] = {}
            for _path, provider, _kind in items:
                by_provider[provider] = by_provider.get(provider, 0) + 1
            return {"apply": False, "discovered": len(items),
                    "source_bytes": sum(path.stat().st_size for path, _, _ in items),
                    "by_provider": by_provider, "skipped_live_state_db": skipped_state_db,
                    "indexing": "none; encrypted copy only"}
        async with self._vault_lock:
            report = await asyncio.to_thread(archive.archive_many, items)
        report["discovered"] = len(items)
        report["skipped_live_state_db"] = skipped_state_db
        report["indexing"] = "none; encrypted copy only"
        report["at"] = time.time()
        self.last_sync = report
        return report

    async def run_import(self, *, apply: bool, providers: Optional[List[str]] = None,
                         since: Optional[float] = None) -> Dict[str, Any]:
        require_legacy_history_disabled()
        async with self._run_lock:
            # Snapshot the manifest quickly. Vault copies are atomically
            # replaced, so sync/capture can proceed during a long backfill.
            async with self._vault_lock:
                vault_files = await asyncio.to_thread(self.vault.files)
            self.progress = {"running": True, "apply": apply, "started_at": time.time()}
            try:
                report = await import_history(self.memory, self.vault, apply=apply, providers=providers,
                                              since=since, progress=self.progress,
                                              vault_files=vault_files, unit_lock=self._import_lock)
            finally:
                self.progress["running"] = False
        if apply:
            report["finished_at"] = time.time()
            self.last_import = report
            if not providers and not since:
                await asyncio.to_thread(self.memory._metadata.set_meta, AUTO_IMPORT_META, "1")
        return report

    async def run_analysis(self, *, apply: bool, **options: Any) -> Dict[str, Any]:
        require_legacy_history_disabled()
        from muninn.history.insights import analyze_threads

        if not apply:
            return await analyze_threads(self.memory, self.vault, apply=False, **options)
        async with self._analysis_lock:
            self.progress = {"running": True, "analysis": True, "started_at": time.time()}
            try:
                report = await analyze_threads(self.memory, self.vault, apply=True, progress=self.progress, **options)
            finally:
                self.progress["running"] = False
            report["finished_at"] = time.time()
            self.last_analysis = report
            return report

    def start_analysis(self, **options: Any) -> bool:
        require_legacy_history_disabled()
        if self._job and not self._job.done():
            return False
        self._job = asyncio.create_task(self._guarded(self.run_analysis(apply=True, **options),
                                                       operation="analysis"))
        return True

    def start_import(self, *, providers: Optional[List[str]] = None, since: Optional[float] = None) -> bool:
        """Apply in the background (large histories take a while); poll status()."""
        require_legacy_history_disabled()
        if self._job and not self._job.done():
            return False
        self._job = asyncio.create_task(self._guarded(self.run_import(apply=True, providers=providers, since=since)))
        return True

    async def _guarded(self, coro, *, operation: str = "import") -> None:
        try:
            await coro
        except Exception as exc:
            logger.exception("History %s failed", operation)
            if operation == "import":
                self.last_import = {"error": str(exc), "finished_at": time.time()}
            elif operation == "analysis":
                self.last_analysis = {"error": str(exc), "finished_at": time.time()}

    async def capture(self, path: str, provider: str, *, force: bool = False) -> Dict[str, Any]:
        """Vault and import one transcript now (called by agent hooks)."""
        if strict_history_mode():
            archive = self._require_secure_archive()
            source_path = self._validate_capture_source(path, provider)
            key = str(source_path)
            async with self._vault_lock:
                now = time.time()
                if not force and now - self._captured.get(key, 0.0) < STRICT_CAPTURE_DEBOUNCE_SECONDS:
                    return {"skipped": "debounced"}
                outcome = await asyncio.to_thread(archive.archive_file, source_path, provider)
                self._captured[key] = time.time()
            return {"captured": outcome["status"] == "captured", "archive": outcome,
                    "indexing": "metadata-only projector pending"}
        require_legacy_history_disabled()
        key = str(Path(path).resolve())
        now = time.time()
        if not force and now - self._captured.get(key, 0.0) < CAPTURE_DEBOUNCE_SECONDS:
            return {"skipped": "debounced"}
        self._captured[key] = now
        async with self._vault_lock:
            outcome = await asyncio.to_thread(self.vault.capture, Path(path), provider)
            if outcome == "missing":
                return {"captured": False}
            vault_files = await asyncio.to_thread(self.vault.files)
        report = await import_history(self.memory, self.vault, apply=True, sources=[key],
                                      vault_files=vault_files, unit_lock=self._import_lock)
        self._launch_auto_analysis()
        return {"captured": True, "vault": outcome, "turn_memories": report["turn_memories"],
                "compaction_memories": report["compaction_memories"]}

    def capture_later(self, path: str, provider: str, *, force: bool = False) -> None:
        """Hooks must answer fast (Codex allows 1 s at session end): capture in the background."""
        if strict_history_mode():
            self._open_secure_archive()
            if self.secure_archive is None:
                logger.warning("Strict history mode: encrypted capture is unavailable until archive initialization")
                return
        task = asyncio.create_task(self._guarded(self.capture(path, provider, force=force),
                                                operation="capture"))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def thread(self, thread_key: str, offset: int = 0, limit: int = 50) -> Dict[str, Any]:
        require_legacy_history_disabled()
        return await read_thread(self.memory, thread_key, offset, limit)

    def secure_catalog(self, *, provider: Optional[str] = None, offset: int = 0,
                       limit: int = 100) -> List[Dict[str, Any]]:
        if not strict_history_mode():
            raise RuntimeError("Secure metadata catalog requires strict history mode")
        archive = self._require_secure_archive()
        return [item.as_dict() for item in archive.metadata_catalog(
            provider=provider, offset=offset, limit=limit)]

    def status(self) -> Dict[str, Any]:
        strict = strict_history_mode()
        if strict:
            self._open_secure_archive()
        archive_status = None
        archive_ready = self.secure_archive is not None
        archive_error = self.secure_archive_error
        if archive_ready:
            try:
                archive_status = self.secure_archive.status()
            except (VaultIntegrityError, OSError, PermissionError):
                archive_ready = False
                archive_error = "Encrypted archive integrity unavailable"
        sources = [
            ({"provider": s.provider, "found": s.exists, "retention": s.retention}
             if strict else
             {"provider": s.provider, "home": str(s.home), "found": s.exists,
              "relocated_by": s.relocated_by, "retention": s.retention})
            for home in history_homes(self.home) for s in history_sources(home)
        ]
        return {
            "vault": self.vault.status() if self.vault is not None else {
                "mode": "strict", "ready": archive_ready,
                "archive": archive_status,
                "error": archive_error,
            },
            "history_security": "strict" if strict else "legacy_plaintext",
            "sources": sources,
            "sync_interval_minutes": self.interval / 60,
            "auto_import": self.auto_import_enabled(),
            "last_sync": self.last_sync,
            "last_import": self.last_import,
            "last_analysis": self.last_analysis,
            "auto_analyze": _flag("MUNINN_INSIGHTS_AUTO"),
            "last_auto_route": self.last_auto_route,
            "import_progress": self.progress,
            "warnings": ([] if strict else self.retention_warnings()),
        }

    # --- background loop ------------------------------------------------------------

    async def start(self) -> None:
        if strict_history_mode():
            return
        if self._task is None:
            self._auto_since()
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        for task in (self._task, self._job, self._auto_task, *self._background):
            if task and not task.done():
                task.cancel()
        self._task = None
        if self.vault is not None:
            self.vault.close()

    def _auto_since(self) -> Optional[float]:
        """First activation excludes existing historical threads from auto-analysis.

        Manual Phase 9 analysis can still process them after its separate approval.
        """
        if strict_history_mode() or not _flag("MUNINN_INSIGHTS_AUTO"):
            return None
        key = "history_auto_insights_since"
        value = self.memory._metadata.get_meta(key)
        if value is None:
            value = str(time.time())
            self.memory._metadata.set_meta(key, value)
        return float(value)

    def _launch_auto_analysis(self) -> None:
        if strict_history_mode():
            return
        if not (_flag("MUNINN_INSIGHTS_AUTO") and self.auto_import_enabled()):
            return
        if self._auto_task and not self._auto_task.done():
            return
        if self._job and not self._job.done():
            return
        self._auto_task = asyncio.create_task(self._guarded(self._auto_analyze(),
                                                            operation="analysis"))

    async def _auto_analyze(self) -> None:
        """Analyze one new thread with a model suited to its size and current GPU."""
        require_legacy_history_disabled()
        from muninn.history.auto_routing import (
            choose_route,
            guarded_openrouter_available,
            model_hints_for_thread,
            probe_gpu,
            probe_ollama,
        )

        since = self._auto_since()
        if since is None:
            return
        pending = await asyncio.to_thread(
            self.memory._metadata.list_history_threads, None, 1,
            since=since, needs_analysis=True,
        )
        if not pending:
            return
        thread = pending[0]
        hints = model_hints_for_thread(int(thread["turns_imported"]))
        gpu = await asyncio.to_thread(probe_gpu)
        installed, loaded = await asyncio.to_thread(
            probe_ollama, os.environ.get("MUNINN_OLLAMA_URL", "http://localhost:11434")
        )
        if gpu is not None:
            gpu = replace(gpu, loaded_models=loaded)
        route = choose_route(gpu, installed, model_hints=hints,
                             cloud_allowed=False)
        if route.provider == "deferred":
            # Do not touch credentials or OpenRouter when a local route fits.
            # Remote fallback needs a provider-enforced daily key limit.
            cloud_ready = await asyncio.to_thread(guarded_openrouter_available, 1.0)
            if cloud_ready:
                route = choose_route(gpu, installed, model_hints=hints,
                                     cloud_allowed=True, cloud_available=True)
        self.last_auto_route = {"provider": route.provider, "model": route.model,
                                "reason": route.reason, "free_mib": route.free_mib,
                                "at": time.time()}
        if route.provider == "ollama":
            await self.run_analysis(apply=True, provider="ollama", model=route.model,
                                    since=since, limit=1, concurrency=1,
                                    thread_key=thread["thread_key"])
        elif route.provider == "openrouter":
            await self.run_analysis(apply=True, provider="openrouter", since=since,
                                    limit=1, concurrency=1,
                                    thread_key=thread["thread_key"])

    async def _loop(self) -> None:
        while True:
            try:
                require_legacy_history_disabled()
                await self.sync()
                if self.auto_import_enabled() and not (self._job and not self._job.done()):
                    await self.run_import(apply=True)
                    self._launch_auto_analysis()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("History sync failed")
            await asyncio.sleep(self.interval)
