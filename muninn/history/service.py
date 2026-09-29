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
import sqlite3
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx

from muninn.history.blind_index import SearchCancelled, SecureHistoryBlindIndex
from muninn.history.blind_index import _terms as _search_terms
from muninn.history.capture_journal import CaptureJournal
from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.importer import import_history, read_thread
from muninn.history.locations import app_data_dirs, export_candidates, history_homes, history_sources
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.vault import HistoryVault, require_legacy_history_disabled, strict_history_mode

if TYPE_CHECKING:
    from muninn.core.memory import MuninnMemory

logger = logging.getLogger("Muninn.history")
AUTO_IMPORT_META = "history_auto_import"
# Stop fires after every reply; import a live thread at most this often from it.
CAPTURE_DEBOUNCE_SECONDS = 120.0
# Each completed discovery cycle hashes one HMAC-sharded slice of unchanged
# sources. This detects same-size/same-mtime rewrites without rereading the
# full corpus on every cycle (default complete pass: 64 scan cadences).
STRICT_VERIFY_BUCKETS = 64
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
        self._capture_journal: Optional[CaptureJournal] = None
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
        self._secure_index_task: Optional[asyncio.Task] = None
        self._secure_index_wakeup = asyncio.Event()
        self._secure_capture_task: Optional[asyncio.Task] = None
        self._secure_capture_wakeup = asyncio.Event()
        self._secure_search_task: Optional[asyncio.Task] = None
        self._secure_search_wakeup = asyncio.Event()
        self._secure_search_active: tuple[str, threading.Event] | None = None
        self._secure_analysis_task: Optional[asyncio.Task] = None
        self._secure_analysis_wakeup = asyncio.Event()
        self._secure_analysis_active: tuple[str, threading.Event] | None = None
        self._secure_scan_task: Optional[asyncio.Task] = None
        self.last_capture_scan: Optional[Dict[str, Any]] = None
        self.last_secure_capture: Optional[Dict[str, Any]] = None
        self.last_secure_index: Optional[Dict[str, Any]] = None
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

    def _require_capture_journal(self) -> CaptureJournal:
        if getattr(self, "_capture_journal", None) is None:
            self._capture_journal = CaptureJournal(self._require_secure_archive())
        return self._capture_journal

    def _validate_capture_source(self, path: str, provider: str, *, allow_missing: bool = False) -> Path:
        if provider not in ("codex", "claude_code", "gemini_cli"):
            raise ValueError("Strict hook capture only accepts configured chat providers")
        source_path = Path(path).resolve(strict=not allow_missing)
        if provider == "gemini_cli":
            for home in history_homes(self.home):
                try:
                    return self._validate_sync_source(source_path, provider, "transcript", home)
                except ValueError:
                    continue
            raise ValueError("Transcript is outside configured provider history roots")
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
        if report["captured"]:
            self._secure_index_wakeup.set()
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
            async with self._vault_lock:
                outcome = await asyncio.to_thread(
                    archive.archive_file, source_path, provider, expected_source=source_path,
                )
            if outcome["status"] == "captured":
                self._secure_index_wakeup.set()
            return {"captured": outcome["status"] == "captured", "archive": outcome,
                    "indexing": "CPU-only encrypted index eligible"}
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

    def capture_later(self, path: str, provider: str, *, force: bool = False) -> str | None:
        """Commit strict capture intent before acknowledging the hook."""
        if strict_history_mode():
            journal = self._require_capture_journal()
            source = self._validate_capture_source(path, provider, allow_missing=True)
            result = journal.enqueue(source, provider, force=force)
            self._secure_capture_wakeup.set()
            return result
        task = asyncio.create_task(self._guarded(self.capture(path, provider, force=force),
                                                operation="capture"))
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return None

    async def _process_capture_job_once(self) -> bool:
        journal = self._require_capture_journal()
        job = await asyncio.to_thread(journal.claim_due)
        if job is None:
            return False
        try:
            source = self._validate_capture_source(str(job.path), job.provider)
            result = await self.capture(str(source), job.provider, force=True)
            if result.get("archive", {}).get("status") not in {"captured", "unchanged"}:
                await asyncio.to_thread(journal.fail, job, "archive")
                self.last_secure_capture = {"state": "retry", "error_code": "archive", "at": time.time()}
                return True
            committed = await asyncio.to_thread(journal.finish, job, archived=True)
            self.last_secure_capture = {"state": "archived" if committed else "superseded",
                                        "at": time.time()}
        except FileNotFoundError:
            await asyncio.to_thread(journal.fail, job, "missing")
            self.last_secure_capture = {"state": "retry", "error_code": "missing", "at": time.time()}
        except PermissionError:
            await asyncio.to_thread(journal.fail, job, "permission")
            self.last_secure_capture = {"state": "retry", "error_code": "permission", "at": time.time()}
        except (RuntimeError, OSError, ValueError) as exc:
            code = "changed" if isinstance(exc, RuntimeError) and "changed during" in str(exc) else "archive"
            await asyncio.to_thread(journal.fail, job, code)
            self.last_secure_capture = {"state": "retry", "error_code": code, "at": time.time()}
        return True

    async def scan_capture_sources(self) -> Dict[str, int]:
        """Find missed/changed chat transcripts without reading their contents."""
        archive = self._require_secure_archive()
        journal = self._require_capture_journal()

        def scan() -> Dict[str, int]:
            signatures = archive.latest_source_signatures()
            generation = journal.begin_scan()
            batch: list[tuple[str, str]] = []
            for home in history_homes(self.home):
                for source in history_sources(home):
                    if source.provider not in {"codex", "claude_code", "gemini_cli"}:
                        continue
                    if not source.exists:
                        root_key = journal.source_key(source.home.resolve(strict=False), source.provider)
                        if not journal.scan_seen(root_key, generation):
                            batch.append((root_key, "missing"))
                        if len(batch) >= 250:
                            journal.record_scan_batch(generation, batch)
                            batch.clear()
                        continue
                    for item in HistoryVault._source_files(source):
                        key = journal.source_key(item["path"].resolve(strict=False), item["provider"])
                        if journal.scan_seen(key, generation):
                            continue
                        if item["kind"] != "transcript":
                            outcome = "excluded"
                        else:
                            try:
                                path = self._validate_capture_source(str(item["path"]), item["provider"])
                                stat = path.stat()
                                if signatures.get(str(path)) == (stat.st_size, stat.st_mtime_ns):
                                    if int(key[:8], 16) % STRICT_VERIFY_BUCKETS == generation % STRICT_VERIFY_BUCKETS:
                                        outcome = journal.enqueue(path, item["provider"], force=True,
                                                                  immediate=True)
                                        if outcome != "queued":
                                            outcome = "unchanged"
                                    else:
                                        outcome = "unchanged"
                                else:
                                    outcome = journal.enqueue(path, item["provider"], immediate=True)
                                    if outcome != "queued":
                                        outcome = "unchanged"
                            except FileNotFoundError:
                                outcome = "missing"
                            except (OSError, RuntimeError, ValueError):
                                outcome = "errors"
                        batch.append((key, outcome))
                        if len(batch) == 250:
                            journal.record_scan_batch(generation, batch)
                            batch.clear()
            journal.record_scan_batch(generation, batch)
            return journal.finish_scan(generation)

        report = await asyncio.to_thread(scan)
        self.last_capture_scan = {**report, "at": time.time()}
        if report["queued"]:
            self._secure_capture_wakeup.set()
        return report

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

    def secure_search(self, query: str, *, limit: int = 20) -> Dict[str, Any]:
        """Search encrypted history locally; return metadata and short-lived fetch grants."""
        if not strict_history_mode():
            raise RuntimeError("Secure history search requires strict history mode")
        archive = self._require_secure_archive()
        return SecureHistoryBlindIndex(archive).search(query, limit=limit, max_candidates=20)

    def queue_secure_search(self, query: str, *, limit: int = 20) -> str:
        """Commit a private CPU search job before acknowledging its request."""
        if not strict_history_mode():
            raise RuntimeError("Secure history search requires strict history mode")
        job_id = self._require_capture_journal().enqueue_search(query, limit=limit)
        self._secure_search_wakeup.set()
        return job_id

    def secure_search_job_status(self, job_id: str) -> dict[str, Any] | None:
        if not strict_history_mode():
            raise RuntimeError("Secure history search requires strict history mode")
        return self._require_capture_journal().get_search_job(job_id)

    def cancel_secure_search_job(self, job_id: str) -> bool:
        if not strict_history_mode():
            raise RuntimeError("Secure history search requires strict history mode")
        cancelled = self._require_capture_journal().cancel_search(job_id)
        if cancelled and self._secure_search_active and self._secure_search_active[0] == job_id:
            self._secure_search_active[1].set()
        return cancelled

    def secure_analysis_job_status(self, job_id: str) -> dict[str, Any] | None:
        if not strict_history_mode():
            raise RuntimeError("Secure history analysis requires strict history mode")
        return self._require_capture_journal().get_analysis_job(job_id)

    def cancel_secure_analysis_job(self, job_id: str) -> bool:
        if not strict_history_mode():
            raise RuntimeError("Secure history analysis requires strict history mode")
        cancelled = self._require_capture_journal().cancel_analysis(job_id)
        if cancelled and self._secure_analysis_active and self._secure_analysis_active[0] == job_id:
            self._secure_analysis_active[1].set()
        return cancelled

    def secure_fetch_span(self, capability: str, *, max_chars: int = 3000) -> Dict[str, Any]:
        """Return a bounded sanitized span after full snapshot authentication."""
        if not strict_history_mode():
            raise RuntimeError("Secure history fetch requires strict history mode")
        archive = self._require_secure_archive()
        return SecureHistoryBlindIndex(archive).fetch_span(capability, max_chars=max_chars)

    def _secure_model_window(self, capability: str) -> str:
        """In-process inference only; never add this to HTTP or MCP dispatch."""
        if not strict_history_mode():
            raise RuntimeError("Secure history analysis requires strict history mode")
        return SecureHistoryBlindIndex(self._require_secure_archive())._model_window(capability)

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
            "last_secure_index": self.last_secure_index,
            "capture_queue": (self._capture_journal.status() if strict and self._capture_journal is not None
                              else None),
            "last_capture_scan": self.last_capture_scan,
            "last_secure_capture": self.last_secure_capture,
            "import_progress": self.progress,
            "warnings": ([] if strict else self.retention_warnings()),
        }

    # --- background loop ------------------------------------------------------------

    async def start(self) -> None:
        if strict_history_mode():
            self._require_capture_journal()
            if self._secure_capture_task is None:
                self._secure_capture_task = asyncio.create_task(self._secure_capture_loop())
            if self._secure_scan_task is None:
                self._secure_scan_task = asyncio.create_task(self._secure_scan_loop())
            if self._secure_search_task is None:
                self._secure_search_task = asyncio.create_task(self._secure_search_loop())
            if _flag("MUNINN_SECURE_AUTO_ANALYSIS") and self._secure_analysis_task is None:
                self._secure_analysis_task = asyncio.create_task(self._secure_analysis_loop())
            if _flag("MUNINN_HISTORY_INDEX_AUTO") and self._secure_index_task is None:
                self._secure_index_task = asyncio.create_task(self._secure_index_loop())
            return
        if self._task is None:
            self._auto_since()
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        tasks = [task for task in (self._task, self._job, self._auto_task, self._secure_index_task,
                                  self._secure_capture_task, self._secure_scan_task,
                                  self._secure_search_task,
                                  self._secure_analysis_task,
                                  *self._background) if task and not task.done()]
        if self._secure_search_active:
            self._secure_search_active[1].set()
        if self._secure_analysis_active:
            self._secure_analysis_active[1].set()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._task = None
        self._secure_index_task = None
        self._secure_capture_task = None
        self._secure_scan_task = None
        self._secure_search_task = None
        self._secure_analysis_task = None
        if self.vault is not None:
            self.vault.close()

    async def _process_secure_search_once(self) -> bool:
        journal = self._require_capture_journal()
        job = await asyncio.to_thread(journal.claim_search)
        if job is None:
            return False
        cancelled = threading.Event()
        self._secure_search_active = (job.job_id, cancelled)
        search_task: asyncio.Task | None = None
        try:
            index = SecureHistoryBlindIndex(self._require_secure_archive())
            search_task = asyncio.create_task(asyncio.to_thread(
                index.search, job.query, limit=job.limit, max_candidates=20,
                should_cancel=cancelled.is_set,
            ))
            started = time.monotonic()
            while True:
                try:
                    result = await asyncio.wait_for(asyncio.shield(search_task), timeout=15)
                    break
                except asyncio.TimeoutError:
                    if time.monotonic() - started > 1800:
                        cancelled.set()
                    alive = await asyncio.to_thread(
                        journal.heartbeat_search, job.job_id, job.lease_token,
                    )
                    if not alive:
                        cancelled.set()
            if cancelled.is_set():
                raise SearchCancelled()
            analysis_target = None
            if _flag("MUNINN_SECURE_AUTO_ANALYSIS") and result["matches"]:
                try:
                    terms = list(dict.fromkeys(_search_terms(job.query)))
                    analysis_target = await asyncio.to_thread(
                        index._analysis_target, result["matches"][0]["fetch_capability"], terms,
                    )
                except ValueError:
                    # Search evidence remains useful when an expiring hit grant
                    # cannot be converted into an immutable inference target.
                    analysis_target = None
            finished = await asyncio.to_thread(
                journal.finish_search, job.job_id, job.lease_token, result,
                analysis_target=analysis_target,
                analysis_reason=("target_unavailable" if _flag("MUNINN_SECURE_AUTO_ANALYSIS") else "disabled"),
            )
            if finished and analysis_target is not None:
                self._secure_analysis_wakeup.set()
        except asyncio.CancelledError:
            cancelled.set()
            if search_task is not None:
                await asyncio.gather(search_task, return_exceptions=True)
            await asyncio.to_thread(journal.fail_search, job.job_id, job.lease_token,
                                    "worker_timeout")
            raise
        except SearchCancelled:
            await asyncio.to_thread(journal.fail_search, job.job_id, job.lease_token,
                                    "worker_timeout")
        except VaultIntegrityError:
            await asyncio.to_thread(journal.fail_search, job.job_id, job.lease_token,
                                    "vault_integrity")
        except sqlite3.OperationalError:
            await asyncio.to_thread(journal.fail_search, job.job_id, job.lease_token,
                                    "locked")
        except (OSError, RuntimeError, ValueError):
            await asyncio.to_thread(journal.fail_search, job.job_id, job.lease_token,
                                    "archive_unavailable")
        finally:
            self._secure_search_active = None
        return True

    async def _secure_search_loop(self) -> None:
        """Run one authenticated CPU search at a time, independently of hooks."""
        while True:
            try:
                processed = await self._process_secure_search_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Secure history search worker deferred: %s", type(exc).__name__)
                processed = False
            if not processed:
                try:
                    await asyncio.wait_for(self._secure_search_wakeup.wait(), timeout=2)
                    self._secure_search_wakeup.clear()
                except asyncio.TimeoutError:
                    pass

    async def _secure_analysis_heartbeat(self, job_id: str, lease_token: str,
                                         cancelled: threading.Event) -> None:
        journal = self._require_capture_journal()
        while not cancelled.is_set():
            await asyncio.sleep(15)
            if not await asyncio.to_thread(journal.heartbeat_analysis, job_id, lease_token):
                cancelled.set()
                return

    async def _process_secure_analysis_once(self) -> bool:
        """Interpret one pertinent immutable hit without occupying idle VRAM."""
        journal = self._require_capture_journal()
        job = await asyncio.to_thread(journal.claim_analysis)
        if job is None:
            return False
        cancelled = threading.Event()
        self._secure_analysis_active = (job.job_id, cancelled)
        heartbeat = asyncio.create_task(
            self._secure_analysis_heartbeat(job.job_id, job.lease_token, cancelled)
        )
        try:
            index = SecureHistoryBlindIndex(self._require_secure_archive())
            capability = await asyncio.to_thread(index._analysis_capability, job.target)
            if cancelled.is_set():
                return True

            async def before_remote() -> bool:
                if cancelled.is_set():
                    return False
                # This durable marker precedes the HTTP request. After it is
                # set, an interrupted run is outcome_unknown, never auto-retry.
                return await asyncio.to_thread(
                    journal.mark_remote_dispatched, job.job_id, job.lease_token,
                )

            from muninn.history.auto_routing import _local_setting
            from muninn.history.secure_analysis import analyze_secure_hit

            remote_enabled = _local_setting("MUNINN_STRICT_REMOTE_ANALYSIS").lower() in {"1", "true"}

            outcome = await analyze_secure_hit(
                self, capability, allow_remote=remote_enabled, should_cancel=cancelled.is_set,
                before_remote=before_remote,
            )
            if cancelled.is_set():
                return True
            if outcome["status"] == "ok":
                await asyncio.to_thread(journal.finish_analysis, job.job_id,
                                        job.lease_token, outcome)
            elif outcome["status"] == "deferred":
                await asyncio.to_thread(journal.defer_analysis, job.job_id,
                                        job.lease_token, outcome.get("reason", "deferred"))
            else:
                await asyncio.to_thread(journal.fail_analysis, job.job_id,
                                        job.lease_token, "insufficient_context")
        except asyncio.CancelledError:
            cancelled.set()
            # Keep the lease until expiry; remote dispatch is then recovered as
            # outcome_unknown and local work can retry without double charge.
            raise
        except VaultIntegrityError:
            await asyncio.to_thread(journal.fail_analysis, job.job_id,
                                    job.lease_token, "vault_integrity")
        except ValueError:
            await asyncio.to_thread(journal.fail_analysis, job.job_id,
                                    job.lease_token, "snapshot_unavailable")
        except (OSError, RuntimeError, sqlite3.OperationalError, httpx.HTTPError):
            await asyncio.to_thread(journal.fail_analysis, job.job_id,
                                    job.lease_token, "model_unavailable")
        finally:
            cancelled.set()
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            self._secure_analysis_active = None
        return True

    async def _secure_analysis_loop(self) -> None:
        """One optional background inference worker; capture/search stay CPU-only."""
        while True:
            try:
                processed = await self._process_secure_analysis_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Secure analysis worker deferred: %s", type(exc).__name__)
                processed = False
            if not processed:
                try:
                    await asyncio.wait_for(self._secure_analysis_wakeup.wait(), timeout=5)
                    self._secure_analysis_wakeup.clear()
                except asyncio.TimeoutError:
                    pass

    async def _secure_index_loop(self) -> None:
        """CPU-only, resumable history projection; no model or ordinary-memory writes."""
        while True:
            try:
                archive = self._require_secure_archive()
                report = await asyncio.to_thread(
                    SecureHistoryBlindIndex(archive).build, max_snapshots=20,
                )
                self.last_secure_index = {**report, "at": time.time()}
                delay = 300 if report["missing"] == 0 else 30
            except asyncio.CancelledError:
                raise
            except RuntimeError as exc:
                # Another process may own the short-lived builder lock.
                logger.info("Secure history indexing deferred: %s", type(exc).__name__)
                delay = 60
            except Exception:
                logger.exception("Secure history indexing failed")
                delay = 300
            try:
                await asyncio.wait_for(self._secure_index_wakeup.wait(), timeout=delay)
                self._secure_index_wakeup.clear()
            except asyncio.TimeoutError:
                pass

    async def _secure_capture_loop(self) -> None:
        """Replay committed hook requests without retaining a model in memory."""
        while True:
            try:
                in_flight = asyncio.create_task(self._process_capture_job_once())
                try:
                    processed = await asyncio.shield(in_flight)
                except asyncio.CancelledError:
                    # asyncio.to_thread cannot cancel an active archive write.
                    # Wait for it before shutdown reports completion; replay of
                    # any still-claimed row remains idempotent on next startup.
                    await asyncio.gather(in_flight, return_exceptions=True)
                    raise
                if processed:
                    continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Encrypted history capture worker failed (%s)", type(exc).__name__)
            try:
                await asyncio.wait_for(self._secure_capture_wakeup.wait(), timeout=2.0)
                self._secure_capture_wakeup.clear()
            except asyncio.TimeoutError:
                pass

    async def _secure_scan_loop(self) -> None:
        while True:
            try:
                in_flight = asyncio.create_task(self.scan_capture_sources())
                try:
                    await asyncio.shield(in_flight)
                except asyncio.CancelledError:
                    await asyncio.gather(in_flight, return_exceptions=True)
                    raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Encrypted history discovery failed (%s)", type(exc).__name__)
            await asyncio.sleep(self.interval)

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
            cloud_ready = await asyncio.to_thread(guarded_openrouter_available)
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
