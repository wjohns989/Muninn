"""
Periodic ingestion scheduler for local-first Muninn deployments.

This module provides a lightweight asyncio scheduler that can repeatedly run
``memory.ingest_sources`` on a fixed cadence with overlap protection and
runtime status visibility.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger("Muninn.Ingestion.Periodic")
_SUPPORTED_MODEL_PROFILES = {"low_latency", "balanced", "high_reasoning"}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_int(name: str, *, min_value: int = 1) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r (expected integer)", name, raw)
        return None
    if value < min_value:
        logger.warning("Ignoring %s=%r because it is below minimum %d", name, raw, min_value)
        return None
    return value


def _env_optional_json_object(name: str) -> Dict[str, Any]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Ignoring invalid JSON object in %s", name)
        return {}
    if not isinstance(parsed, dict):
        logger.warning("Ignoring %s because it is not a JSON object", name)
        return {}
    return parsed


def _env_float(name: str, default: float, *, min_value: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r (expected float)", name, raw)
        return default
    if value < min_value:
        logger.warning(
            "Ignoring %s=%r because it is below minimum %.3f",
            name,
            raw,
            min_value,
        )
        return default
    return value


def _env_optional_float(name: str, *, min_value: float) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r (expected float)", name, raw)
        return None
    if value < min_value:
        logger.warning(
            "Ignoring %s=%r because it is below minimum %.3f",
            name,
            raw,
            min_value,
        )
        return None
    return value


def _parse_sources(raw_sources: str) -> List[str]:
    values = [part.strip() for part in raw_sources.split(os.pathsep)]
    return [value for value in values if value]


@dataclass(frozen=True)
class PeriodicIngestionSettings:
    """Configuration for periodic ingestion cadence."""

    enabled_on_startup: bool = False
    run_on_start: bool = False
    interval_seconds: float = 900.0
    sources: List[str] = field(default_factory=list)
    user_id: str = "global_user"
    namespace: str = "global"
    project: str = "global"
    metadata: Dict[str, Any] = field(default_factory=dict)
    recursive: bool = False
    chronological_order: str = "none"
    max_file_size_bytes: Optional[int] = None
    chunk_size_chars: Optional[int] = None
    chunk_overlap_chars: Optional[int] = None
    min_chunk_chars: Optional[int] = None
    model_profile: Optional[str] = None
    skip_extraction: bool = False
    extraction_timeout_seconds: Optional[float] = None
    run_timeout_seconds: Optional[float] = None
    run_timeout_skip_warmup_runs: int = 0
    failure_backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 3600.0
    jitter_ratio: float = 0.1

    @classmethod
    def from_env(cls) -> "PeriodicIngestionSettings":
        raw_interval = os.environ.get("MUNINN_PERIODIC_INGESTION_INTERVAL_SECONDS", "900").strip()
        try:
            interval_seconds = float(raw_interval)
        except ValueError:
            logger.warning(
                "Ignoring invalid MUNINN_PERIODIC_INGESTION_INTERVAL_SECONDS=%r; using 900.0",
                raw_interval,
            )
            interval_seconds = 900.0
        if interval_seconds < 5.0:
            logger.warning(
                "MUNINN_PERIODIC_INGESTION_INTERVAL_SECONDS=%s is too low; clamping to 5.0",
                interval_seconds,
            )
            interval_seconds = 5.0

        chronological_order = os.environ.get(
            "MUNINN_PERIODIC_INGESTION_CHRONOLOGICAL_ORDER",
            "none",
        ).strip()
        if chronological_order not in {"none", "oldest_first", "newest_first"}:
            logger.warning(
                "Ignoring invalid MUNINN_PERIODIC_INGESTION_CHRONOLOGICAL_ORDER=%r; using 'none'",
                chronological_order,
            )
            chronological_order = "none"

        model_profile = os.environ.get("MUNINN_PERIODIC_INGESTION_MODEL_PROFILE", "").strip()
        if model_profile == "":
            model_profile = None
        elif model_profile not in _SUPPORTED_MODEL_PROFILES:
            logger.warning(
                "Ignoring invalid MUNINN_PERIODIC_INGESTION_MODEL_PROFILE=%r; expected one of %s",
                model_profile,
                sorted(_SUPPORTED_MODEL_PROFILES),
            )
            model_profile = None

        failure_backoff_multiplier = _env_float(
            "MUNINN_PERIODIC_INGESTION_FAILURE_BACKOFF_MULTIPLIER",
            2.0,
            min_value=1.0,
        )
        max_backoff_seconds = _env_float(
            "MUNINN_PERIODIC_INGESTION_MAX_BACKOFF_SECONDS",
            3600.0,
            min_value=5.0,
        )
        if max_backoff_seconds < interval_seconds:
            logger.warning(
                "MUNINN_PERIODIC_INGESTION_MAX_BACKOFF_SECONDS=%.3f is below interval %.3f; clamping to interval",
                max_backoff_seconds,
                interval_seconds,
            )
            max_backoff_seconds = interval_seconds
        jitter_ratio = _env_float(
            "MUNINN_PERIODIC_INGESTION_JITTER_RATIO",
            0.1,
            min_value=0.0,
        )
        if jitter_ratio > 1.0:
            logger.warning(
                "MUNINN_PERIODIC_INGESTION_JITTER_RATIO=%.3f is above 1.0; clamping to 1.0",
                jitter_ratio,
            )
            jitter_ratio = 1.0

        return cls(
            enabled_on_startup=_env_flag("MUNINN_PERIODIC_INGESTION_ENABLED", False),
            run_on_start=_env_flag("MUNINN_PERIODIC_INGESTION_RUN_ON_START", False),
            interval_seconds=interval_seconds,
            sources=_parse_sources(os.environ.get("MUNINN_PERIODIC_INGESTION_SOURCES", "")),
            user_id=os.environ.get("MUNINN_PERIODIC_INGESTION_USER_ID", "global_user"),
            namespace=os.environ.get("MUNINN_PERIODIC_INGESTION_NAMESPACE", "global"),
            project=os.environ.get("MUNINN_PERIODIC_INGESTION_PROJECT", "global"),
            metadata=_env_optional_json_object("MUNINN_PERIODIC_INGESTION_METADATA_JSON"),
            recursive=_env_flag("MUNINN_PERIODIC_INGESTION_RECURSIVE", False),
            chronological_order=chronological_order,
            max_file_size_bytes=_env_optional_int("MUNINN_PERIODIC_INGESTION_MAX_FILE_SIZE_BYTES"),
            chunk_size_chars=_env_optional_int("MUNINN_PERIODIC_INGESTION_CHUNK_SIZE_CHARS"),
            chunk_overlap_chars=_env_optional_int(
                "MUNINN_PERIODIC_INGESTION_CHUNK_OVERLAP_CHARS",
                min_value=0,
            ),
            min_chunk_chars=_env_optional_int("MUNINN_PERIODIC_INGESTION_MIN_CHUNK_CHARS"),
            model_profile=model_profile,
            skip_extraction=_env_flag("MUNINN_PERIODIC_INGESTION_SKIP_EXTRACTION", False),
            extraction_timeout_seconds=_env_optional_float(
                "MUNINN_PERIODIC_INGESTION_EXTRACT_TIMEOUT_SECONDS",
                min_value=0.1,
            ),
            run_timeout_seconds=_env_optional_float(
                "MUNINN_PERIODIC_INGESTION_RUN_TIMEOUT_SECONDS",
                min_value=1.0,
            ),
            run_timeout_skip_warmup_runs=(
                _env_optional_int(
                    "MUNINN_PERIODIC_INGESTION_RUN_TIMEOUT_SKIP_WARMUP_RUNS",
                    min_value=0,
                )
                or 0
            ),
            failure_backoff_multiplier=failure_backoff_multiplier,
            max_backoff_seconds=max_backoff_seconds,
            jitter_ratio=jitter_ratio,
        )


class PeriodicIngestionScheduler:
    """
    Async scheduler for repeat ingestion runs.

    Guarantees at-most-one active ingestion cycle from this scheduler instance.
    """

    def __init__(
        self,
        *,
        memory: Any,
        settings: PeriodicIngestionSettings,
        now_fn: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        random_fn: Callable[[], float] = random.random,
    ) -> None:
        self._memory = memory
        self._settings = settings
        self._now_fn = now_fn
        self._sleep_fn = sleep_fn
        self._random_fn = random_fn
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._run_lock = asyncio.Lock()

        self._last_run_started_at: Optional[float] = None
        self._last_run_finished_at: Optional[float] = None
        self._last_run_status: str = "never"
        self._last_error: Optional[str] = None
        self._last_result: Optional[Dict[str, Any]] = None
        self._next_run_epoch: Optional[float] = None
        self._run_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._consecutive_failures = 0
        self._last_scheduled_sleep_seconds: Optional[float] = None
        self._last_run_elapsed_seconds: Optional[float] = None
        self._last_run_timeout_enforced: bool = False

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "configured": {
                "enabled_on_startup": self._settings.enabled_on_startup,
                "run_on_start": self._settings.run_on_start,
                "interval_seconds": self._settings.interval_seconds,
                "sources": list(self._settings.sources),
                "recursive": self._settings.recursive,
                "chronological_order": self._settings.chronological_order,
                "user_id": self._settings.user_id,
                "namespace": self._settings.namespace,
                "project": self._settings.project,
                "model_profile": self._settings.model_profile,
                "skip_extraction": self._settings.skip_extraction,
                "extraction_timeout_seconds": self._settings.extraction_timeout_seconds,
                "run_timeout_seconds": self._settings.run_timeout_seconds,
                "run_timeout_skip_warmup_runs": self._settings.run_timeout_skip_warmup_runs,
                "failure_backoff_multiplier": self._settings.failure_backoff_multiplier,
                "max_backoff_seconds": self._settings.max_backoff_seconds,
                "jitter_ratio": self._settings.jitter_ratio,
            },
            "runtime": {
                "running": self._running,
                "inflight": self._run_lock.locked(),
                "next_run_epoch": self._next_run_epoch,
                "last_run_started_at": self._last_run_started_at,
                "last_run_finished_at": self._last_run_finished_at,
                "last_run_status": self._last_run_status,
                "last_error": self._last_error,
                "run_count": self._run_count,
                "success_count": self._success_count,
                "failure_count": self._failure_count,
                "consecutive_failures": self._consecutive_failures,
                "last_scheduled_sleep_seconds": self._last_scheduled_sleep_seconds,
                "last_run_elapsed_seconds": self._last_run_elapsed_seconds,
                "last_run_timeout_enforced": self._last_run_timeout_enforced,
            },
            "last_result": self._last_result,
        }

    async def start(self) -> bool:
        if self._running:
            return False
        if not self._settings.sources:
            logger.warning(
                "Periodic ingestion start requested without configured sources; scheduler remains stopped"
            )
            return False
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="muninn-periodic-ingestion")
        logger.info(
            "Periodic ingestion started (interval=%.1fs, sources=%d)",
            self._settings.interval_seconds,
            len(self._settings.sources),
        )
        return True

    async def stop(self) -> bool:
        if not self._running and self._task is None:
            return False
        self._running = False
        self._next_run_epoch = None
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        logger.info("Periodic ingestion stopped")
        return True

    async def trigger_once(self, *, reason: str) -> Dict[str, Any]:
        if not self._settings.sources:
            return {
                "success": False,
                "event": "PERIODIC_INGESTION_SKIPPED",
                "reason": "no_sources_configured",
            }
        if self._run_lock.locked():
            return {
                "success": False,
                "event": "PERIODIC_INGESTION_SKIPPED",
                "reason": "already_running",
            }

        async with self._run_lock:
            self._run_count += 1
            self._last_run_started_at = self._now_fn()
            self._last_error = None
            self._last_run_timeout_enforced = False
            run_id = self._run_count
            try:
                metadata = dict(self._settings.metadata)
                if self._settings.model_profile and "operator_model_profile" not in metadata:
                    metadata["operator_model_profile"] = self._settings.model_profile
                if self._settings.skip_extraction and "muninn_skip_extraction" not in metadata:
                    metadata["muninn_skip_extraction"] = True
                if (
                    self._settings.extraction_timeout_seconds is not None
                    and "muninn_extraction_timeout_seconds" not in metadata
                ):
                    metadata["muninn_extraction_timeout_seconds"] = (
                        self._settings.extraction_timeout_seconds
                    )

                ingest_coro = self._memory.ingest_sources(
                    sources=list(self._settings.sources),
                    user_id=self._settings.user_id,
                    namespace=self._settings.namespace,
                    project=self._settings.project,
                    metadata=metadata,
                    recursive=self._settings.recursive,
                    chronological_order=self._settings.chronological_order,
                    max_file_size_bytes=self._settings.max_file_size_bytes,
                    chunk_size_chars=self._settings.chunk_size_chars,
                    chunk_overlap_chars=self._settings.chunk_overlap_chars,
                    min_chunk_chars=self._settings.min_chunk_chars,
                )
                timeout_enforced = (
                    self._settings.run_timeout_seconds is not None
                    and run_id > self._settings.run_timeout_skip_warmup_runs
                )
                self._last_run_timeout_enforced = timeout_enforced
                if timeout_enforced:
                    result = await asyncio.wait_for(
                        ingest_coro,
                        timeout=self._settings.run_timeout_seconds,
                    )
                else:
                    result = await ingest_coro
            except asyncio.TimeoutError:
                self._failure_count += 1
                self._consecutive_failures += 1
                self._last_run_status = "failed"
                self._last_error = "periodic run timeout"
                self._last_result = None
                logger.error("Periodic ingestion run #%d timed out", run_id)
                return {
                    "success": False,
                    "event": "PERIODIC_INGESTION_FAILED",
                    "reason": reason,
                    "run_id": run_id,
                    "error": "periodic run timeout",
                    "timeout_enforced": self._last_run_timeout_enforced,
                }
            except Exception as exc:
                self._failure_count += 1
                self._consecutive_failures += 1
                self._last_run_status = "failed"
                self._last_error = str(exc)
                self._last_result = None
                logger.error("Periodic ingestion run #%d failed: %s", run_id, exc)
                return {
                    "success": False,
                    "event": "PERIODIC_INGESTION_FAILED",
                    "reason": reason,
                    "run_id": run_id,
                    "error": str(exc),
                    "timeout_enforced": self._last_run_timeout_enforced,
                }
            finally:
                self._last_run_finished_at = self._now_fn()
                self._last_run_elapsed_seconds = max(
                    0.0,
                    self._last_run_finished_at - self._last_run_started_at,
                )

            self._success_count += 1
            self._consecutive_failures = 0
            self._last_run_status = "completed"
            self._last_result = result
            logger.info("Periodic ingestion run #%d completed (%s)", run_id, reason)
            return {
                "success": True,
                "event": "PERIODIC_INGESTION_COMPLETED",
                "reason": reason,
                "run_id": run_id,
                "result": result,
                "timeout_enforced": self._last_run_timeout_enforced,
            }

    def _compute_next_sleep_seconds(self) -> float:
        if self._consecutive_failures <= 0:
            delay = self._settings.interval_seconds
        else:
            delay = self._settings.interval_seconds * (
                self._settings.failure_backoff_multiplier ** self._consecutive_failures
            )
            delay = min(delay, self._settings.max_backoff_seconds)

        if self._settings.jitter_ratio > 0:
            jitter_max = delay * self._settings.jitter_ratio
            delay += self._random_fn() * jitter_max
        return delay

    async def _run_loop(self) -> None:
        try:
            if self._settings.run_on_start:
                await self.trigger_once(reason="startup")
            while self._running:
                sleep_seconds = self._compute_next_sleep_seconds()
                self._last_scheduled_sleep_seconds = sleep_seconds
                self._next_run_epoch = self._now_fn() + sleep_seconds
                try:
                    await self._sleep_fn(sleep_seconds)
                except asyncio.CancelledError:
                    break
                if not self._running:
                    break
                await self.trigger_once(reason="scheduled")
        finally:
            self._next_run_epoch = None
