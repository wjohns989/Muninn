"""Isolated, synthetic memory and performance benchmark for Muninn.

The harness is intentionally hostile to accidental production access:

* it binds only to a loopback address on an ephemeral non-production port;
* every writable store must resolve beneath a temporary benchmark root;
* child processes receive a minimal environment with no inherited credentials;
* all workload content is deterministic and synthetic; and
* reports redact the synthetic authentication token and omit child environment
  values.

The executable workload runner is added below the reusable safety and metrics
helpers so those contracts can be unit-tested without starting a server.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PRODUCTION_PORT = 42069
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
DEFAULT_SOAK_SECONDS = 30 * 60
DEFAULT_SAMPLE_INTERVAL_SECONDS = 1.0
BENCHMARK_TOKEN_HEADER = "X-Muninn-Benchmark-Token"

_SYSTEM_ENV_ALLOWLIST = frozenset(
    {
        "ALLUSERSPROFILE",
        "APPDATA",
        "COMMONPROGRAMFILES",
        "COMMONPROGRAMFILES(X86)",
        "COMMONPROGRAMW6432",
        "COMSPEC",
        "DRIVERDATA",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "NUMBER_OF_PROCESSORS",
        "OS",
        "PATH",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "PROCESSOR_IDENTIFIER",
        "PROCESSOR_LEVEL",
        "PROCESSOR_REVISION",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "PROGRAMW6432",
        "PSMODULEPATH",
        "PUBLIC",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERDOMAIN",
        "USERNAME",
        "USERPROFILE",
        "WINDIR",
    }
)


def choose_isolated_port(host: str = "127.0.0.1") -> int:
    """Return an unused loopback port that is never Muninn's production port."""
    if host not in LOOPBACK_HOSTS:
        raise ValueError("Benchmark host must be loopback")
    bind_host = "127.0.0.1" if host == "localhost" else host
    family = socket.AF_INET6 if bind_host == "::1" else socket.AF_INET
    for _ in range(20):
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.bind((bind_host, 0))
            port = int(probe.getsockname()[1])
        if port != PRODUCTION_PORT:
            return port
    raise RuntimeError("Could not allocate a non-production benchmark port")


@dataclass(frozen=True)
class IsolationGuard:
    """Fail-closed validation for benchmark network and storage boundaries."""

    root: Path
    data_dir: Path
    host: str
    port: int

    def validate(self) -> None:
        if self.host not in LOOPBACK_HOSTS:
            raise ValueError("Benchmark host must be a loopback address")
        if self.port == PRODUCTION_PORT:
            raise ValueError(f"Refusing to use production port {PRODUCTION_PORT}")
        if not (1 <= int(self.port) <= 65535):
            raise ValueError("Benchmark port must be between 1 and 65535")

        root = self.root.resolve(strict=False)
        data_dir = self.data_dir.resolve(strict=False)
        try:
            data_dir.relative_to(root)
        except ValueError as exc:
            raise ValueError("Benchmark data directory must stay inside temporary benchmark root") from exc
        if data_dir == root:
            raise ValueError("Benchmark data directory must be a child of the temporary benchmark root")


def build_child_environment(
    *,
    inherited: Mapping[str, str],
    root: Path,
    data_dir: Path,
    fixtures_dir: Path,
    port: int,
    benchmark_token: str,
    reranker_enabled: bool,
    conflict_detection_enabled: bool,
) -> dict[str, str]:
    """Build a minimal child environment without inheriting user credentials."""
    guard = IsolationGuard(root=root, data_dir=data_dir, host="127.0.0.1", port=port)
    guard.validate()
    if not benchmark_token:
        raise ValueError("A synthetic benchmark token is required")

    env = {
        key: value
        for key, value in inherited.items()
        if key.upper() in _SYSTEM_ENV_ALLOWLIST and isinstance(value, str)
    }
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_PROXY": "127.0.0.1,localhost,::1",
            "MUNINN_BENCHMARK_MODE": "1",
            "MUNINN_BENCHMARK_ROOT": str(root),
            "MUNINN_BENCHMARK_TOKEN": benchmark_token,
            "MUNINN_DATA_DIR": str(data_dir),
            "MUNINN_HOST": "127.0.0.1",
            "MUNINN_PORT": str(port),
            "MUNINN_AUTH_TOKEN": benchmark_token,
            "MUNINN_SERVER_AUTH_TOKEN": benchmark_token,
            "MUNINN_API_KEY": benchmark_token,
            "MUNINN_INGESTION_ALLOWED_ROOTS": str(fixtures_dir),
            "MUNINN_MULTI_SOURCE_INGESTION": "1",
            "MUNINN_XLAM_ENABLED": "false",
            "MUNINN_INSTRUCTOR_ENABLED": "false",
            "MUNINN_INSTRUCTOR_EXTRACTION": "0",
            # If FastEmbed is unavailable, fail the fallback immediately rather
            # than contacting or starting a machine-wide Ollama service.
            "MUNINN_OLLAMA_URL": "http://127.0.0.1:1",
            "MUNINN_XLAM_URL": "http://127.0.0.1:1/v1",
            "MUNINN_INSTRUCTOR_URL": "http://127.0.0.1:1/v1",
            "MUNINN_RERANKER_ENABLED": "true" if reranker_enabled else "false",
            "MUNINN_CONFLICT_DETECTION": "1" if conflict_detection_enabled else "0",
            "MUNINN_CONSOLIDATION_ENABLED": "false",
            "MUNINN_LEGACY_DISCOVERY_ENABLED": "false",
            "MUNINN_FEDERATION_ENABLED": "false",
            "MUNINN_COLBERT_ENABLED": "false",
            "MUNINN_COLBERT_MULTIVEC": "0",
            "MUNINN_VISION_ENABLED": "false",
            "MUNINN_AUDIO_ENABLED": "false",
            "MUNINN_MCP_AUTOSTART_ON_LAUNCH": "0",
            "MUNINN_MCP_AUTOSTART_SERVER": "0",
            "MUNINN_MCP_AUTOSTART_OLLAMA": "0",
            "MUNINN_BENCHMARK_TRACEMALLOC_FRAMES": "1",
            "MUNINN_LOG_LEVEL": "warning",
        }
    )
    return env


@dataclass(frozen=True)
class ProcessSample:
    elapsed_seconds: float
    phase: str
    working_set_bytes: int
    private_bytes: int
    uss_bytes: int
    thread_count: int
    handle_count: int
    open_file_count: int
    python_heap_bytes: int
    python_peak_bytes: int
    mapped_file_bytes: int
    async_task_count: int = 0
    child_process_count: int = 0
    cpu_percent: float = 0.0


def _mean(samples: Sequence[ProcessSample], field_name: str) -> float:
    values = [float(getattr(sample, field_name)) for sample in samples]
    return statistics.fmean(values) if values else 0.0


def summarize_samples(samples: Sequence[ProcessSample]) -> dict[str, Any]:
    """Summarize peak and retained memory without forcing garbage collection."""
    if not samples:
        return {"sample_count": 0, "peak": {}, "post_workload": {}}

    numeric_fields = (
        "working_set_bytes",
        "private_bytes",
        "uss_bytes",
        "thread_count",
        "handle_count",
        "open_file_count",
        "python_heap_bytes",
        "python_peak_bytes",
        "mapped_file_bytes",
        "async_task_count",
        "child_process_count",
        "cpu_percent",
    )
    post = [sample for sample in samples if sample.phase == "post_workload"]
    if not post:
        tail_count = max(1, min(len(samples), max(3, len(samples) // 10)))
        post = list(samples[-tail_count:])

    return {
        "sample_count": len(samples),
        "peak": {field_name: max(getattr(sample, field_name) for sample in samples) for field_name in numeric_fields},
        "post_workload": {field_name: _mean(post, field_name) for field_name in numeric_fields},
        "phases": sorted({sample.phase for sample in samples}),
    }


def summarize_phases(samples: Sequence[ProcessSample]) -> dict[str, Any]:
    """Report mean and peak resource use independently for each workload phase."""
    fields = (
        "working_set_bytes",
        "private_bytes",
        "uss_bytes",
        "python_heap_bytes",
        "python_peak_bytes",
        "mapped_file_bytes",
        "thread_count",
        "handle_count",
        "open_file_count",
        "async_task_count",
        "child_process_count",
        "cpu_percent",
    )
    grouped: dict[str, list[ProcessSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.phase, []).append(sample)
    return {
        phase: {
            "sample_count": len(phase_samples),
            "mean": {field_name: _mean(phase_samples, field_name) for field_name in fields},
            "peak": {
                field_name: max(getattr(sample, field_name) for sample in phase_samples)
                for field_name in fields
            },
        }
        for phase, phase_samples in sorted(grouped.items())
    }


def compute_quality_metrics(
    ranked_results: Iterable[tuple[str, Sequence[str]]],
    *,
    k: int,
) -> dict[str, Any]:
    """Compute single-relevant-document Recall, MRR, and nDCG at *k*."""
    if k <= 0:
        raise ValueError("k must be positive")
    rows = list(ranked_results)
    recall_total = 0.0
    reciprocal_rank_total = 0.0
    ndcg_total = 0.0
    for expected_id, ranked_ids in rows:
        limited = list(ranked_ids[:k])
        try:
            rank = limited.index(expected_id) + 1
        except ValueError:
            continue
        recall_total += 1.0
        reciprocal_rank_total += 1.0 / rank
        ndcg_total += 1.0 / math.log2(rank + 1)
    count = len(rows)
    denominator = float(count) if count else 1.0
    return {
        "query_count": count,
        "k": k,
        "recall_at_k": recall_total / denominator,
        "mrr_at_k": reciprocal_rank_total / denominator,
        "ndcg_at_k": ndcg_total / denominator,
    }


def sanitize_for_report(value: Any, *, secrets: Sequence[str]) -> Any:
    """Convert report values to JSON-safe types and redact known secret strings."""
    known = tuple(secret for secret in secrets if secret)

    def _sanitize(item: Any) -> Any:
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, Mapping):
            return {str(key): _sanitize(child) for key, child in item.items()}
        if isinstance(item, (list, tuple, set, frozenset)):
            return [_sanitize(child) for child in item]
        if isinstance(item, str):
            rendered = item
            for secret in known:
                rendered = rendered.replace(secret, "[REDACTED]")
            return rendered
        return item

    return _sanitize(value)


def compute_growth_rate(samples: Sequence[ProcessSample], field_name: str) -> dict[str, float]:
    """Return ordinary least-squares growth per second/hour for a sample window."""
    if len(samples) < 2:
        return {"bytes_per_second": 0.0, "bytes_per_hour": 0.0}
    x = [float(sample.elapsed_seconds) for sample in samples]
    y = [float(getattr(sample, field_name)) for sample in samples]
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    slope = 0.0 if denominator == 0 else sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x, y)
    ) / denominator
    return {"bytes_per_second": slope, "bytes_per_hour": slope * 3600.0}


def _request_json(
    base_url: str,
    path: str,
    *,
    auth_token: str,
    benchmark_token: str,
    method: str = "GET",
    payload: Mapping[str, Any] | None = None,
    timeout: float = 120.0,
) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Accept": "application/json",
        BENCHMARK_TOKEN_HEADER: benchmark_token,
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{base_url}{path}", data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code} for {path}: {detail}") from exc
    return json.loads(body) if body else None


def _extract_memory_id(add_response: Any) -> str:
    data = add_response.get("data", {}) if isinstance(add_response, Mapping) else {}
    if isinstance(data, Mapping):
        memory_id = data.get("id") or data.get("memory_id")
        if isinstance(memory_id, str) and memory_id:
            return memory_id
    raise ValueError("Add response did not contain a memory id")


def _extract_ranked_ids(search_response: Any) -> list[str]:
    rows = search_response.get("data", []) if isinstance(search_response, Mapping) else []
    ranked: list[str] = []
    if not isinstance(rows, list):
        return ranked
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        memory_id = row.get("id") or row.get("memory_id")
        if not isinstance(memory_id, str):
            memory = row.get("memory")
            if isinstance(memory, Mapping):
                memory_id = memory.get("id") or memory.get("memory_id")
        if isinstance(memory_id, str):
            ranked.append(memory_id)
    return ranked


class ProcessSampler:
    """Sample parent and child process memory while a workload runs."""

    def __init__(
        self,
        *,
        pid: int,
        base_url: str,
        auth_token: str,
        benchmark_token: str,
        interval_seconds: float,
    ) -> None:
        self.pid = pid
        self.base_url = base_url
        self.auth_token = auth_token
        self.benchmark_token = benchmark_token
        self.interval_seconds = max(0.1, interval_seconds)
        self.samples: list[ProcessSample] = []
        self._phase = "startup"
        self._phase_lock = threading.Lock()
        self._stop = threading.Event()
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="muninn-memory-sampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def set_phase(self, phase: str) -> None:
        with self._phase_lock:
            self._phase = phase

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(5.0, self.interval_seconds * 3))

    def _run(self) -> None:
        try:
            import psutil
        except ImportError:
            return
        try:
            root = psutil.Process(self.pid)
        except psutil.Error:
            return
        root.cpu_percent(None)
        while not self._stop.is_set():
            with self._phase_lock:
                phase = self._phase
            processes = [root]
            try:
                processes.extend(root.children(recursive=True))
            except psutil.Error:
                pass
            working_set = private_bytes = uss = threads = handles = open_files = mapped = 0
            cpu_percent = 0.0
            live_processes = 0
            for process in processes:
                try:
                    info = process.memory_info()
                    full = process.memory_full_info()
                    working_set += int(getattr(info, "rss", 0))
                    private_bytes += int(getattr(full, "private", getattr(info, "private", 0)))
                    uss += int(getattr(full, "uss", 0))
                    threads += process.num_threads()
                    handles += int(process.num_handles()) if hasattr(process, "num_handles") else 0
                    open_files += len(process.open_files())
                    cpu_percent += process.cpu_percent(None)
                    live_processes += 1
                    try:
                        for region in process.memory_maps(grouped=True):
                            path = str(getattr(region, "path", ""))
                            if path and not path.startswith("["):
                                mapped += int(getattr(region, "rss", 0))
                    except (psutil.AccessDenied, psutil.NoSuchProcess, NotImplementedError):
                        pass
                except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                    continue

            internal: Mapping[str, Any] = {}
            try:
                internal = _request_json(
                    self.base_url,
                    "/_benchmark/snapshot",
                    auth_token=self.auth_token,
                    benchmark_token=self.benchmark_token,
                    timeout=min(5.0, self.interval_seconds * 2 + 1),
                )
            except Exception:
                pass
            self.samples.append(
                ProcessSample(
                    elapsed_seconds=time.monotonic() - self._started_at,
                    phase=phase,
                    working_set_bytes=working_set,
                    private_bytes=private_bytes,
                    uss_bytes=uss,
                    thread_count=threads,
                    handle_count=handles,
                    open_file_count=open_files,
                    python_heap_bytes=int(internal.get("python_heap_bytes", 0)),
                    python_peak_bytes=int(internal.get("python_peak_bytes", 0)),
                    mapped_file_bytes=mapped,
                    async_task_count=int(internal.get("async_task_count", 0)),
                    child_process_count=max(0, live_processes - 1),
                    cpu_percent=cpu_percent,
                )
            )
            self._stop.wait(self.interval_seconds)


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": statistics.median(values),
        "p95_ms": ordered[p95_index],
        "max_ms": max(values),
    }


def _timed_request(*args, **kwargs) -> tuple[Any, float]:
    started = time.perf_counter()
    result = _request_json(*args, **kwargs)
    return result, (time.perf_counter() - started) * 1000.0


def _wait_for_health(base_url: str, token: str, benchmark_token: str, timeout: float = 300.0) -> dict:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            health = _request_json(
                base_url,
                "/health",
                auth_token=token,
                benchmark_token=benchmark_token,
                timeout=3.0,
            )
            if isinstance(health, Mapping) and health.get("status") in {"ok", "healthy"}:
                return dict(health)
        except Exception as exc:
            last_error = exc
        time.sleep(0.25)
    raise TimeoutError(f"Isolated Muninn server did not become healthy: {last_error}")


def _synthetic_records(count: int) -> list[dict[str, str]]:
    return [
        {
            "topic": f"synthetic-topic-{index:04d}",
            "content": (
                f"Synthetic benchmark fact {index:04d}. The unique retrieval marker is "
                f"quartz-orbit-{index:04d}. This record contains no personal information."
            ),
            "query": f"Which memory contains quartz-orbit-{index:04d}?",
        }
        for index in range(count)
    ]


def _write_ingestion_fixtures(fixtures_dir: Path, count: int = 12) -> list[Path]:
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(count):
        path = fixtures_dir / f"synthetic_ingest_{index:03d}.txt"
        path.write_text(
            (
                f"Synthetic ingestion document {index}. "
                f"Its deterministic marker is amber-pipeline-{index:03d}. "
            ) * 12,
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def _run_workload(
    *,
    base_url: str,
    token: str,
    benchmark_token: str,
    fixtures_dir: Path,
    sampler: ProcessSampler,
    record_count: int,
    repeated_searches: int,
    mixed_operations: int,
    soak_seconds: float,
    idle_soak_seconds: float,
    reranker_enabled: bool,
) -> dict[str, Any]:
    phase_latencies: dict[str, list[float]] = {}
    records = _synthetic_records(record_count)
    expected: list[tuple[str, str]] = []

    sampler.set_phase("warm_idle")
    time.sleep(5.0)

    sampler.set_phase("add_records")
    for record in records:
        response, latency = _timed_request(
            base_url,
            "/add",
            auth_token=token,
            benchmark_token=benchmark_token,
            method="POST",
            payload={
                "content": record["content"],
                "user_id": "benchmark-user",
                "namespace": "benchmark",
                "scope": "project",
                "metadata": {
                    "project": "synthetic-memory-benchmark",
                    "synthetic_topic": record["topic"],
                    "muninn_skip_extraction": True,
                },
            },
        )
        phase_latencies.setdefault("add", []).append(latency)
        expected.append((record["query"], _extract_memory_id(response)))

    search_payload = {
        "query": expected[0][0],
        "user_id": "benchmark-user",
        "limit": 5,
        "rerank": reranker_enabled,
        "filters": {"project": "synthetic-memory-benchmark"},
        "namespaces": ["benchmark"],
    }
    sampler.set_phase("first_search")
    _, first_search_ms = _timed_request(
        base_url,
        "/search",
        auth_token=token,
        benchmark_token=benchmark_token,
        method="POST",
        payload=search_payload,
    )

    sampler.set_phase("repeated_search")
    for index in range(repeated_searches):
        query, _ = expected[index % len(expected)]
        payload = dict(search_payload)
        payload["query"] = query
        _, latency = _timed_request(
            base_url,
            "/search",
            auth_token=token,
            benchmark_token=benchmark_token,
            method="POST",
            payload=payload,
        )
        phase_latencies.setdefault("search", []).append(latency)

    sampler.set_phase("quality_before_consolidation")
    quality_rows: list[tuple[str, Sequence[str]]] = []
    for query, expected_id in expected:
        payload = dict(search_payload)
        payload["query"] = query
        response = _request_json(
            base_url,
            "/search",
            auth_token=token,
            benchmark_token=benchmark_token,
            method="POST",
            payload=payload,
        )
        quality_rows.append((expected_id, _extract_ranked_ids(response)))
    quality_before = compute_quality_metrics(quality_rows, k=5)

    sampler.set_phase("ingestion")
    fixture_paths = _write_ingestion_fixtures(fixtures_dir)
    ingest_response, ingest_ms = _timed_request(
        base_url,
        "/ingest",
        auth_token=token,
        benchmark_token=benchmark_token,
        method="POST",
        payload={
            "sources": [str(path) for path in fixture_paths],
            "user_id": "benchmark-user",
            "namespace": "benchmark",
            "project": "synthetic-memory-benchmark",
            "metadata": {"muninn_skip_extraction": True, "synthetic": True},
            "recursive": False,
        },
        timeout=300.0,
    )

    sampler.set_phase("consolidation")
    consolidation_response, consolidation_ms = _timed_request(
        base_url,
        "/consolidation/run",
        auth_token=token,
        benchmark_token=benchmark_token,
        method="POST",
        payload={},
        timeout=300.0,
    )

    sampler.set_phase("mixed_workload")
    mixed_errors = 0
    for index in range(mixed_operations):
        try:
            if index % 4 == 0:
                _, latency = _timed_request(
                    base_url,
                    "/add",
                    auth_token=token,
                    benchmark_token=benchmark_token,
                    method="POST",
                    payload={
                        "content": f"Synthetic mixed-write marker cobalt-mix-{index:05d}.",
                        "user_id": "benchmark-user",
                        "namespace": "benchmark",
                        "scope": "project",
                        "metadata": {
                            "project": "synthetic-memory-benchmark",
                            "muninn_skip_extraction": True,
                        },
                    },
                )
                phase_latencies.setdefault("mixed_write", []).append(latency)
            else:
                payload = dict(search_payload)
                payload["query"] = expected[index % len(expected)][0]
                _, latency = _timed_request(
                    base_url,
                    "/search",
                    auth_token=token,
                    benchmark_token=benchmark_token,
                    method="POST",
                    payload=payload,
                )
                phase_latencies.setdefault("mixed_read", []).append(latency)
        except Exception:
            mixed_errors += 1

    sampler.set_phase("soak")
    soak_deadline = time.monotonic() + max(0.0, soak_seconds)
    soak_operations = soak_errors = 0
    while time.monotonic() < soak_deadline:
        try:
            payload = dict(search_payload)
            payload["query"] = expected[soak_operations % len(expected)][0]
            _request_json(
                base_url,
                "/search",
                auth_token=token,
                benchmark_token=benchmark_token,
                method="POST",
                payload=payload,
            )
            soak_operations += 1
        except Exception:
            soak_errors += 1
        time.sleep(min(5.0, max(0.25, soak_deadline - time.monotonic())))

    sampler.set_phase("quality_after_consolidation")
    quality_after_rows: list[tuple[str, Sequence[str]]] = []
    for query, expected_id in expected:
        payload = dict(search_payload)
        payload["query"] = query
        response = _request_json(
            base_url,
            "/search",
            auth_token=token,
            benchmark_token=benchmark_token,
            method="POST",
            payload=payload,
        )
        quality_after_rows.append((expected_id, _extract_ranked_ids(response)))

    sampler.set_phase("idle_soak")
    idle_deadline = time.monotonic() + max(0.0, idle_soak_seconds)
    while time.monotonic() < idle_deadline:
        time.sleep(min(5.0, max(0.25, idle_deadline - time.monotonic())))

    sampler.set_phase("post_workload")
    time.sleep(5.0)
    return {
        "first_search_ms": first_search_ms,
        "latency": {name: _latency_summary(values) for name, values in phase_latencies.items()},
        "ingestion_ms": ingest_ms,
        "ingestion_result": ingest_response,
        "consolidation_ms": consolidation_ms,
        "consolidation_result": consolidation_response,
        "quality_before_consolidation": quality_before,
        "quality_after_consolidation": compute_quality_metrics(quality_after_rows, k=5),
        "mixed_errors": mixed_errors,
        "soak_operations": soak_operations,
        "soak_errors": soak_errors,
        "idle_soak_seconds": idle_soak_seconds,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import psutil  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("psutil is required for the memory benchmark") from exc

    repo_root = Path(__file__).resolve().parent.parent
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    port = choose_isolated_port()
    base_url = f"http://127.0.0.1:{port}"
    started_at = datetime.now(timezone.utc)
    report: dict[str, Any]

    with tempfile.TemporaryDirectory(prefix="muninn-memory-benchmark-") as raw_root:
        root = Path(raw_root)
        data_dir = root / "data"
        fixtures_dir = root / "fixtures"
        logs_dir = root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        IsolationGuard(root=root, data_dir=data_dir, host="127.0.0.1", port=port).validate()
        env = build_child_environment(
            inherited=os.environ,
            root=root,
            data_dir=data_dir,
            fixtures_dir=fixtures_dir,
            port=port,
            benchmark_token=token,
            reranker_enabled=args.reranker,
            conflict_detection_enabled=args.conflict_detection,
        )
        stdout_path = logs_dir / "server.stdout.log"
        stderr_path = logs_dir / "server.stderr.log"
        startup_started = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                [sys.executable, "-m", "eval.memory_profile_server"],
                cwd=str(repo_root),
                env=env,
                stdout=stdout,
                stderr=stderr,
                text=True,
                creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
            )
            sampler = ProcessSampler(
                pid=process.pid,
                base_url=base_url,
                auth_token=token,
                benchmark_token=token,
                interval_seconds=args.sample_interval,
            )
            sampler.start()
            clean_shutdown = False
            workload: dict[str, Any] = {}
            health: dict[str, Any] = {}
            initial_snapshot: dict[str, Any] = {}
            final_snapshot: dict[str, Any] = {}
            error: str | None = None
            try:
                health = _wait_for_health(base_url, token, token)
                startup_ms = (time.perf_counter() - startup_started) * 1000.0
                initial_snapshot = _request_json(
                    base_url,
                    "/_benchmark/snapshot",
                    auth_token=token,
                    benchmark_token=token,
                )
                workload = _run_workload(
                    base_url=base_url,
                    token=token,
                    benchmark_token=token,
                    fixtures_dir=fixtures_dir,
                    sampler=sampler,
                    record_count=max(1, args.records),
                    repeated_searches=max(1, args.repeated_searches),
                    mixed_operations=max(1, args.mixed_operations),
                    soak_seconds=max(0.0, args.soak_seconds),
                    idle_soak_seconds=max(0.0, args.idle_soak_seconds),
                    reranker_enabled=args.reranker,
                )
                final_snapshot = _request_json(
                    base_url,
                    "/_benchmark/snapshot",
                    auth_token=token,
                    benchmark_token=token,
                )
                sampler.set_phase("shutdown")
                _request_json(
                    base_url,
                    "/_benchmark/shutdown",
                    auth_token=token,
                    benchmark_token=token,
                    method="POST",
                    payload={},
                    timeout=10.0,
                )
                process.wait(timeout=60.0)
                clean_shutdown = process.returncode == 0
            except Exception as exc:
                startup_ms = (time.perf_counter() - startup_started) * 1000.0
                error = f"{type(exc).__name__}: {exc}"
            finally:
                if process.poll() is None:
                    try:
                        _request_json(
                            base_url,
                            "/_benchmark/shutdown",
                            auth_token=token,
                            benchmark_token=token,
                            method="POST",
                            payload={},
                            timeout=10.0,
                        )
                        process.wait(timeout=60.0)
                        clean_shutdown = process.returncode == 0
                    except Exception:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=20.0)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait(timeout=10.0)
                sampler.stop()

        summary = summarize_samples(sampler.samples)
        soak_samples = [sample for sample in sampler.samples if sample.phase == "soak"]
        idle_soak_samples = [sample for sample in sampler.samples if sample.phase == "idle_soak"]
        report = {
            "schema": "muninn_memory_profile/v1",
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "isolation": {
                "loopback_only": True,
                "production_port_rejected": True,
                "ephemeral_port": port,
                "temporary_storage": True,
                "production_store_accessed": False,
                "credentials_inherited": False,
            },
            "configuration": {
                "records": args.records,
                "repeated_searches": args.repeated_searches,
                "mixed_operations": args.mixed_operations,
                "soak_seconds": args.soak_seconds,
                "idle_soak_seconds": args.idle_soak_seconds,
                "sample_interval_seconds": args.sample_interval,
                "reranker_enabled": args.reranker,
                "conflict_detection_enabled": args.conflict_detection,
                "external_model_calls_disabled": True,
            },
            "startup_ms": startup_ms,
            "health": health,
            "snapshots": {
                "after_startup": initial_snapshot,
                "after_workload": final_snapshot,
            },
            "workload": workload,
            "memory": summary,
            "phase_memory": summarize_phases(sampler.samples),
            "growth": {
                "working_set": compute_growth_rate(soak_samples, "working_set_bytes"),
                "private_bytes": compute_growth_rate(soak_samples, "private_bytes"),
                "python_heap": compute_growth_rate(soak_samples, "python_heap_bytes"),
            },
            "idle_growth": {
                "working_set": compute_growth_rate(idle_soak_samples, "working_set_bytes"),
                "private_bytes": compute_growth_rate(idle_soak_samples, "private_bytes"),
                "python_heap": compute_growth_rate(idle_soak_samples, "python_heap_bytes"),
            },
            "clean_shutdown": clean_shutdown,
            "process_returncode": process.returncode,
            "error": error,
        }

    sanitized = sanitize_for_report(report, secrets=[token])
    output_path.write_text(json.dumps(sanitized, indent=2, sort_keys=True), encoding="utf-8")
    return sanitized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an isolated synthetic Muninn memory benchmark")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument("--soak-seconds", type=float, default=DEFAULT_SOAK_SECONDS)
    parser.add_argument("--idle-soak-seconds", type=float, default=DEFAULT_SOAK_SECONDS)
    parser.add_argument("--sample-interval", type=float, default=DEFAULT_SAMPLE_INTERVAL_SECONDS)
    parser.add_argument("--records", type=int, default=40)
    parser.add_argument("--repeated-searches", type=int, default=100)
    parser.add_argument("--mixed-operations", type=int, default=100)
    parser.add_argument("--reranker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--conflict-detection", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.soak_seconds < 0:
        parser.error("--soak-seconds must be non-negative")
    if args.idle_soak_seconds < 0:
        parser.error("--idle-soak-seconds must be non-negative")
    if args.sample_interval <= 0:
        parser.error("--sample-interval must be positive")
    report = run_benchmark(args)
    print(json.dumps({
        "output": str(args.output),
        "clean_shutdown": report["clean_shutdown"],
        "error": report["error"],
    }, sort_keys=True))
    return 0 if report["clean_shutdown"] and report["error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
