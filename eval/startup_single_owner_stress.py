#!/usr/bin/env python3
"""
Muninn single-owner startup stress harness.

Purpose:
- Validate that many concurrent callers of `ensure_server_running()` converge
  on one backend owner process.
- Run in an isolated local port + data dir to avoid disrupting active Muninn.
- Report whether listener ownership ever exceeds 1 PID concurrently.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _listener_pids_for_port(port: int) -> Set[int]:
    try:
        out = subprocess.check_output(
            ["netstat", "-ano", "-p", "tcp"],
            text=True,
            errors="ignore",
        )
    except Exception:
        return set()

    pids: Set[int] = set()
    for raw in out.splitlines():
        line = raw.strip()
        if not line or "LISTENING" not in line:
            continue
        parts = line.split()
        # Windows netstat format:
        # Proto LocalAddress ForeignAddress State PID
        if len(parts) < 5:
            continue
        local = parts[1]
        state = parts[3].upper()
        pid_raw = parts[4]
        if state != "LISTENING":
            continue
        if not local.endswith(f":{port}"):
            continue
        try:
            pids.add(int(pid_raw))
        except ValueError:
            continue
    return pids


def _kill_pid(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:
        return


def _health_ok(server_url: str) -> bool:
    try:
        import requests

        r = requests.get(f"{server_url}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _worker_main(
    worker_id: int,
    attempts: int,
    jitter_ms: int,
    out_q: mp.Queue,
    repo_root: str,
) -> None:
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    # Import in child so lifecycle constants are read from child env.
    from muninn.mcp.lifecycle import ensure_server_running

    ok_count = 0
    fail_count = 0
    durations_ms: List[float] = []
    errors: List[str] = []

    for _ in range(attempts):
        t0 = time.perf_counter()
        try:
            ok = bool(ensure_server_running())
        except Exception as exc:
            ok = False
            errors.append(repr(exc))
        dt_ms = (time.perf_counter() - t0) * 1000.0
        durations_ms.append(dt_ms)
        if ok:
            ok_count += 1
        else:
            fail_count += 1
        if jitter_ms > 0:
            time.sleep(random.uniform(0.0, jitter_ms / 1000.0))

    out_q.put(
        {
            "worker_id": worker_id,
            "ok_count": ok_count,
            "fail_count": fail_count,
            "attempts": attempts,
            "max_ms": max(durations_ms) if durations_ms else 0.0,
            "p95_ms": sorted(durations_ms)[int(0.95 * (len(durations_ms) - 1))]
            if durations_ms
            else 0.0,
            "errors": errors[:3],
        }
    )


@dataclass
class StressSummary:
    server_url: str
    data_dir: str
    workers: int
    attempts_per_worker: int
    sample_seconds: float
    samples_taken: int
    max_concurrent_listener_pids: int
    unique_listener_pids_seen: List[int]
    worker_failures: int
    health_ok: bool
    pass_single_owner: bool
    pass_workers_ok: bool
    pass_health: bool
    passed: bool
    manual_launcher_exit_code: int | None = None
    manual_launcher_running_after_wait: bool = False


def main() -> int:
    parser = argparse.ArgumentParser(description="Stress test Muninn single-owner startup behavior")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--jitter-ms", type=int, default=120)
    parser.add_argument("--sample-seconds", type=float, default=8.0)
    parser.add_argument("--sample-interval-ms", type=int, default=200)
    parser.add_argument("--manual-launch", action="store_true")
    parser.add_argument("--manual-wait-seconds", type=float, default=4.0)
    parser.add_argument("--port", type=int, default=0, help="0 = auto-pick free port")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    port = args.port if args.port > 0 else _pick_free_port()
    server_url = f"http://127.0.0.1:{port}"
    data_dir = Path(tempfile.mkdtemp(prefix="muninn_stress_"))
    token = secrets.token_hex(24)

    os.environ["MUNINN_SERVER_URL"] = server_url
    os.environ["MUNINN_PORT"] = str(port)
    os.environ["MUNINN_DATA_DIR"] = str(data_dir)
    os.environ["MUNINN_AUTH_TOKEN"] = token
    os.environ.setdefault("MUNINN_MCP_AUTOSTART_SERVER", "1")
    os.environ.setdefault("MUNINN_MCP_AUTOSTART_ON_LAUNCH", "1")
    os.environ.setdefault("MUNINN_MCP_AUTOSTART_OLLAMA", "0")

    q: mp.Queue = mp.Queue()
    procs: List[mp.Process] = []
    for i in range(args.workers):
        p = mp.Process(
            target=_worker_main,
            args=(i, args.attempts, args.jitter_ms, q, str(repo_root)),
            daemon=False,
        )
        p.start()
        procs.append(p)

    manual_exit_code = None
    manual_running_after_wait = False
    manual_proc = None
    if args.manual_launch:
        manual_proc = subprocess.Popen(
            [sys.executable, "server.py"],
            cwd=str(repo_root),
            env=dict(os.environ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(max(0.1, args.manual_wait_seconds))
        manual_exit_code = manual_proc.poll()
        if manual_exit_code is None:
            manual_running_after_wait = True

    sample_interval = max(0.05, args.sample_interval_ms / 1000.0)
    sample_deadline = time.monotonic() + max(0.5, args.sample_seconds)
    samples = 0
    max_concurrent = 0
    unique_listener_pids: Set[int] = set()
    while time.monotonic() < sample_deadline or any(p.is_alive() for p in procs):
        pids = _listener_pids_for_port(port)
        unique_listener_pids.update(pids)
        max_concurrent = max(max_concurrent, len(pids))
        samples += 1
        time.sleep(sample_interval)

    for p in procs:
        p.join(timeout=10)

    worker_results: List[Dict[str, object]] = []
    while True:
        try:
            worker_results.append(q.get_nowait())
        except Exception:
            break

    worker_failures = int(sum(int(r.get("fail_count", 0)) for r in worker_results))
    health_ok = _health_ok(server_url)

    pass_single_owner = max_concurrent <= 1
    pass_workers_ok = worker_failures == 0 and len(worker_results) == args.workers
    pass_health = health_ok
    passed = pass_single_owner and pass_workers_ok and pass_health

    summary = StressSummary(
        server_url=server_url,
        data_dir=str(data_dir),
        workers=args.workers,
        attempts_per_worker=args.attempts,
        sample_seconds=float(args.sample_seconds),
        samples_taken=samples,
        max_concurrent_listener_pids=max_concurrent,
        unique_listener_pids_seen=sorted(unique_listener_pids),
        worker_failures=worker_failures,
        health_ok=health_ok,
        pass_single_owner=pass_single_owner,
        pass_workers_ok=pass_workers_ok,
        pass_health=pass_health,
        passed=passed,
        manual_launcher_exit_code=manual_exit_code,
        manual_launcher_running_after_wait=manual_running_after_wait,
    )

    print(json.dumps({"summary": asdict(summary), "workers": worker_results}, indent=2))

    # Cleanup isolated spawned backend.
    for pid in _listener_pids_for_port(port):
        _kill_pid(pid)
    if manual_proc is not None and manual_proc.poll() is None:
        manual_proc.terminate()
        try:
            manual_proc.wait(timeout=5)
        except Exception:
            manual_proc.kill()

    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
