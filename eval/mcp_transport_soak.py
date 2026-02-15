from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO


DEFAULT_SERVER_URL = "http://127.0.0.1:9"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _decode_bytes(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = max(0.0, min(1.0, p)) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return float(ordered[lo])
    frac = rank - lo
    return float((ordered[lo] * (1.0 - frac)) + (ordered[hi] * frac))


def _latency_stats(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        return {
            "count": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "avg_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
        }
    return {
        "count": float(len(samples_ms)),
        "min_ms": float(min(samples_ms)),
        "max_ms": float(max(samples_ms)),
        "avg_ms": float(sum(samples_ms) / len(samples_ms)),
        "p50_ms": _percentile(samples_ms, 0.50),
        "p95_ms": _percentile(samples_ms, 0.95),
    }


def _report_path(report_dir: Path, run_id: str) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir / f"mcp_transport_soak_{run_id}.json"


def _read_rpc_message(stream: BinaryIO) -> dict[str, Any] | None:
    while True:
        first_line = stream.readline()
        if not first_line:
            return None
        if not first_line.strip():
            continue

        lowered = first_line.lower()
        if lowered.startswith(b"content-length:"):
            try:
                content_length = int(first_line.split(b":", 1)[1].strip())
            except Exception:
                continue

            while True:
                header_line = stream.readline()
                if not header_line:
                    return None
                if header_line in (b"\r\n", b"\n"):
                    break

            payload = stream.read(content_length)
            if not payload or len(payload) != content_length:
                return None
            try:
                parsed = json.loads(_decode_bytes(payload))
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
            continue

        try:
            parsed = json.loads(_decode_bytes(first_line))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed


def _write_rpc_message(stream: BinaryIO, payload: dict[str, Any], transport: str) -> None:
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if transport == "framed":
        frame = (
            f"Content-Length: {len(encoded)}\r\n"
            "Content-Type: application/json\r\n"
            "\r\n"
        ).encode("ascii") + encoded
        stream.write(frame)
    else:
        stream.write(encoded + b"\n")
    stream.flush()


def _inject_malformed_frame(stream: BinaryIO) -> None:
    stream.write(b"Content-Length: nope\r\n\r\n")
    stream.flush()


class _RpcSession:
    def __init__(self, proc: subprocess.Popen[bytes], transport: str):
        self._proc = proc
        self._transport = transport
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self._pending: dict[str, dict[str, Any]] = {}
        self._write_lock = threading.Lock()
        self._reader_done = threading.Event()
        self._reader_error: str | None = None
        self._reader = threading.Thread(target=self._reader_loop, daemon=True, name="mcp-soak-reader")
        self._reader.start()

    def _reader_loop(self) -> None:
        try:
            assert self._proc.stdout is not None
            while True:
                msg = _read_rpc_message(self._proc.stdout)
                if msg is None:
                    return
                self._messages.put(msg)
        except Exception as exc:  # pragma: no cover - defensive guard for runtime diagnostics
            self._reader_error = repr(exc)
        finally:
            self._reader_done.set()

    def send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        with self._write_lock:
            assert self._proc.stdin is not None
            _write_rpc_message(self._proc.stdin, payload, self._transport)

    def call(self, request_id: str, method: str, params: dict[str, Any] | None, timeout_sec: float) -> dict[str, Any]:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params

        with self._write_lock:
            assert self._proc.stdin is not None
            _write_rpc_message(self._proc.stdin, payload, self._transport)

        pending_hit = self._pending.pop(request_id, None)
        if pending_hit is not None:
            return pending_hit

        deadline = time.monotonic() + timeout_sec
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for response id={request_id}")
            try:
                message = self._messages.get(timeout=remaining)
            except queue.Empty as exc:
                if self._reader_error:
                    raise RuntimeError(f"reader failure: {self._reader_error}") from exc
                if self._reader_done.is_set() and self._proc.poll() is not None:
                    raise RuntimeError("wrapper process exited while awaiting response") from exc
                raise TimeoutError(f"timed out waiting for response id={request_id}") from exc

            msg_id = message.get("id")
            if msg_id is None:
                continue
            msg_id_text = str(msg_id)
            if msg_id_text == request_id:
                return message
            self._pending[msg_id_text] = message


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Soak-test MCP wrapper transport resilience under backend outage and malformed input."
    )
    parser.add_argument("--iterations", type=int, default=25, help="Measured request count.")
    parser.add_argument("--warmup-requests", type=int, default=1, help="Unscored warmup requests.")
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument("--transport", choices=("framed", "line"), default="framed")
    parser.add_argument("--server-url", type=str, default=DEFAULT_SERVER_URL)
    parser.add_argument("--failure-threshold", type=int, default=1)
    parser.add_argument("--cooldown-sec", type=float, default=30.0)
    parser.add_argument("--max-p95-ms", type=float, default=400.0)
    parser.add_argument(
        "--inject-malformed-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inject malformed framed header before probe ping (framed transport only).",
    )
    parser.add_argument("--report-dir", type=Path, default=Path("eval/reports/mcp_transport"))
    parser.add_argument("--wrapper", type=Path, default=Path("mcp_wrapper.py"))
    return parser


def _start_wrapper(wrapper_path: Path, env: dict[str, str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, str(wrapper_path)],
        cwd=str(wrapper_path.parent),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def run(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.warmup_requests < 0:
        raise ValueError("--warmup-requests must be non-negative")
    if args.timeout_sec <= 0:
        raise ValueError("--timeout-sec must be positive")
    if args.max_p95_ms <= 0:
        raise ValueError("--max-p95-ms must be positive")

    wrapper_path = args.wrapper.resolve()
    if not wrapper_path.exists():
        raise FileNotFoundError(f"wrapper path not found: {wrapper_path}")

    run_id = _utc_now().strftime("%Y%m%d_%H%M%S")
    report_file = _report_path(args.report_dir, run_id)

    env = dict(os.environ)
    env.update(
        {
            "MUNINN_SERVER_URL": args.server_url,
            "MUNINN_MCP_AUTOSTART_ON_LAUNCH": "0",
            "MUNINN_MCP_AUTOSTART_SERVER": "0",
            "MUNINN_MCP_AUTOSTART_OLLAMA": "0",
            "MUNINN_MCP_BACKEND_FAILURE_THRESHOLD": str(args.failure_threshold),
            "MUNINN_MCP_BACKEND_COOLDOWN_SEC": str(args.cooldown_sec),
        }
    )

    started_at = _utc_now().isoformat()
    proc: subprocess.Popen[bytes] | None = None
    outcome = "pass"
    failures: list[str] = []
    timings_ms: list[float] = []
    error_codes: dict[str, int] = {}
    malformed_probe_ok = True

    try:
        proc = _start_wrapper(wrapper_path, env)
        session = _RpcSession(proc, args.transport)

        initialize = session.call(
            request_id="init-1",
            method="initialize",
            params={"protocolVersion": "2025-11-25", "capabilities": {}},
            timeout_sec=args.timeout_sec,
        )
        if "result" not in initialize:
            failures.append("initialize_missing_result")
        session.send_notification("notifications/initialized", {})

        if args.inject_malformed_frame and args.transport == "framed":
            assert proc.stdin is not None
            _inject_malformed_frame(proc.stdin)
            probe = session.call(
                request_id="probe-ping",
                method="ping",
                params={},
                timeout_sec=args.timeout_sec,
            )
            malformed_probe_ok = "result" in probe and "error" not in probe
            if not malformed_probe_ok:
                failures.append("malformed_frame_probe_failed")

        total_requests = args.warmup_requests + args.iterations
        for idx in range(total_requests):
            req_id = f"soak-{idx}"
            t0 = time.perf_counter()
            try:
                response = session.call(
                    request_id=req_id,
                    method="tools/call",
                    params={
                        "name": "search_memory",
                        "arguments": {"query": f"soak-{idx}", "limit": 1},
                    },
                    timeout_sec=args.timeout_sec,
                )
            except Exception as exc:
                if idx >= args.warmup_requests:
                    measured_idx = idx - args.warmup_requests
                    failures.append(f"iteration_{measured_idx}_exception:{exc}")
                continue

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if idx < args.warmup_requests:
                continue
            measured_idx = idx - args.warmup_requests
            timings_ms.append(elapsed_ms)

            if "error" not in response:
                failures.append(f"iteration_{measured_idx}_missing_error")
                continue
            code = str(response["error"].get("code", "unknown"))
            error_codes[code] = error_codes.get(code, 0) + 1

        stats = _latency_stats(timings_ms)
        if stats["count"] != float(args.iterations):
            failures.append(f"response_count_mismatch:{int(stats['count'])}/{args.iterations}")
        if stats["p95_ms"] > args.max_p95_ms:
            failures.append(f"p95_exceeds_budget:{stats['p95_ms']:.2f}>{args.max_p95_ms:.2f}")
        if "-32603" not in error_codes:
            failures.append("expected_error_code_-32603_not_observed")
        if len(error_codes.keys()) > 1:
            failures.append(f"multiple_error_codes_observed:{sorted(error_codes.keys())}")

        outcome = "pass" if not failures else "fail"
        report = {
            "run_id": run_id,
            "started_at": started_at,
            "completed_at": _utc_now().isoformat(),
            "outcome": outcome,
            "config": {
                "iterations": args.iterations,
                "warmup_requests": args.warmup_requests,
                "timeout_sec": args.timeout_sec,
                "transport": args.transport,
                "server_url": args.server_url,
                "failure_threshold": args.failure_threshold,
                "cooldown_sec": args.cooldown_sec,
                "max_p95_ms": args.max_p95_ms,
                "inject_malformed_frame": bool(args.inject_malformed_frame),
                "wrapper": str(wrapper_path),
            },
            "results": {
                "malformed_probe_ok": malformed_probe_ok,
                "error_codes": error_codes,
                "latency": stats,
                "failures": failures,
            },
        }
        report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if outcome == "pass" else 2
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
