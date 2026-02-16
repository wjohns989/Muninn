from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


TELEMETRY_LINE_RE = re.compile(
    r"Tool call telemetry: "
    r"name=(?P<name>\S+) "
    r"id=(?P<msg_id>.+?) "
    r"outcome=(?P<outcome>\w+) "
    r"elapsed_ms=(?P<elapsed_ms>[0-9.]+) "
    r"responses=(?P<responses>\d+) "
    r"response_bytes_total=(?P<bytes_total>\d+) "
    r"response_bytes_max=(?P<bytes_max>\d+) "
    r"budget_ms=(?P<budget_ms>\S+) "
    r"remaining_budget_ms=(?P<remaining_budget_ms>\S+)"
)
LOG_PREFIX_TS_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - ")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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


def _parse_prefixed_timestamp(line: str) -> datetime | None:
    match = LOG_PREFIX_TS_RE.match(line)
    if not match:
        return None
    ts_text = match.group("ts")
    try:
        naive = datetime.strptime(ts_text, "%Y-%m-%d %H:%M:%S,%f")
    except ValueError:
        return None
    return naive.replace(tzinfo=timezone.utc)


def _parse_tool_call_telemetry(line: str) -> dict[str, Any] | None:
    match = TELEMETRY_LINE_RE.search(line)
    if not match:
        return None

    def _parse_budget(value: str) -> float | None:
        if value == "n/a":
            return None
        try:
            return float(value)
        except ValueError:
            return None

    try:
        elapsed_ms = float(match.group("elapsed_ms"))
        responses = int(match.group("responses"))
        bytes_total = int(match.group("bytes_total"))
        bytes_max = int(match.group("bytes_max"))
    except ValueError:
        return None

    return {
        "name": match.group("name"),
        "outcome": match.group("outcome"),
        "elapsed_ms": elapsed_ms,
        "responses": responses,
        "response_bytes_total": bytes_total,
        "response_bytes_max": bytes_max,
        "budget_ms": _parse_budget(match.group("budget_ms")),
        "remaining_budget_ms": _parse_budget(match.group("remaining_budget_ms")),
    }


def _load_recent_transport_reports(report_dir: Path, prefix: str, limit: int) -> list[dict[str, Any]]:
    pattern = f"{prefix}_*.json"
    paths = sorted(report_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    summaries: list[dict[str, Any]] = []
    for path in paths[: max(0, limit)]:
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        summary: dict[str, Any] = {
            "path": str(path),
            "run_id": parsed.get("run_id"),
            "completed_at": parsed.get("completed_at"),
            "outcome": parsed.get("outcome"),
        }
        if prefix == "mcp_transport_soak":
            results = parsed.get("results", {}) if isinstance(parsed.get("results"), dict) else {}
            latency = results.get("latency", {}) if isinstance(results.get("latency"), dict) else {}
            summary["p95_ms"] = latency.get("p95_ms")
            summary["error_codes"] = results.get("error_codes", {})
            summary["task_result_probe"] = results.get("task_result_probe")
        elif prefix == "mcp_transport_closure":
            results = parsed.get("results", {}) if isinstance(parsed.get("results"), dict) else {}
            summary["closure_ready"] = results.get("closure_ready")
            summary["streak"] = results.get("current_consecutive_pass_streak")
            summary["criteria"] = results.get("criteria", {})
            summary["telemetry"] = results.get("telemetry", {})
        summaries.append(summary)
    return summaries


def _report_path(report_dir: Path, run_id: str) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir / f"mcp_transport_diagnostics_{run_id}.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic transport diagnostics bundle from wrapper logs and recent transport reports."
    )
    parser.add_argument("--log-path", type=Path, default=Path("mcp_wrapper.log"))
    parser.add_argument("--report-dir", type=Path, default=Path("eval/reports/mcp_transport"))
    parser.add_argument("--lookback-hours", type=float, default=24.0)
    parser.add_argument("--near-timeout-ms", type=float, default=90000.0)
    parser.add_argument("--recent-soak-limit", type=int, default=5)
    parser.add_argument("--recent-closure-limit", type=int, default=3)
    parser.add_argument("--max-transport-closed-count", type=int, default=0)
    parser.add_argument("--max-deadline-exhaustion-count", type=int, default=0)
    parser.add_argument("--max-near-timeout-count", type=int, default=0)
    parser.add_argument(
        "--enforce-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Return non-zero exit status when diagnostics gate thresholds are exceeded.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def run(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.lookback_hours <= 0:
        raise ValueError("--lookback-hours must be positive")
    if args.near_timeout_ms < 0:
        raise ValueError("--near-timeout-ms must be non-negative")
    if args.recent_soak_limit < 0:
        raise ValueError("--recent-soak-limit must be non-negative")
    if args.recent_closure_limit < 0:
        raise ValueError("--recent-closure-limit must be non-negative")
    if args.max_transport_closed_count < 0:
        raise ValueError("--max-transport-closed-count must be non-negative")
    if args.max_deadline_exhaustion_count < 0:
        raise ValueError("--max-deadline-exhaustion-count must be non-negative")
    if args.max_near_timeout_count < 0:
        raise ValueError("--max-near-timeout-count must be non-negative")

    now = _utc_now()
    window_start = now - timedelta(hours=float(args.lookback_hours))
    run_id = now.strftime("%Y%m%d_%H%M%S")
    output_path = args.output.resolve() if args.output is not None else _report_path(args.report_dir, run_id)

    tool_latencies: dict[str, list[float]] = {}
    tool_response_bytes: dict[str, list[int]] = {}
    tool_outcomes: dict[str, dict[str, int]] = {}
    near_timeout_events: list[dict[str, Any]] = []
    transport_closed_count = 0
    deadline_exhaustion_count = 0
    startup_budget_skip_count = 0
    total_window_lines = 0
    telemetry_match_count = 0

    if args.log_path.exists():
        with args.log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                ts = _parse_prefixed_timestamp(raw_line)
                if ts is None or ts < window_start:
                    continue
                total_window_lines += 1
                if "MCP stdio transport closed while sending JSON-RPC message" in raw_line:
                    transport_closed_count += 1
                if "Aborting request due to deadline budget exhaustion" in raw_line:
                    deadline_exhaustion_count += 1
                if "Skipping preflight startup recovery due to low remaining deadline budget" in raw_line:
                    startup_budget_skip_count += 1

                telemetry = _parse_tool_call_telemetry(raw_line)
                if telemetry is None:
                    continue
                telemetry_match_count += 1
                tool_name = str(telemetry["name"])
                elapsed_ms = float(telemetry["elapsed_ms"])
                bytes_total = int(telemetry["response_bytes_total"])
                outcome = str(telemetry["outcome"])

                tool_latencies.setdefault(tool_name, []).append(elapsed_ms)
                tool_response_bytes.setdefault(tool_name, []).append(bytes_total)
                tool_outcomes.setdefault(tool_name, {})
                tool_outcomes[tool_name][outcome] = tool_outcomes[tool_name].get(outcome, 0) + 1

                if elapsed_ms >= float(args.near_timeout_ms):
                    near_timeout_events.append(
                        {
                            "tool": tool_name,
                            "elapsed_ms": elapsed_ms,
                            "outcome": outcome,
                            "budget_ms": telemetry.get("budget_ms"),
                            "remaining_budget_ms": telemetry.get("remaining_budget_ms"),
                        }
                    )

    per_tool: dict[str, Any] = {}
    for tool_name, latencies in tool_latencies.items():
        per_tool[tool_name] = {
            "latency": _latency_stats(latencies),
            "response_bytes_total_p95": _percentile([float(v) for v in tool_response_bytes.get(tool_name, [])], 0.95),
            "outcomes": tool_outcomes.get(tool_name, {}),
        }

    recent_soak = _load_recent_transport_reports(args.report_dir, "mcp_transport_soak", args.recent_soak_limit)
    recent_closure = _load_recent_transport_reports(
        args.report_dir, "mcp_transport_closure", args.recent_closure_limit
    )

    root_causes: list[str] = []
    if transport_closed_count > 0:
        root_causes.append("transport_write_failures_detected")
    if deadline_exhaustion_count > 0:
        root_causes.append("request_deadline_exhaustion_detected")
    if near_timeout_events:
        root_causes.append("near_timeout_tool_calls_detected")
    latest_closure_ready = None
    if recent_closure:
        latest = recent_closure[0]
        latest_closure_ready = bool(latest.get("closure_ready"))
        if latest_closure_ready and not root_causes:
            root_causes.append("no_wrapper_level_failure_signals_in_window")

    report = {
        "run_id": run_id,
        "generated_at": now.isoformat(),
        "config": {
            "log_path": str(args.log_path.resolve()),
            "report_dir": str(args.report_dir.resolve()),
            "lookback_hours": args.lookback_hours,
            "near_timeout_ms": args.near_timeout_ms,
            "recent_soak_limit": args.recent_soak_limit,
            "recent_closure_limit": args.recent_closure_limit,
            "max_transport_closed_count": args.max_transport_closed_count,
            "max_deadline_exhaustion_count": args.max_deadline_exhaustion_count,
            "max_near_timeout_count": args.max_near_timeout_count,
            "enforce_gate": bool(args.enforce_gate),
            "output": str(output_path),
        },
        "results": {
            "log_window": {
                "start_at": window_start.isoformat(),
                "end_at": now.isoformat(),
                "log_exists": args.log_path.exists(),
                "window_line_count": total_window_lines,
                "telemetry_match_count": telemetry_match_count,
            },
            "incidents": {
                "transport_closed_count": transport_closed_count,
                "deadline_exhaustion_count": deadline_exhaustion_count,
                "startup_budget_skip_count": startup_budget_skip_count,
                "near_timeout_count": len(near_timeout_events),
            },
            "telemetry": {
                "per_tool": per_tool,
                "near_timeout_events": near_timeout_events,
            },
            "recent_reports": {
                "soak": recent_soak,
                "closure": recent_closure,
            },
            "blocker_signals": {
                "latest_closure_ready": latest_closure_ready,
                "suspected_root_causes": root_causes,
            },
        },
    }
    gate_violations: list[str] = []
    if transport_closed_count > args.max_transport_closed_count:
        gate_violations.append(
            f"transport_closed_count_exceeds_limit:{transport_closed_count}>{args.max_transport_closed_count}"
        )
    if deadline_exhaustion_count > args.max_deadline_exhaustion_count:
        gate_violations.append(
            "deadline_exhaustion_count_exceeds_limit:"
            f"{deadline_exhaustion_count}>{args.max_deadline_exhaustion_count}"
        )
    if len(near_timeout_events) > args.max_near_timeout_count:
        gate_violations.append(
            f"near_timeout_count_exceeds_limit:{len(near_timeout_events)}>{args.max_near_timeout_count}"
        )
    gate_passed = len(gate_violations) == 0
    report["results"]["gate"] = {
        "passed": gate_passed,
        "violations": gate_violations,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.enforce_gate and not gate_passed:
        return 2
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
