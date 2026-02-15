from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _closure_report_path(report_dir: Path, run_id: str) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir / f"mcp_transport_closure_{run_id}.json"


def _parse_transports(raw_value: str) -> list[str]:
    allowed = {"framed", "line"}
    tokens = [token.strip() for token in raw_value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("--transports must include at least one value")
    parsed: list[str] = []
    for token in tokens:
        if token not in allowed:
            raise ValueError(f"Unsupported transport '{token}'. Expected one of {sorted(allowed)}")
        if token not in parsed:
            parsed.append(token)
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic MCP transport closure campaign against blocker criteria."
    )
    parser.add_argument("--streak-target", type=int, default=30)
    parser.add_argument("--max-campaign-runs", type=int, default=60)
    parser.add_argument("--transports", type=str, default="framed,line")
    parser.add_argument("--min-p95-compliance-ratio", type=float, default=0.95)
    parser.add_argument("--soak-iterations", type=int, default=25)
    parser.add_argument("--soak-warmup-requests", type=int, default=2)
    parser.add_argument("--soak-timeout-sec", type=float, default=15.0)
    parser.add_argument("--soak-max-p95-ms", type=float, default=5000.0)
    parser.add_argument("--soak-server-url", type=str, default="http://127.0.0.1:1")
    parser.add_argument("--soak-failure-threshold", type=int, default=1)
    parser.add_argument("--soak-cooldown-sec", type=float, default=30.0)
    parser.add_argument(
        "--inject-malformed-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--open-wrapper-defects",
        type=int,
        default=0,
        help="Number of unresolved wrapper defects currently linked to transport timeout behavior.",
    )
    parser.add_argument(
        "--unresolved-transport-regressions",
        type=int,
        default=0,
        help="Number of unresolved transport regressions in the observation window.",
    )
    parser.add_argument(
        "--unclassified-failures",
        type=int,
        default=0,
        help="Number of transport failures not yet classified to host/environment or resolved.",
    )
    parser.add_argument("--report-dir", type=Path, default=Path("eval/reports/mcp_transport"))
    parser.add_argument("--wrapper", type=Path, default=Path("mcp_wrapper.py"))
    return parser


def _build_soak_command(args: argparse.Namespace, transport: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "eval.mcp_transport_soak",
        "--iterations",
        str(args.soak_iterations),
        "--warmup-requests",
        str(args.soak_warmup_requests),
        "--timeout-sec",
        str(args.soak_timeout_sec),
        "--transport",
        transport,
        "--server-url",
        args.soak_server_url,
        "--failure-threshold",
        str(args.soak_failure_threshold),
        "--cooldown-sec",
        str(args.soak_cooldown_sec),
        "--max-p95-ms",
        str(args.soak_max_p95_ms),
        "--report-dir",
        str(args.report_dir),
        "--wrapper",
        str(args.wrapper),
    ]
    command.append("--inject-malformed-frame" if args.inject_malformed_frame else "--no-inject-malformed-frame")
    return command


def _decode_output(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _run_soak(command: list[str]) -> dict[str, Any]:
    proc = subprocess.run(command, capture_output=True, check=False)
    stdout_text = _decode_output(proc.stdout).strip()
    stderr_text = _decode_output(proc.stderr).strip()
    parsed_report: dict[str, Any] | None = None
    if stdout_text:
        try:
            parsed_report = json.loads(stdout_text)
        except json.JSONDecodeError:
            parsed_report = None
    return {
        "returncode": int(proc.returncode),
        "stdout": stdout_text,
        "stderr": stderr_text,
        "report": parsed_report,
    }


def _transport_attempt_summary(transport: str, attempt: dict[str, Any], max_p95_ms: float) -> dict[str, Any]:
    report = attempt.get("report") if isinstance(attempt.get("report"), dict) else None
    report_outcome = report.get("outcome") if report else "error"
    latency = report.get("results", {}).get("latency", {}) if report else {}
    p95 = float(latency.get("p95_ms", 0.0) or 0.0)
    p95_compliant = bool(report_outcome == "pass" and p95 <= max_p95_ms)
    report_failures = report.get("results", {}).get("failures", []) if report else []
    failures: list[str] = []
    if isinstance(report_failures, list):
        failures.extend(str(item) for item in report_failures)
    if report is None and attempt.get("stderr"):
        failures.append(f"stderr:{attempt['stderr']}")
    if report is None and attempt.get("stdout"):
        failures.append("invalid_soak_stdout_json")
    if attempt.get("returncode", 0) not in (0, 2):
        failures.append(f"soak_process_exit:{attempt.get('returncode')}")

    return {
        "transport": transport,
        "returncode": int(attempt.get("returncode", 1)),
        "outcome": report_outcome,
        "report_run_id": report.get("run_id") if report else None,
        "p95_ms": p95,
        "p95_compliant": p95_compliant,
        "failures": failures,
    }


def _evaluate_campaign(
    *,
    campaign_runs: list[dict[str, Any]],
    streak_target: int,
    min_p95_compliance_ratio: float,
    unresolved_transport_regressions: int,
    open_wrapper_defects: int,
    unclassified_failures: int,
) -> dict[str, Any]:
    streak = 0
    for run in reversed(campaign_runs):
        if run.get("pass"):
            streak += 1
            continue
        break

    streak_target_met = streak >= streak_target
    if streak_target_met:
        window = campaign_runs[-streak_target:]
    else:
        window = campaign_runs[-min(streak_target, len(campaign_runs)) :] if campaign_runs else []

    window_size = len(window)
    window_pass_count = sum(1 for run in window if run.get("pass"))
    window_p95_count = sum(1 for run in window if run.get("p95_compliant"))
    window_p95_ratio = (window_p95_count / window_size) if window_size else 0.0

    criteria = {
        "streak_target_met": streak_target_met,
        "no_regressions_in_window": bool(window_size == streak_target and window_pass_count == window_size),
        "p95_compliance_met": bool(
            window_size == streak_target and window_p95_ratio >= min_p95_compliance_ratio
        ),
        "no_unresolved_transport_regressions": unresolved_transport_regressions <= 0,
        "no_open_wrapper_defects": open_wrapper_defects <= 0,
        "no_unclassified_failures": unclassified_failures <= 0,
    }
    closure_ready = all(criteria.values())

    return {
        "closure_ready": closure_ready,
        "current_consecutive_pass_streak": streak,
        "window_size": window_size,
        "window_pass_count": window_pass_count,
        "window_p95_compliance_ratio": window_p95_ratio,
        "criteria": criteria,
    }


def run(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.streak_target <= 0:
        raise ValueError("--streak-target must be positive")
    if args.max_campaign_runs < args.streak_target:
        raise ValueError("--max-campaign-runs must be >= --streak-target")
    if args.soak_iterations <= 0:
        raise ValueError("--soak-iterations must be positive")
    if args.soak_warmup_requests < 0:
        raise ValueError("--soak-warmup-requests must be non-negative")
    if args.soak_timeout_sec <= 0:
        raise ValueError("--soak-timeout-sec must be positive")
    if args.soak_max_p95_ms <= 0:
        raise ValueError("--soak-max-p95-ms must be positive")
    if not 0.0 <= args.min_p95_compliance_ratio <= 1.0:
        raise ValueError("--min-p95-compliance-ratio must be in [0,1]")
    transports = _parse_transports(args.transports)

    run_id = _utc_now().strftime("%Y%m%d_%H%M%S")
    report_file = _closure_report_path(args.report_dir, run_id)
    campaign_runs: list[dict[str, Any]] = []
    started_at = _utc_now().isoformat()

    for idx in range(1, args.max_campaign_runs + 1):
        per_transport: list[dict[str, Any]] = []
        for transport in transports:
            command = _build_soak_command(args, transport)
            attempt = _run_soak(command)
            per_transport.append(
                _transport_attempt_summary(
                    transport=transport,
                    attempt=attempt,
                    max_p95_ms=args.soak_max_p95_ms,
                )
            )

        run_pass = all(item["outcome"] == "pass" for item in per_transport)
        run_p95_compliant = all(item["p95_compliant"] for item in per_transport)
        run_failures: list[str] = []
        for item in per_transport:
            run_failures.extend(f"{item['transport']}:{failure}" for failure in item["failures"])
        campaign_runs.append(
            {
                "index": idx,
                "pass": run_pass,
                "p95_compliant": run_p95_compliant,
                "transports": per_transport,
                "failures": run_failures,
            }
        )
        evaluation = _evaluate_campaign(
            campaign_runs=campaign_runs,
            streak_target=args.streak_target,
            min_p95_compliance_ratio=args.min_p95_compliance_ratio,
            unresolved_transport_regressions=args.unresolved_transport_regressions,
            open_wrapper_defects=args.open_wrapper_defects,
            unclassified_failures=args.unclassified_failures,
        )
        if evaluation["criteria"]["streak_target_met"]:
            break

    evaluation = _evaluate_campaign(
        campaign_runs=campaign_runs,
        streak_target=args.streak_target,
        min_p95_compliance_ratio=args.min_p95_compliance_ratio,
        unresolved_transport_regressions=args.unresolved_transport_regressions,
        open_wrapper_defects=args.open_wrapper_defects,
        unclassified_failures=args.unclassified_failures,
    )

    report = {
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": _utc_now().isoformat(),
        "config": {
            "streak_target": args.streak_target,
            "max_campaign_runs": args.max_campaign_runs,
            "transports": transports,
            "min_p95_compliance_ratio": args.min_p95_compliance_ratio,
            "soak_iterations": args.soak_iterations,
            "soak_warmup_requests": args.soak_warmup_requests,
            "soak_timeout_sec": args.soak_timeout_sec,
            "soak_max_p95_ms": args.soak_max_p95_ms,
            "soak_server_url": args.soak_server_url,
            "soak_failure_threshold": args.soak_failure_threshold,
            "soak_cooldown_sec": args.soak_cooldown_sec,
            "inject_malformed_frame": bool(args.inject_malformed_frame),
            "wrapper": str(args.wrapper),
            "open_wrapper_defects": args.open_wrapper_defects,
            "unresolved_transport_regressions": args.unresolved_transport_regressions,
            "unclassified_failures": args.unclassified_failures,
        },
        "results": {
            **evaluation,
            "attempted_campaign_runs": len(campaign_runs),
            "campaign_runs": campaign_runs,
        },
    }
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["results"]["closure_ready"] else 2


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
