from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_json_reports(report_dir: Path, prefix: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in sorted(report_dir.glob(f"{prefix}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        completed_at = _parse_dt(parsed.get("completed_at"))
        if completed_at is None:
            completed_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        items.append(
            {
                "path": str(path),
                "completed_at": completed_at,
                "parsed": parsed,
            }
        )
    return items


def _report_path(report_dir: Path, run_id: str) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir / f"mcp_transport_blocker_decision_{run_id}.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate transport blocker closure readiness from replay and closure artifacts."
        )
    )
    parser.add_argument("--report-dir", type=Path, default=Path("eval/reports/mcp_transport"))
    parser.add_argument("--lookback-hours", type=float, default=48.0)
    parser.add_argument("--min-replay-runs", type=int, default=3)
    parser.add_argument("--max-replay-signature-count", type=int, default=0)
    parser.add_argument(
        "--require-replay-provenance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require replay reports to include log path existence and SHA-256 provenance.",
    )
    parser.add_argument(
        "--replay-provenance-policy",
        choices=("all", "latest_min"),
        default="all",
        help=(
            "How to evaluate replay provenance when required: "
            "'all' validates every replay report in-window, "
            "'latest_min' validates only the latest required replay evidence set."
        ),
    )
    parser.add_argument("--min-closure-runs", type=int, default=1)
    parser.add_argument(
        "--require-latest-closure-ready",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-latest-probe-criterion",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enforce-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Return non-zero exit code when blocker closure criteria are not met.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def run(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.lookback_hours <= 0:
        raise ValueError("--lookback-hours must be positive")
    if args.min_replay_runs < 0:
        raise ValueError("--min-replay-runs must be non-negative")
    if args.max_replay_signature_count < 0:
        raise ValueError("--max-replay-signature-count must be non-negative")
    if args.min_closure_runs < 0:
        raise ValueError("--min-closure-runs must be non-negative")

    now = _utc_now()
    window_start = now - timedelta(hours=float(args.lookback_hours))
    run_id = now.strftime("%Y%m%d_%H%M%S")
    output_path = args.output.resolve() if args.output is not None else _report_path(args.report_dir, run_id)

    replay_reports_all = _load_json_reports(args.report_dir, "mcp_transport_incident_replay")
    closure_reports_all = _load_json_reports(args.report_dir, "mcp_transport_closure")

    replay_reports = [item for item in replay_reports_all if item["completed_at"] >= window_start]
    closure_reports = [item for item in closure_reports_all if item["completed_at"] >= window_start]

    replay_summaries: list[dict[str, Any]] = []
    replay_signature_budget_met = True
    replay_diagnostics_return_code_ok = True

    for item in replay_reports:
        parsed = item["parsed"]
        results = parsed.get("results") if isinstance(parsed.get("results"), dict) else {}
        scan = results.get("scan") if isinstance(results.get("scan"), dict) else {}
        diagnostics = results.get("diagnostics") if isinstance(results.get("diagnostics"), dict) else {}
        log_file = scan.get("log_file") if isinstance(scan.get("log_file"), dict) else {}

        signature_count = int(scan.get("total_signature_count", 0) or 0)
        diag_return_code = diagnostics.get("return_code")
        if signature_count > args.max_replay_signature_count:
            replay_signature_budget_met = False
        if diagnostics.get("executed") is True and diag_return_code not in (0, None):
            replay_diagnostics_return_code_ok = False

        provenance_ok = bool(
            log_file.get("exists") is True
            and isinstance(log_file.get("sha256"), str)
            and log_file.get("sha256")
        )

        replay_summaries.append(
            {
                "path": item["path"],
                "run_id": parsed.get("run_id"),
                "completed_at": item["completed_at"].isoformat(),
                "signature_count": signature_count,
                "triggered": bool(results.get("triggered", False)),
                "diagnostics_executed": bool(diagnostics.get("executed", False)),
                "diagnostics_return_code": diag_return_code,
                "provenance_ok": provenance_ok,
            }
        )

    replay_provenance_required_count = 0
    replay_provenance_evaluated_count = 0
    replay_provenance_passing_count = 0
    replay_provenance_selected_paths: list[str] = []
    replay_provenance_met = True
    if args.require_replay_provenance:
        selected = replay_summaries
        if args.replay_provenance_policy == "latest_min":
            replay_provenance_required_count = max(int(args.min_replay_runs), 1)
            selected = replay_summaries[:replay_provenance_required_count]
        else:
            replay_provenance_required_count = len(selected)

        replay_provenance_selected_paths = [str(item.get("path", "")) for item in selected]
        replay_provenance_evaluated_count = len(selected)
        replay_provenance_passing_count = sum(
            1 for item in selected if bool(item.get("provenance_ok", False))
        )
        replay_provenance_met = (
            replay_provenance_evaluated_count >= replay_provenance_required_count
            and replay_provenance_passing_count >= replay_provenance_required_count
        )

    closure_summaries: list[dict[str, Any]] = []
    latest_closure_ready = None
    latest_probe_criterion = None

    for item in closure_reports:
        parsed = item["parsed"]
        results = parsed.get("results") if isinstance(parsed.get("results"), dict) else {}
        criteria = results.get("criteria") if isinstance(results.get("criteria"), dict) else {}
        closure_ready = bool(results.get("closure_ready", False))
        probe_ok = criteria.get("nonterminal_task_result_probe_met")
        closure_summaries.append(
            {
                "path": item["path"],
                "run_id": parsed.get("run_id"),
                "completed_at": item["completed_at"].isoformat(),
                "closure_ready": closure_ready,
                "nonterminal_task_result_probe_met": probe_ok,
            }
        )

    if closure_summaries:
        latest = closure_summaries[0]
        latest_closure_ready = bool(latest.get("closure_ready", False))
        latest_probe_criterion = latest.get("nonterminal_task_result_probe_met")

    criteria = {
        "replay_run_count_meets_min": len(replay_reports) >= args.min_replay_runs,
        "replay_signature_budget_met": replay_signature_budget_met,
        "replay_diagnostics_return_code_ok": replay_diagnostics_return_code_ok,
        "replay_provenance_met": replay_provenance_met,
        "closure_run_count_meets_min": len(closure_reports) >= args.min_closure_runs,
        "latest_closure_ready_met": (not args.require_latest_closure_ready)
        or (latest_closure_ready is True),
        "latest_probe_criterion_met": (not args.require_latest_probe_criterion)
        or (latest_probe_criterion is True),
    }

    violations = [name for name, ok in criteria.items() if not bool(ok)]
    blocker_closure_ready = len(violations) == 0

    report = {
        "run_id": run_id,
        "completed_at": now.isoformat(),
        "inputs": {
            "report_dir": str(args.report_dir),
            "lookback_hours": float(args.lookback_hours),
            "window_start": window_start.isoformat(),
            "window_end": now.isoformat(),
            "min_replay_runs": args.min_replay_runs,
            "max_replay_signature_count": args.max_replay_signature_count,
            "require_replay_provenance": bool(args.require_replay_provenance),
            "replay_provenance_policy": args.replay_provenance_policy,
            "min_closure_runs": args.min_closure_runs,
            "require_latest_closure_ready": bool(args.require_latest_closure_ready),
            "require_latest_probe_criterion": bool(args.require_latest_probe_criterion),
            "enforce_gate": bool(args.enforce_gate),
            "output": str(output_path),
        },
        "results": {
            "blocker_closure_ready": blocker_closure_ready,
            "criteria": criteria,
            "violations": violations,
            "replay_provenance": {
                "required": bool(args.require_replay_provenance),
                "policy": args.replay_provenance_policy,
                "required_count": replay_provenance_required_count,
                "evaluated_count": replay_provenance_evaluated_count,
                "passing_count": replay_provenance_passing_count,
                "selected_paths": replay_provenance_selected_paths,
            },
            "replay_reports_analyzed": replay_summaries,
            "closure_reports_analyzed": closure_summaries,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if args.enforce_gate and not blocker_closure_ready:
        return 2
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
