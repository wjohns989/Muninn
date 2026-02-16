from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from eval import mcp_transport_diagnostics as diagnostics


def _log_ts_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]


def test_parse_tool_call_telemetry_line() -> None:
    line = (
        f"{_log_ts_now()} - Muninn - INFO - Tool call telemetry: "
        "name=search_memory id='req-1' outcome=error elapsed_ms=123.4 responses=1 "
        "response_bytes_total=140 response_bytes_max=140 budget_ms=110000.0 remaining_budget_ms=109876.6"
    )
    parsed = diagnostics._parse_tool_call_telemetry(line)
    assert parsed is not None
    assert parsed["name"] == "search_memory"
    assert parsed["outcome"] == "error"
    assert parsed["elapsed_ms"] == 123.4
    assert parsed["response_bytes_total"] == 140
    assert parsed["budget_ms"] == 110000.0


def test_run_builds_diagnostics_bundle(tmp_path: Path) -> None:
    log_path = tmp_path / "mcp_wrapper.log"
    report_dir = tmp_path / "reports"
    output_path = report_dir / "bundle.json"
    report_dir.mkdir(parents=True, exist_ok=True)

    now_line = _log_ts_now()
    log_path.write_text(
        "\n".join(
            [
                (
                    f"{now_line} - Muninn - INFO - Tool call telemetry: "
                    "name=search_memory id='req-1' outcome=error elapsed_ms=91500.0 responses=1 "
                    "response_bytes_total=140 response_bytes_max=140 budget_ms=110000.0 remaining_budget_ms=18500.0"
                ),
                (
                    f"{now_line} - Muninn - WARNING - "
                    "MCP stdio transport closed while sending JSON-RPC message: Broken pipe"
                ),
                (
                    f"{now_line} - Muninn - WARNING - "
                    "Aborting request due to deadline budget exhaustion: Request deadline budget exhausted before backend call."
                ),
            ]
        ),
        encoding="utf-8",
    )

    soak_report = {
        "run_id": "soak-1",
        "completed_at": "2026-02-16T00:00:00+00:00",
        "outcome": "pass",
        "results": {
            "latency": {"p95_ms": 120.0},
            "error_codes": {"-32603": 10},
            "task_result_probe": {"enabled": True, "observed_retryable_nonterminal_error": True},
        },
    }
    closure_report = {
        "run_id": "closure-1",
        "completed_at": "2026-02-16T00:00:00+00:00",
        "results": {
            "closure_ready": True,
            "current_consecutive_pass_streak": 5,
            "criteria": {"streak_target_met": True},
            "telemetry": {"retryable_task_result_error_count": 0},
        },
    }
    (report_dir / "mcp_transport_soak_20260216_000000.json").write_text(
        json.dumps(soak_report), encoding="utf-8"
    )
    (report_dir / "mcp_transport_closure_20260216_000100.json").write_text(
        json.dumps(closure_report), encoding="utf-8"
    )

    exit_code = diagnostics.run(
        [
            "--log-path",
            str(log_path),
            "--report-dir",
            str(report_dir),
            "--lookback-hours",
            "4",
            "--near-timeout-ms",
            "90000",
            "--recent-soak-limit",
            "1",
            "--recent-closure-limit",
            "1",
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 0
    parsed = json.loads(output_path.read_text(encoding="utf-8"))
    incidents = parsed["results"]["incidents"]
    assert incidents["transport_closed_count"] == 1
    assert incidents["deadline_exhaustion_count"] == 1
    assert incidents["near_timeout_count"] == 1
    assert parsed["results"]["recent_reports"]["soak"][0]["run_id"] == "soak-1"
    assert parsed["results"]["recent_reports"]["closure"][0]["run_id"] == "closure-1"


def test_run_enforce_gate_fails_when_threshold_exceeded(tmp_path: Path) -> None:
    log_path = tmp_path / "mcp_wrapper.log"
    report_dir = tmp_path / "reports"
    output_path = report_dir / "bundle.json"
    report_dir.mkdir(parents=True, exist_ok=True)
    now_line = _log_ts_now()
    log_path.write_text(
        (
            f"{now_line} - Muninn - WARNING - "
            "MCP stdio transport closed while sending JSON-RPC message: Broken pipe"
        ),
        encoding="utf-8",
    )

    exit_code = diagnostics.run(
        [
            "--log-path",
            str(log_path),
            "--report-dir",
            str(report_dir),
            "--lookback-hours",
            "4",
            "--max-transport-closed-count",
            "0",
            "--enforce-gate",
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 2
    parsed = json.loads(output_path.read_text(encoding="utf-8"))
    assert parsed["results"]["gate"]["passed"] is False
    assert any(
        str(item).startswith("transport_closed_count_exceeds_limit:")
        for item in parsed["results"]["gate"]["violations"]
    )
