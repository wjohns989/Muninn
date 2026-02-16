from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from eval import mcp_transport_blocker_decision as decision


def _write_replay_report(path: Path, *, run_id: str, completed_at: str, signature_count: int, diag_rc: int | None, sha256: str | None) -> None:
    payload = {
        "run_id": run_id,
        "completed_at": completed_at,
        "results": {
            "scan": {
                "total_signature_count": signature_count,
                "log_file": {
                    "path": "mcp_wrapper.log",
                    "exists": True,
                    "size_bytes": 123,
                    "modified_at": completed_at,
                    "sha256": sha256,
                },
            },
            "triggered": False,
            "diagnostics": {
                "executed": False if diag_rc is None else True,
                "return_code": diag_rc,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_closure_report(path: Path, *, run_id: str, completed_at: str, closure_ready: bool, probe_ok: bool) -> None:
    payload = {
        "run_id": run_id,
        "completed_at": completed_at,
        "results": {
            "closure_ready": closure_ready,
            "criteria": {
                "nonterminal_task_result_probe_met": probe_ok,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_passes_with_replay_and_closure_evidence(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    _write_replay_report(
        report_dir / "mcp_transport_incident_replay_1.json",
        run_id="replay-1",
        completed_at=now,
        signature_count=0,
        diag_rc=None,
        sha256="abc123",
    )
    _write_closure_report(
        report_dir / "mcp_transport_closure_1.json",
        run_id="closure-1",
        completed_at=now,
        closure_ready=True,
        probe_ok=True,
    )

    output = report_dir / "decision.json"
    exit_code = decision.run(
        [
            "--report-dir",
            str(report_dir),
            "--min-replay-runs",
            "1",
            "--min-closure-runs",
            "1",
            "--require-replay-provenance",
            "--enforce-gate",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed["results"]["blocker_closure_ready"] is True
    assert parsed["results"]["violations"] == []


def test_run_fails_on_replay_signature_budget(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    _write_replay_report(
        report_dir / "mcp_transport_incident_replay_1.json",
        run_id="replay-1",
        completed_at=now,
        signature_count=2,
        diag_rc=None,
        sha256="abc123",
    )
    _write_closure_report(
        report_dir / "mcp_transport_closure_1.json",
        run_id="closure-1",
        completed_at=now,
        closure_ready=True,
        probe_ok=True,
    )

    output = report_dir / "decision.json"
    exit_code = decision.run(
        [
            "--report-dir",
            str(report_dir),
            "--min-replay-runs",
            "1",
            "--min-closure-runs",
            "1",
            "--max-replay-signature-count",
            "0",
            "--enforce-gate",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 2
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert "replay_signature_budget_met" in parsed["results"]["violations"]


def test_run_fails_on_missing_replay_provenance_when_required(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    _write_replay_report(
        report_dir / "mcp_transport_incident_replay_1.json",
        run_id="replay-1",
        completed_at=now,
        signature_count=0,
        diag_rc=None,
        sha256=None,
    )
    _write_closure_report(
        report_dir / "mcp_transport_closure_1.json",
        run_id="closure-1",
        completed_at=now,
        closure_ready=True,
        probe_ok=True,
    )

    output = report_dir / "decision.json"
    exit_code = decision.run(
        [
            "--report-dir",
            str(report_dir),
            "--min-replay-runs",
            "1",
            "--min-closure-runs",
            "1",
            "--require-replay-provenance",
            "--enforce-gate",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 2
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert "replay_provenance_met" in parsed["results"]["violations"]
