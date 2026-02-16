from __future__ import annotations

import json
import os
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


def test_run_passes_with_latest_min_provenance_policy(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    oldest = now.replace(microsecond=0).isoformat()
    middle = now.replace(microsecond=1).isoformat()
    latest = now.replace(microsecond=2).isoformat()

    _write_replay_report(
        report_dir / "mcp_transport_incident_replay_1.json",
        run_id="replay-legacy-missing-provenance",
        completed_at=oldest,
        signature_count=0,
        diag_rc=None,
        sha256=None,
    )
    _write_replay_report(
        report_dir / "mcp_transport_incident_replay_2.json",
        run_id="replay-middle-valid",
        completed_at=middle,
        signature_count=0,
        diag_rc=None,
        sha256="bbb222",
    )
    _write_replay_report(
        report_dir / "mcp_transport_incident_replay_3.json",
        run_id="replay-latest-valid",
        completed_at=latest,
        signature_count=0,
        diag_rc=None,
        sha256="ccc333",
    )
    # Ensure deterministic replay ordering: utility sorts by file mtime descending.
    oldest_path = report_dir / "mcp_transport_incident_replay_1.json"
    middle_path = report_dir / "mcp_transport_incident_replay_2.json"
    latest_path = report_dir / "mcp_transport_incident_replay_3.json"
    os.utime(oldest_path, (now.timestamp() - 3, now.timestamp() - 3))
    os.utime(middle_path, (now.timestamp() - 2, now.timestamp() - 2))
    os.utime(latest_path, (now.timestamp() - 1, now.timestamp() - 1))

    _write_closure_report(
        report_dir / "mcp_transport_closure_1.json",
        run_id="closure-1",
        completed_at=latest,
        closure_ready=True,
        probe_ok=True,
    )

    output = report_dir / "decision.json"
    exit_code = decision.run(
        [
            "--report-dir",
            str(report_dir),
            "--min-replay-runs",
            "2",
            "--min-closure-runs",
            "1",
            "--require-replay-provenance",
            "--replay-provenance-policy",
            "latest_min",
            "--enforce-gate",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed["results"]["blocker_closure_ready"] is True
    provenance = parsed["results"]["replay_provenance"]
    assert provenance["policy"] == "latest_min"
    assert provenance["required_count"] == 2
    assert provenance["passing_count"] == 2
