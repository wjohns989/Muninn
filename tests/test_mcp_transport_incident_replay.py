from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from eval import mcp_transport_incident_replay as replay


def _log_ts_now() -> str:
    return replay._utc_now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]


def test_run_skips_diagnostics_when_no_signatures(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "mcp_wrapper.log"
    report_dir = tmp_path / "reports"
    output_path = report_dir / "replay.json"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"{_log_ts_now()} - Muninn - INFO - Tool call telemetry: name=search_memory id=1 outcome=ok elapsed_ms=120 responses=1 response_bytes_total=42 response_bytes_max=42 budget_ms=1000 remaining_budget_ms=880",
        encoding="utf-8",
    )

    def _unexpected_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("diagnostics command should not execute when no signatures match")

    monkeypatch.setattr(replay.subprocess, "run", _unexpected_run)

    exit_code = replay.run(
        [
            "--log-path",
            str(log_path),
            "--report-dir",
            str(report_dir),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    parsed = json.loads(output_path.read_text(encoding="utf-8"))
    assert parsed["results"]["triggered"] is False
    assert parsed["results"]["scan"]["total_signature_count"] == 0
    assert parsed["results"]["diagnostics"]["executed"] is False


def test_run_triggers_diagnostics_on_signature(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "mcp_wrapper.log"
    report_dir = tmp_path / "reports"
    output_path = report_dir / "replay.json"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"{_log_ts_now()} - Muninn - WARNING - MCP stdio transport closed while sending JSON-RPC message: Broken pipe",
        encoding="utf-8",
    )

    diagnostics_payload = {
        "inputs": {"output": str(tmp_path / "diagnostics.json")},
        "results": {"gate": {"passed": True, "violations": []}},
    }

    def _fake_run(tokens, check, shell, capture_output):  # type: ignore[no-untyped-def]
        assert check is False
        assert shell is False
        assert capture_output is True
        return SimpleNamespace(returncode=0, stdout=json.dumps(diagnostics_payload).encode("utf-8"), stderr=b"")

    monkeypatch.setattr(replay.subprocess, "run", _fake_run)

    exit_code = replay.run(
        [
            "--log-path",
            str(log_path),
            "--report-dir",
            str(report_dir),
            "--diagnostics-command",
            'python -m eval.mcp_transport_diagnostics --lookback-hours 24',
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    parsed = json.loads(output_path.read_text(encoding="utf-8"))
    assert parsed["results"]["triggered"] is True
    assert parsed["results"]["scan"]["total_signature_count"] == 1
    assert parsed["results"]["diagnostics"]["executed"] is True
    assert parsed["results"]["diagnostics"]["return_code"] == 0
    assert parsed["results"]["diagnostics"]["artifact_path"] == diagnostics_payload["inputs"]["output"]


def test_run_returns_nonzero_when_diagnostics_fails(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "mcp_wrapper.log"
    report_dir = tmp_path / "reports"
    output_path = report_dir / "replay.json"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"{_log_ts_now()} - Muninn - WARNING - MCP stdio transport closed while sending JSON-RPC message: Broken pipe",
        encoding="utf-8",
    )

    def _fake_run(tokens, check, shell, capture_output):  # type: ignore[no-untyped-def]
        return SimpleNamespace(returncode=7, stdout=b"", stderr=b"command failed")

    monkeypatch.setattr(replay.subprocess, "run", _fake_run)

    exit_code = replay.run(
        [
            "--log-path",
            str(log_path),
            "--report-dir",
            str(report_dir),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 7
    parsed = json.loads(output_path.read_text(encoding="utf-8"))
    assert parsed["results"]["triggered"] is True
    assert parsed["results"]["diagnostics"]["return_code"] == 7


def test_run_fails_when_log_is_required_but_missing(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    output_path = report_dir / "replay.json"
    report_dir.mkdir(parents=True, exist_ok=True)

    exit_code = replay.run(
        [
            "--log-path",
            str(tmp_path / "missing.log"),
            "--report-dir",
            str(report_dir),
            "--require-log-path-exists",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 4
    parsed = json.loads(output_path.read_text(encoding="utf-8"))
    assert parsed["results"]["scan"]["log_path_exists"] is False
