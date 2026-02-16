from __future__ import annotations

from pathlib import Path

from eval import mcp_transport_soak as soak


def test_percentile_interpolates() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    assert soak._percentile(values, 0.0) == 10.0
    assert soak._percentile(values, 0.5) == 25.0
    assert soak._percentile(values, 1.0) == 40.0


def test_latency_stats_empty() -> None:
    stats = soak._latency_stats([])
    assert stats["count"] == 0.0
    assert stats["p95_ms"] == 0.0
    assert stats["avg_ms"] == 0.0


def test_report_path_builds_under_dir(tmp_path: Path) -> None:
    output = soak._report_path(tmp_path / "reports", "20260215_070000")
    assert output.parent.exists()
    assert output.name == "mcp_transport_soak_20260215_070000.json"


def test_extract_task_id_from_tools_call_response() -> None:
    response = {"result": {"task": {"taskId": "task-123"}}}
    assert soak._extract_task_id_from_tools_call_response(response) == "task-123"


def test_extract_task_id_from_tools_call_response_missing() -> None:
    assert soak._extract_task_id_from_tools_call_response({"result": {}}) is None
