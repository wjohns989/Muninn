from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from eval import mcp_transport_closure as closure


def test_parse_transports_rejects_invalid_value() -> None:
    try:
        closure._parse_transports("framed,invalid")
    except ValueError as exc:
        assert "Unsupported transport" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected ValueError for invalid transport")


def test_build_soak_command_contains_expected_flags(tmp_path: Path) -> None:
    args = Namespace(
        soak_iterations=30,
        soak_warmup_requests=3,
        soak_timeout_sec=20.0,
        soak_max_p95_ms=1500.0,
        soak_server_url="http://127.0.0.1:9",
        soak_failure_threshold=2,
        soak_cooldown_sec=45.0,
        soak_task_result_mode="immediate_retry",
        soak_task_result_auto_retry_clients="claude desktop,cursor",
        inject_malformed_frame=False,
        report_dir=tmp_path / "reports",
        wrapper=Path("mcp_wrapper.py"),
    )
    command = closure._build_soak_command(args, "framed")
    assert "--iterations" in command and "30" in command
    assert "--warmup-requests" in command and "3" in command
    assert "--transport" in command and "framed" in command
    assert "--task-result-mode" in command and "immediate_retry" in command
    assert "--task-result-auto-retry-clients" in command and "claude desktop,cursor" in command
    assert "--no-inject-malformed-frame" in command


def test_evaluate_campaign_ready_when_all_criteria_met() -> None:
    runs = [
        {"pass": True, "p95_compliant": True},
        {"pass": True, "p95_compliant": True},
        {"pass": True, "p95_compliant": True},
    ]
    result = closure._evaluate_campaign(
        campaign_runs=runs,
        streak_target=3,
        min_p95_compliance_ratio=0.95,
        unresolved_transport_regressions=0,
        open_wrapper_defects=0,
        unclassified_failures=0,
    )
    assert result["closure_ready"] is True
    assert result["criteria"]["streak_target_met"] is True


def test_evaluate_campaign_not_ready_with_open_wrapper_defect() -> None:
    runs = [
        {"pass": True, "p95_compliant": True},
        {"pass": True, "p95_compliant": True},
        {"pass": True, "p95_compliant": True},
    ]
    result = closure._evaluate_campaign(
        campaign_runs=runs,
        streak_target=3,
        min_p95_compliance_ratio=0.95,
        unresolved_transport_regressions=0,
        open_wrapper_defects=1,
        unclassified_failures=0,
    )
    assert result["closure_ready"] is False
    assert result["criteria"]["no_open_wrapper_defects"] is False


def test_aggregate_campaign_telemetry_collects_error_codes_and_modes() -> None:
    telemetry = closure._aggregate_campaign_telemetry(
        [
            {
                "transports": [
                    {
                        "error_codes": {"-32603": 4, "-32002": 1},
                        "task_result_mode": "auto",
                        "task_result_auto_retry_clients": "claude desktop,cursor",
                    },
                    {
                        "error_codes": {"-32603": 5},
                        "task_result_mode": "auto",
                        "task_result_auto_retry_clients": "claude desktop,cursor",
                    },
                ]
            },
            {
                "transports": [
                    {
                        "error_codes": {"-32603": 2},
                        "task_result_mode": "blocking",
                        "task_result_auto_retry_clients": "none",
                    }
                ]
            },
        ]
    )
    assert telemetry["error_code_totals"]["-32603"] == 11
    assert telemetry["error_code_totals"]["-32002"] == 1
    assert telemetry["task_result_mode_distribution"]["auto"] == 2
    assert telemetry["task_result_mode_distribution"]["blocking"] == 1
    assert telemetry["retryable_task_result_error_count"] == 1
    assert telemetry["retryable_task_result_error_ratio"] > 0.0
