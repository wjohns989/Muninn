from __future__ import annotations

from eval import phase_hygiene as hygiene


def test_parse_pytest_summary_extracts_counts() -> None:
    output = "418 passed, 2 skipped, 0 warnings in 32.12s"
    summary = hygiene._parse_pytest_summary(output)
    assert summary["passed"] == 418
    assert summary["skipped"] == 2
    assert summary["warnings"] == 0
    assert summary["failed"] == 0


def test_evaluate_policy_flags_expected_violations() -> None:
    open_prs = [{"number": 31}, {"number": 32}]
    target_pr = {
        "reviewDecision": "CHANGES_REQUESTED",
        "checks": {"failing": ["ci / tests"]},
    }
    pytest_summary = {"skipped": 3, "warnings": 2}

    violations = hygiene._evaluate_policy(
        open_prs=open_prs,
        target_pr=target_pr,
        pytest_return_code=1,
        pytest_summary=pytest_summary,
        max_open_prs=1,
        require_open_pr=False,
        fail_on_changes_requested=True,
        fail_on_failing_checks=True,
        max_skipped_tests=0,
        max_test_warnings=0,
    )

    assert "open_pr_count_exceeds_limit: 2 > 1" in violations
    assert "pr_review_decision_changes_requested" in violations
    assert any(v.startswith("pr_has_failing_checks") for v in violations)
    assert "pytest_return_code_nonzero: 1" in violations
    assert "pytest_skipped_exceeds_budget: 3 > 0" in violations
    assert "pytest_warnings_exceeds_budget: 2 > 0" in violations


def test_evaluate_policy_passes_within_budgets() -> None:
    open_prs = [{"number": 41}]
    target_pr = {
        "reviewDecision": "APPROVED",
        "checks": {"failing": []},
    }
    pytest_summary = {"skipped": 0, "warnings": 0}

    violations = hygiene._evaluate_policy(
        open_prs=open_prs,
        target_pr=target_pr,
        pytest_return_code=0,
        pytest_summary=pytest_summary,
        max_open_prs=1,
        require_open_pr=True,
        fail_on_changes_requested=True,
        fail_on_failing_checks=True,
        max_skipped_tests=0,
        max_test_warnings=0,
    )

    assert violations == []
