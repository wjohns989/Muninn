from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run_json_command(command: list[str]) -> Any:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"command failed: {' '.join(command)}\n{stderr}")
    stdout = completed.stdout.strip()
    return json.loads(stdout) if stdout else None


def _split_command(command: str) -> list[str]:
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError as exc:
        raise ValueError(f"invalid command string: {exc}") from exc
    if not tokens:
        raise ValueError("empty command string")
    return tokens


def _is_pytest_command(tokens: list[str]) -> bool:
    if not tokens:
        return False
    executable = Path(tokens[0]).name.lower()
    if executable in {"pytest", "pytest.exe", "py.test", "py.test.exe"}:
        return True
    if len(tokens) >= 3 and executable.startswith("python") and tokens[1] == "-m" and tokens[2] == "pytest":
        return True
    return False


def _has_option(tokens: list[str], option: str) -> bool:
    return any(token == option or token.startswith(f"{option}=") for token in tokens)


def _parse_junit_summary(path: Path) -> dict[str, int]:
    tree = ET.parse(path)
    root = tree.getroot()
    suites: list[ET.Element]
    if root.tag == "testsuite":
        suites = [root]
    else:
        suites = list(root.findall(".//testsuite"))

    tests = failures = errors = skipped = 0
    for suite in suites:
        tests += int(suite.attrib.get("tests", "0"))
        failures += int(suite.attrib.get("failures", "0"))
        errors += int(suite.attrib.get("errors", "0"))
        skipped += int(suite.attrib.get("skipped", "0"))

    passed = max(tests - failures - errors - skipped, 0)
    return {
        "passed": passed,
        "failed": failures,
        "errors": errors,
        "skipped": skipped,
        "xfailed": 0,
        "xpassed": 0,
        "warnings": 0,
    }


def _run_text_command(command: str) -> tuple[int, str, dict[str, int] | None, list[str]]:
    tokens = _split_command(command)
    run_tokens = list(tokens)
    junit_path: Path | None = None

    if _is_pytest_command(tokens) and not _has_option(tokens, "--junitxml"):
        with tempfile.NamedTemporaryFile(prefix="phase_hygiene_", suffix=".xml", delete=False) as tmp:
            junit_path = Path(tmp.name)
        run_tokens.append(f"--junitxml={junit_path}")

    completed = subprocess.run(
        run_tokens,
        shell=False,
        capture_output=True,
        text=True,
        check=False,
    )
    output = "\n".join([completed.stdout, completed.stderr]).strip()
    junit_summary: dict[str, int] | None = None
    if junit_path is not None and junit_path.exists():
        try:
            junit_summary = _parse_junit_summary(junit_path)
        except (OSError, ET.ParseError):
            junit_summary = None
        finally:
            try:
                junit_path.unlink()
            except OSError:
                pass
    return completed.returncode, output, junit_summary, run_tokens


def _extract_count(pattern: str, summary_text: str) -> int:
    match = re.search(pattern, summary_text, flags=re.IGNORECASE)
    if not match:
        return 0
    return int(match.group(1))


def _parse_pytest_summary(summary_text: str) -> dict[str, int]:
    return {
        "passed": _extract_count(r"(\d+)\s+passed", summary_text),
        "failed": _extract_count(r"(\d+)\s+failed", summary_text),
        "errors": _extract_count(r"(\d+)\s+error", summary_text),
        "skipped": _extract_count(r"(\d+)\s+skipped", summary_text),
        "xfailed": _extract_count(r"(\d+)\s+xfailed", summary_text),
        "xpassed": _extract_count(r"(\d+)\s+xpassed", summary_text),
        "warnings": _extract_count(r"(\d+)\s+warning", summary_text),
    }


def _summarize_check_rollup(entries: list[dict[str, Any]]) -> dict[str, Any]:
    failing: list[str] = []
    pending: list[str] = []
    passing: list[str] = []

    for entry in entries:
        name = entry.get("name") or entry.get("context") or entry.get("__typename", "unknown")
        conclusion = (entry.get("conclusion") or "").upper()
        state = (entry.get("state") or entry.get("status") or "").upper()

        if conclusion in {"FAILURE", "TIMED_OUT", "CANCELLED", "ACTION_REQUIRED", "STALE"}:
            failing.append(name)
            continue
        if conclusion in {"SUCCESS", "NEUTRAL", "SKIPPED"}:
            passing.append(name)
            continue

        if state in {"PENDING", "IN_PROGRESS", "QUEUED", "WAITING"}:
            pending.append(name)
        elif not conclusion and not state:
            pending.append(name)

    return {
        "total": len(entries),
        "failing": failing,
        "pending": pending,
        "passing": passing,
    }


def _select_target_pr(open_prs: list[dict[str, Any]], explicit_pr_number: int | None) -> int | None:
    if explicit_pr_number is not None:
        return explicit_pr_number
    if len(open_prs) == 1:
        return int(open_prs[0]["number"])
    return None


def _collect_open_prs() -> list[dict[str, Any]]:
    payload = _run_json_command(
        [
            "gh",
            "pr",
            "list",
            "--state",
            "open",
            "--json",
            "number,title,headRefName,baseRefName,updatedAt,reviewDecision,isDraft,mergeStateStatus,url",
        ]
    )
    return payload or []


def _collect_pr_details(pr_number: int) -> dict[str, Any]:
    return _run_json_command(
        [
            "gh",
            "pr",
            "view",
            str(pr_number),
            "--json",
            "number,title,url,isDraft,reviewDecision,comments,reviews,statusCheckRollup",
        ]
    )


def _evaluate_policy(
    *,
    open_prs: list[dict[str, Any]],
    target_pr: dict[str, Any] | None,
    pytest_return_code: int | None,
    pytest_summary: dict[str, int] | None,
    max_open_prs: int,
    require_open_pr: bool,
    fail_on_changes_requested: bool,
    fail_on_failing_checks: bool,
    max_skipped_tests: int,
    max_test_warnings: int,
) -> list[str]:
    violations: list[str] = []

    if len(open_prs) > max_open_prs:
        violations.append(f"open_pr_count_exceeds_limit: {len(open_prs)} > {max_open_prs}")
    if require_open_pr and not open_prs:
        violations.append("open_pr_required_but_none_found")

    if target_pr:
        review_decision = (target_pr.get("reviewDecision") or "").upper()
        if fail_on_changes_requested and review_decision == "CHANGES_REQUESTED":
            violations.append("pr_review_decision_changes_requested")

        checks = target_pr.get("checks") or {}
        if fail_on_failing_checks and checks.get("failing"):
            violations.append(f"pr_has_failing_checks: {', '.join(checks['failing'])}")

    if pytest_return_code is not None and pytest_return_code != 0:
        violations.append(f"pytest_return_code_nonzero: {pytest_return_code}")

    if pytest_summary:
        if pytest_summary.get("skipped", 0) > max_skipped_tests:
            violations.append(
                f"pytest_skipped_exceeds_budget: {pytest_summary['skipped']} > {max_skipped_tests}"
            )
        if pytest_summary.get("warnings", 0) > max_test_warnings:
            violations.append(
                f"pytest_warnings_exceeds_budget: {pytest_summary['warnings']} > {max_test_warnings}"
            )

    return violations


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase-boundary hygiene gate for PR status, review signals, and test summary budgets."
    )
    parser.add_argument("--max-open-prs", type=int, default=1)
    parser.add_argument("--require-open-pr", action="store_true")
    parser.add_argument("--pr-number", type=int, help="Explicit PR number to inspect.")
    parser.add_argument(
        "--fail-on-changes-requested",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fail-on-failing-checks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pytest-command",
        default="python -m pytest -q",
        help="Command to run for test summary evaluation. Pass empty string to skip test command.",
    )
    parser.add_argument("--max-skipped-tests", type=int, default=0)
    parser.add_argument("--max-test-warnings", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="eval/reports/hygiene",
        help="Directory where the JSON report is written.",
    )
    parser.add_argument(
        "--output",
        help="Explicit output JSON path. Defaults to output-dir/phase_hygiene_<timestamp>.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    generated_at = _utc_now()
    open_prs = _collect_open_prs()
    target_pr_number = _select_target_pr(open_prs, args.pr_number)
    target_pr: dict[str, Any] | None = None
    if target_pr_number is not None:
        target_pr = _collect_pr_details(target_pr_number)
        target_pr["checks"] = _summarize_check_rollup(target_pr.get("statusCheckRollup") or [])
        target_pr["comments"] = target_pr.get("comments") or []
        target_pr["reviews"] = target_pr.get("reviews") or []

    pytest_return_code: int | None = None
    pytest_output = ""
    pytest_summary: dict[str, int] | None = None
    resolved_test_command: list[str] | None = None
    if args.pytest_command and args.pytest_command.strip():
        pytest_return_code, pytest_output, junit_summary, resolved_test_command = _run_text_command(
            args.pytest_command
        )
        summary_from_output = _parse_pytest_summary(pytest_output)
        pytest_summary = junit_summary or summary_from_output
        if junit_summary is not None:
            # Keep warning/xfail/xpass visibility from terminal summary when available.
            pytest_summary["warnings"] = summary_from_output.get("warnings", 0)
            pytest_summary["xfailed"] = summary_from_output.get("xfailed", 0)
            pytest_summary["xpassed"] = summary_from_output.get("xpassed", 0)

    violations = _evaluate_policy(
        open_prs=open_prs,
        target_pr=target_pr,
        pytest_return_code=pytest_return_code,
        pytest_summary=pytest_summary,
        max_open_prs=args.max_open_prs,
        require_open_pr=args.require_open_pr,
        fail_on_changes_requested=args.fail_on_changes_requested,
        fail_on_failing_checks=args.fail_on_failing_checks,
        max_skipped_tests=args.max_skipped_tests,
        max_test_warnings=args.max_test_warnings,
    )

    report = {
        "generated_at_utc": generated_at.isoformat(),
        "policy": {
            "max_open_prs": args.max_open_prs,
            "require_open_pr": args.require_open_pr,
            "fail_on_changes_requested": args.fail_on_changes_requested,
            "fail_on_failing_checks": args.fail_on_failing_checks,
            "max_skipped_tests": args.max_skipped_tests,
            "max_test_warnings": args.max_test_warnings,
        },
        "open_prs": open_prs,
        "target_pr": None,
        "pytest": {
            "command": args.pytest_command,
            "resolved_command": resolved_test_command,
            "return_code": pytest_return_code,
            "summary": pytest_summary,
        },
        "passed": len(violations) == 0,
        "violations": violations,
    }
    if target_pr is not None:
        report["target_pr"] = {
            "number": target_pr.get("number"),
            "title": target_pr.get("title"),
            "url": target_pr.get("url"),
            "isDraft": target_pr.get("isDraft"),
            "reviewDecision": target_pr.get("reviewDecision"),
            "comment_count": len(target_pr.get("comments") or []),
            "review_count": len(target_pr.get("reviews") or []),
            "checks": target_pr.get("checks"),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        output_path = Path(args.output)
    else:
        ts = generated_at.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"phase_hygiene_{ts}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if report["passed"]:
        print(f"[phase-hygiene] PASS -> {output_path}")
    else:
        print(f"[phase-hygiene] FAIL -> {output_path}")
        for violation in violations:
            print(f"[phase-hygiene] violation: {violation}")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
