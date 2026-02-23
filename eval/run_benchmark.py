"""
Automated Benchmark Pipeline — Muninn SOTA+ evidence generation (Phase 17).

Orchestrates the complete benchmark evaluation workflow: LongMemEval adapter,
StructMemEval adapter, and optional SOTA+ signed verdict production. Supports
both dry-run mode (no live server required) and production mode (live Muninn server).

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │                  run_benchmark.py                        │
  │                                                          │
  │  ┌──────────────────┐    ┌──────────────────────────┐   │
  │  │ LongMemEval       │    │ StructMemEval             │   │
  │  │ Adapter          │    │ Adapter                   │   │
  │  │ (nDCG@10,        │    │ (EM, Token-F1, MRR@k)     │   │
  │  │  Recall@10)      │    │                           │   │
  │  └────────┬─────────┘    └────────────┬──────────────┘   │
  │           │                           │                  │
  │           └──────────────┬────────────┘                  │
  │                          ▼                               │
  │               ┌──────────────────┐                       │
  │               │ Combined Report  │                       │
  │               │  (JSON artifact) │                       │
  │               └──────────────────┘                       │
  └─────────────────────────────────────────────────────────┘

Modes:
  --dry-run          Run adapter selftests only (no live server needed)
  --production       Run against live Muninn server (requires server running)

Usage:
  # Dry-run (CI-safe, no server required):
  python -m eval.run_benchmark --dry-run

  # Production run against live server:
  python -m eval.run_benchmark --production \\
    --server-url http://localhost:42069 \\
    --auth-token YOUR_TOKEN \\
    --dataset-lme eval/data/longmemeval_synthetic_v1.jsonl \\
    --dataset-sme eval/data/structmemeval_suite_v1.jsonl \\
    --output eval/reports/benchmark_run.json

  # Production run with signed verdict:
  python -m eval.run_benchmark --production \\
    --server-url http://localhost:42069 \\
    --auth-token YOUR_TOKEN \\
    --signing-key MY_HMAC_KEY \\
    --require-longmemeval \\
    --output eval/reports/benchmark_signed.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import hmac
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

# Resolve the package root (two levels above this file: eval/ → repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_EVAL_DIR = Path(__file__).resolve().parent
_DATA_DIR = _EVAL_DIR / "data"
_REPORTS_DIR = _EVAL_DIR / "reports"

_SYNTHETIC_LME_DATASET = _DATA_DIR / "longmemeval_synthetic_v1.jsonl"
_SYNTHETIC_SME_DATASET = _DATA_DIR / "structmemeval_suite_v1.jsonl"

_DEFAULT_SERVER_URL = "http://localhost:42069"
_DEFAULT_MIN_LME_NDCG = 0.60
_DEFAULT_MIN_LME_RECALL = 0.65
_DEFAULT_MIN_SME_EM = 0.50
_DEFAULT_TIMEOUT_SECONDS = 300


# ─────────────────────────────────────────────────────────────────────────────
# Result data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AdapterResult:
    """Result from running one benchmark adapter."""
    adapter: str           # "longmemeval" | "structmemeval"
    mode: str              # "selftest" | "production"
    passed: bool
    metrics: Dict[str, Any]
    dataset_path: Optional[str]
    case_count: int
    elapsed_seconds: float
    error: Optional[str] = None
    raw_report: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkRunReport:
    """Combined report from a full benchmark pipeline run."""
    run_id: str
    mode: str             # "dry-run" | "production"
    timestamp_utc: str
    server_url: Optional[str]
    overall_passed: bool
    longmemeval: Optional[AdapterResult]
    structmemeval: Optional[AdapterResult]
    gates: Dict[str, Any]
    commit_sha: Optional[str]
    elapsed_total_seconds: float
    signature: Optional[str] = None
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Server health check
# ─────────────────────────────────────────────────────────────────────────────

def _check_server_health(server_url: str, auth_token: str, *, timeout: float = 10.0) -> bool:
    """Return True if the Muninn server is reachable and authenticated."""
    from urllib import error as urllib_error
    from urllib import request as urlrequest

    health_url = f"{server_url.rstrip('/')}/health"
    req = urlrequest.Request(
        health_url,
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib_error.URLError, urllib_error.HTTPError, OSError):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Commit SHA helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_commit_sha(repo_root: Path, *, timeout: float = 5.0) -> Optional[str]:
    """Retrieve the HEAD commit SHA for provenance binding."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            timeout=timeout,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
            if len(sha) == 40 and all(c in "0123456789abcdef" for c in sha):
                return sha
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess adapter runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_longmemeval(
    *,
    mode: str,
    server_url: str,
    auth_token: str,
    dataset_path: Optional[Path],
    output_path: Path,
    limit: Optional[int],
    timeout: float,
) -> AdapterResult:
    """Run the LongMemEval adapter as a subprocess and parse its JSON output."""
    t0 = time.monotonic()

    cmd = [sys.executable, "-m", "eval.longmemeval_adapter"]
    if mode == "selftest":
        cmd += ["--selftest", "--server-url", server_url]
    else:
        if dataset_path is None:
            dataset_path = _SYNTHETIC_LME_DATASET
        cmd += [
            "--dataset", str(dataset_path),
            "--server-url", server_url,
            "--auth-token", auth_token,
            "--output", str(output_path),
        ]
        if limit is not None:
            cmd += ["--limit", str(limit)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        return AdapterResult(
            adapter="longmemeval",
            mode=mode,
            passed=False,
            metrics={},
            dataset_path=str(dataset_path) if dataset_path else None,
            case_count=0,
            elapsed_seconds=elapsed,
            error=f"LongMemEval adapter timed out after {timeout:.0f}s",
        )

    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        return AdapterResult(
            adapter="longmemeval",
            mode=mode,
            passed=False,
            metrics={},
            dataset_path=str(dataset_path) if dataset_path else None,
            case_count=0,
            elapsed_seconds=elapsed,
            error=f"LongMemEval adapter exited {result.returncode}: {result.stderr[:500]}",
        )

    # In selftest mode, extract metrics from stdout (JSON line)
    raw_report: Optional[Dict[str, Any]] = None
    if mode == "selftest":
        for line in (result.stdout or "").splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    raw_report = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
    else:
        if output_path.exists():
            try:
                raw_report = json.loads(output_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

    metrics: Dict[str, Any] = {}
    case_count = 0
    passed = False
    if raw_report:
        metrics = {
            "mean_ndcg_at_k": raw_report.get("mean_ndcg_at_k", raw_report.get("mean_ndcg_at_10")),
            "mean_recall_at_k": raw_report.get("mean_recall_at_k", raw_report.get("mean_recall_at_10")),
            "k": raw_report.get("k", 10),
        }
        case_count = raw_report.get("total_cases", 0)
    # Selftest passes if exit code is 0; gate logic handles metric validation.
    passed = result.returncode == 0

    return AdapterResult(
        adapter="longmemeval",
        mode=mode,
        passed=passed,
        metrics=metrics,
        dataset_path=str(dataset_path) if dataset_path else None,
        case_count=case_count,
        elapsed_seconds=elapsed,
        raw_report=raw_report,
    )


def _run_structmemeval(
    *,
    mode: str,
    server_url: str,
    auth_token: str,
    dataset_path: Optional[Path],
    output_path: Path,
    limit: Optional[int],
    timeout: float,
) -> AdapterResult:
    """Run the StructMemEval adapter as a subprocess and parse its JSON output."""
    t0 = time.monotonic()

    cmd = [sys.executable, "-m", "eval.structmemeval_adapter"]
    if mode == "selftest":
        cmd.append("--selftest")
    else:
        if dataset_path is None:
            dataset_path = _SYNTHETIC_SME_DATASET
        cmd.extend([
            "--dataset", str(dataset_path),
            "--server-url", server_url,
            "--auth-token", auth_token,
            "--output", str(output_path),
        ])
        if limit is not None:
            cmd.extend(["--limit", str(limit)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        return AdapterResult(
            adapter="structmemeval",
            mode=mode,
            passed=False,
            metrics={},
            dataset_path=str(dataset_path) if dataset_path else None,
            case_count=0,
            elapsed_seconds=elapsed,
            error=f"StructMemEval adapter timed out after {timeout:.0f}s",
        )

    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        return AdapterResult(
            adapter="structmemeval",
            mode=mode,
            passed=False,
            metrics={},
            dataset_path=str(dataset_path) if dataset_path else None,
            case_count=0,
            elapsed_seconds=elapsed,
            error=f"StructMemEval adapter exited {result.returncode}: {result.stderr[:500]}",
        )

    raw_report: Optional[Dict[str, Any]] = None
    if output_path.exists():
        try:
            raw_report = json.loads(output_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    if raw_report is None:
        # Try parsing stdout for selftest
        for line in (result.stdout or "").splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    raw_report = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

    metrics: Dict[str, Any] = {}
    case_count = 0
    # Selftest passes if exit code is 0; gate logic handles metric validation.
    passed = result.returncode == 0
    if raw_report:
        # Use the aggregate (mean_*) keys that the adapter writes at report top-level.
        metrics = {
            "mean_exact_match": raw_report.get("mean_exact_match"),
            "mean_token_f1": raw_report.get("mean_token_f1"),
            "mean_mrr_at_k": raw_report.get("mean_mrr_at_k"),
        }
        case_count = raw_report.get("total_cases", 0)

    return AdapterResult(
        adapter="structmemeval",
        mode=mode,
        passed=passed,
        metrics=metrics,
        dataset_path=str(dataset_path) if dataset_path else None,
        case_count=case_count,
        elapsed_seconds=elapsed,
        raw_report=raw_report,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gate evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_lme_gate(
    lme: Optional[AdapterResult],
    *,
    min_ndcg: float,
    min_recall: float,
    require: bool,
) -> Dict[str, Any]:
    """Evaluate the LongMemEval gate from an adapter result."""
    if lme is None:
        return {"passed": not require, "reason": "not_run", "required": require}

    if not lme.passed:
        return {
            "passed": False,
            "reason": lme.error or "adapter_failed",
            "required": require,
        }

    ndcg = lme.metrics.get("mean_ndcg_at_k")
    recall = lme.metrics.get("mean_recall_at_k")

    if ndcg is None or recall is None:
        # Selftest passed but no numeric metrics (acceptable for dry-run)
        return {
            "passed": True,
            "reason": "selftest_pass_no_metrics",
            "required": require,
            "ndcg_at_10": ndcg,
            "recall_at_10": recall,
            "thresholds": {"min_ndcg_at_10": min_ndcg, "min_recall_at_10": min_recall},
        }

    ndcg_ok = ndcg >= min_ndcg
    recall_ok = recall >= min_recall
    gate_passed = ndcg_ok and recall_ok

    return {
        "passed": gate_passed,
        "reason": "threshold_evaluation",
        "required": require,
        "ndcg_at_10": ndcg,
        "recall_at_10": recall,
        "ndcg_ok": ndcg_ok,
        "recall_ok": recall_ok,
        "thresholds": {"min_ndcg_at_10": min_ndcg, "min_recall_at_10": min_recall},
    }


def _evaluate_sme_gate(
    sme: Optional[AdapterResult],
    *,
    min_em: float,
    require: bool,
) -> Dict[str, Any]:
    """Evaluate the StructMemEval gate from an adapter result."""
    if sme is None:
        return {"passed": not require, "reason": "not_run", "required": require}

    if not sme.passed:
        return {
            "passed": False,
            "reason": sme.error or "adapter_failed",
            "required": require,
        }

    em = sme.metrics.get("mean_exact_match")

    if em is None:
        return {
            "passed": True,
            "reason": "selftest_pass_no_metrics",
            "required": require,
            "mean_exact_match": em,
            "thresholds": {"min_exact_match": min_em},
        }

    gate_passed = em >= min_em
    return {
        "passed": gate_passed,
        "reason": "threshold_evaluation",
        "required": require,
        "mean_exact_match": em,
        "mean_token_f1": sme.metrics.get("mean_token_f1"),
        "mean_mrr_at_k": sme.metrics.get("mean_mrr_at_k"),
        "thresholds": {"min_exact_match": min_em},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report serialization
# ─────────────────────────────────────────────────────────────────────────────

def _adapter_result_to_dict(r: Optional[AdapterResult]) -> Optional[Dict[str, Any]]:
    if r is None:
        return None
    d = asdict(r)
    # Remove raw_report from top-level to keep report compact
    d.pop("raw_report", None)
    return d


def _serialize_report(report: BenchmarkRunReport) -> Dict[str, Any]:
    """Serialize a BenchmarkRunReport to a JSON-serializable dict."""
    data = {
        "run_id": report.run_id,
        "mode": report.mode,
        "timestamp_utc": report.timestamp_utc,
        "server_url": report.server_url,
        "overall_passed": report.overall_passed,
        "commit_sha": report.commit_sha,
        "elapsed_total_seconds": report.elapsed_total_seconds,
        "error": report.error,
        "adapters": {
            "longmemeval": _adapter_result_to_dict(report.longmemeval),
            "structmemeval": _adapter_result_to_dict(report.structmemeval),
        },
        "gates": report.gates,
    }
    if report.signature:
        data["signature"] = report.signature
    return data


def _sign_report(data: Dict[str, Any], key: str) -> str:
    """Generate HMAC-SHA256 signature for the report payload."""
    # Ensure signature is not in the payload before signing
    payload_data = {k: v for k, v in data.items() if k != "signature"}
    payload = json.dumps(
        payload_data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    sig = hmac.new(
        key.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"hmac-sha256:{sig}"


# ─────────────────────────────────────────────────────────────────────────────
# Core runner
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    *,
    mode: str,
    server_url: str = _DEFAULT_SERVER_URL,
    auth_token: str = "",
    dataset_lme: Optional[Path] = None,
    dataset_sme: Optional[Path] = None,
    output_path: Optional[Path] = None,
    signing_key: Optional[str] = None,
    require_lme: bool = False,
    require_sme: bool = False,
    min_lme_ndcg: float = _DEFAULT_MIN_LME_NDCG,
    min_lme_recall: float = _DEFAULT_MIN_LME_RECALL,
    min_sme_em: float = _DEFAULT_MIN_SME_EM,
    limit: Optional[int] = None,
    adapter_timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    skip_lme: bool = False,
    skip_sme: bool = False,
) -> BenchmarkRunReport:
    """
    Execute the full benchmark pipeline and return a BenchmarkRunReport.

    Args:
        mode: "dry-run" for selftests (no server), "production" for live evaluation
        server_url: Muninn server URL (required for production)
        auth_token: Bearer auth token (required for production)
        dataset_lme: Path to LongMemEval JSONL (defaults to synthetic dataset)
        dataset_sme: Path to StructMemEval JSONL (defaults to synthetic suite)
        output_path: Where to write the combined JSON report
        signing_key: HMAC-SHA256 key for provenance signing (optional)
        require_lme: Make LongMemEval gate mandatory for overall_passed
        require_sme: Make StructMemEval gate mandatory for overall_passed
        min_lme_ndcg: Minimum nDCG@10 for LongMemEval gate
        min_lme_recall: Minimum Recall@10 for LongMemEval gate
        min_sme_em: Minimum Exact Match for StructMemEval gate
        limit: Cap number of cases per adapter (useful for quick CI runs)
        adapter_timeout: Per-adapter subprocess timeout in seconds
        skip_lme: Skip LongMemEval adapter entirely
        skip_sme: Skip StructMemEval adapter entirely
    """
    total_t0 = time.monotonic()
    run_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    commit_sha = _get_commit_sha(_REPO_ROOT)

    # Ensure output directory exists
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Health check for production mode
    if mode == "production":
        if not auth_token:
            auth_token = os.environ.get("MUNINN_AUTH_TOKEN", "")
        if not _check_server_health(server_url, auth_token):
            elapsed = time.monotonic() - total_t0
            report = BenchmarkRunReport(
                run_id=run_id,
                mode=mode,
                timestamp_utc=timestamp,
                server_url=server_url,
                overall_passed=False,
                longmemeval=None,
                structmemeval=None,
                gates={},
                commit_sha=commit_sha,
                elapsed_total_seconds=elapsed,
                error=f"Muninn server not reachable at {server_url}",
            )
            if output_path:
                _write_report(output_path, report)
            return report

    # Determine effective mode for adapters
    adapter_mode = "selftest" if mode == "dry-run" else "production"

    # Temporary output files for adapter reports
    with tempfile.TemporaryDirectory(prefix="muninn_bench_") as tmpdir:
        lme_out = Path(tmpdir) / "lme_report.json"
        sme_out = Path(tmpdir) / "sme_report.json"

        # Run LongMemEval adapter
        lme_result: Optional[AdapterResult] = None
        if not skip_lme:
            lme_result = _run_longmemeval(
                mode=adapter_mode,
                server_url=server_url,
                auth_token=auth_token,
                dataset_path=dataset_lme,
                output_path=lme_out,
                limit=limit,
                timeout=adapter_timeout,
            )

        # Run StructMemEval adapter
        sme_result: Optional[AdapterResult] = None
        if not skip_sme:
            sme_result = _run_structmemeval(
                mode=adapter_mode,
                server_url=server_url,
                auth_token=auth_token,
                dataset_path=dataset_sme,
                output_path=sme_out,
                limit=limit,
                timeout=adapter_timeout,
            )

    # Evaluate gates
    lme_gate = _evaluate_lme_gate(
        lme_result,
        min_ndcg=min_lme_ndcg,
        min_recall=min_lme_recall,
        require=require_lme,
    )
    sme_gate = _evaluate_sme_gate(
        sme_result,
        min_em=min_sme_em,
        require=require_sme,
    )

    gates = {
        "longmemeval": lme_gate,
        "structmemeval": sme_gate,
    }

    # Overall pass requires all mandatory gates to pass
    overall_passed = all([
        lme_gate["passed"] if require_lme else True,
        sme_gate["passed"] if require_sme else True,
        # In dry-run, selftest failures are hard failures
        (lme_result.passed if lme_result and not skip_lme else True),
        (sme_result.passed if sme_result and not skip_sme else True),
    ])

    elapsed_total = time.monotonic() - total_t0

    report = BenchmarkRunReport(
        run_id=run_id,
        mode=mode,
        timestamp_utc=timestamp,
        server_url=server_url if mode == "production" else None,
        overall_passed=overall_passed,
        longmemeval=lme_result,
        structmemeval=sme_result,
        gates=gates,
        commit_sha=commit_sha,
        elapsed_total_seconds=elapsed_total,
    )

    # Sign if key provided
    if signing_key:
        report.signature = _sign_report(_serialize_report(report), signing_key)

    if output_path:
        _write_report(output_path, report)

    return report


def _write_report(path: Path, report: BenchmarkRunReport) -> None:
    """Write the benchmark report to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _serialize_report(report)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval.run_benchmark",
        description="Automated Muninn SOTA+ benchmark pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Run adapter selftests only — no live server required.",
    )
    mode_group.add_argument(
        "--production",
        action="store_true",
        dest="production",
        help="Run against a live Muninn server.",
    )

    parser.add_argument(
        "--server-url",
        default=_DEFAULT_SERVER_URL,
        help=f"Muninn server URL (default: {_DEFAULT_SERVER_URL}).",
    )
    parser.add_argument(
        "--auth-token",
        default=None,
        help="Muninn Bearer auth token (or set MUNINN_AUTH_TOKEN env var).",
    )
    parser.add_argument(
        "--dataset-lme",
        default=None,
        type=Path,
        help=f"LongMemEval JSONL dataset (default: {_SYNTHETIC_LME_DATASET}).",
    )
    parser.add_argument(
        "--dataset-sme",
        default=None,
        type=Path,
        help=f"StructMemEval JSONL dataset (default: {_SYNTHETIC_SME_DATASET}).",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Write combined JSON report to this path.",
    )
    parser.add_argument(
        "--signing-key",
        default=None,
        help="HMAC-SHA256 key for provenance signing (optional).",
    )
    parser.add_argument(
        "--require-longmemeval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Make LongMemEval gate mandatory for overall_passed.",
    )
    parser.add_argument(
        "--require-structmemeval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Make StructMemEval gate mandatory for overall_passed.",
    )
    parser.add_argument(
        "--min-lme-ndcg",
        default=_DEFAULT_MIN_LME_NDCG,
        type=float,
        help=f"Minimum LongMemEval nDCG@10 (default: {_DEFAULT_MIN_LME_NDCG}).",
    )
    parser.add_argument(
        "--min-lme-recall",
        default=_DEFAULT_MIN_LME_RECALL,
        type=float,
        help=f"Minimum LongMemEval Recall@10 (default: {_DEFAULT_MIN_LME_RECALL}).",
    )
    parser.add_argument(
        "--min-sme-em",
        default=_DEFAULT_MIN_SME_EM,
        type=float,
        help=f"Minimum StructMemEval Exact Match (default: {_DEFAULT_MIN_SME_EM}).",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Cap cases per adapter (useful for quick CI smoke tests).",
    )
    parser.add_argument(
        "--skip-lme",
        action="store_true",
        default=False,
        help="Skip LongMemEval adapter.",
    )
    parser.add_argument(
        "--skip-sme",
        action="store_true",
        default=False,
        help="Skip StructMemEval adapter.",
    )
    parser.add_argument(
        "--adapter-timeout",
        default=_DEFAULT_TIMEOUT_SECONDS,
        type=float,
        help=f"Per-adapter subprocess timeout in seconds (default: {_DEFAULT_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        default=False,
        help="Promote successful benchmark result to eval/reports/sota/promoted_verdict.json",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Print detailed progress to stderr.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    mode = "dry-run" if args.dry_run else "production"
    auth_token = args.auth_token or os.environ.get("MUNINN_AUTH_TOKEN", "")

    if args.verbose:
        print(f"[benchmark] mode={mode} server={args.server_url}", file=sys.stderr)
        print(f"[benchmark] commit_sha={_get_commit_sha(_REPO_ROOT)}", file=sys.stderr)

    # Determine output path
    output_path = args.output
    if output_path is None:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = _REPORTS_DIR / f"benchmark_{mode.replace('-', '_')}_{ts}.json"

    report = run_benchmark(
        mode=mode,
        server_url=args.server_url,
        auth_token=auth_token,
        dataset_lme=args.dataset_lme,
        dataset_sme=args.dataset_sme,
        output_path=output_path,
        signing_key=args.signing_key,
        require_lme=args.require_longmemeval,
        require_sme=args.require_structmemeval,
        min_lme_ndcg=args.min_lme_ndcg,
        min_lme_recall=args.min_lme_recall,
        min_sme_em=args.min_sme_em,
        limit=args.limit,
        adapter_timeout=args.adapter_timeout,
        skip_lme=args.skip_lme,
        skip_sme=args.skip_sme,
    )

    # Print summary to stdout
    summary = _serialize_report(report)
    print(json.dumps(summary, indent=2))

    if args.verbose:
        print(
            f"[benchmark] overall_passed={report.overall_passed} "
            f"elapsed={report.elapsed_total_seconds:.1f}s "
            f"output={output_path}",
            file=sys.stderr,
        )

    # Automated Promotion
    if args.promote and report.overall_passed:
        sota_dir = _REPORTS_DIR / "sota"
        sota_dir.mkdir(parents=True, exist_ok=True)
        promoted_path = sota_dir / "promoted_verdict.json"
        if output_path and output_path.exists():
            shutil.copy2(output_path, promoted_path)
            if args.verbose:
                print(f"[benchmark] promoted verdict to {promoted_path}", file=sys.stderr)

    return 0 if report.overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
