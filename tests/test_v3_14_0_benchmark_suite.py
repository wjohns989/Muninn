"""
Phase 17 (v3.14.0) Test Suite — Synthetic Benchmark Datasets, Automated
Pipeline, and Parser Security Sandbox.

Test Classes:
  TestSyntheticLongMemEvalDataset   (8)  — LongMemEval JSONL dataset integrity
  TestSyntheticStructMemEvalDataset (8)  — StructMemEval JSONL dataset integrity
  TestBenchmarkRunnerImport         (5)  — run_benchmark.py importability & helpers
  TestBenchmarkRunnerDryRun         (10) — dry-run mode invocation & report structure
  TestBenchmarkRunnerGates          (8)  — gate evaluation logic
  TestParserSandbox                 (10) — sandbox.py correctness & error handling
  TestParserSubprocessWorker        (8)  — _parser_subprocess.py worker logic
  TestVersionBump314                (2)  — version consistency

Total: 59 tests
"""

from __future__ import annotations

import json
import sys
import textwrap
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_EVAL_DIR = _REPO_ROOT / "eval"
_DATA_DIR = _EVAL_DIR / "data"
_LME_DATASET = _DATA_DIR / "longmemeval_synthetic_v1.jsonl"
_SME_DATASET = _DATA_DIR / "structmemeval_suite_v1.jsonl"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load all JSON objects from a JSONL file."""
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{lineno}: {exc}") from exc
    return records


# ===========================================================================
# 1. TestSyntheticLongMemEvalDataset
# ===========================================================================

class TestSyntheticLongMemEvalDataset:
    """Validates the synthetic LongMemEval dataset (eval/data/longmemeval_synthetic_v1.jsonl)."""

    @pytest.fixture(scope="class")
    def records(self):
        assert _LME_DATASET.exists(), f"LongMemEval dataset not found: {_LME_DATASET}"
        return _load_jsonl(_LME_DATASET)

    def test_dataset_exists(self):
        assert _LME_DATASET.exists()

    def test_dataset_has_30_cases(self, records):
        assert len(records) == 30, f"Expected 30 cases, got {len(records)}"

    def test_required_fields_present(self, records):
        required = {"question_id", "question_type", "question", "expected_answer",
                    "question_date", "sessions"}
        for i, rec in enumerate(records):
            missing = required - rec.keys()
            assert not missing, f"Case {i} missing fields: {missing}"

    def test_question_ids_unique(self, records):
        ids = [r["question_id"] for r in records]
        assert len(ids) == len(set(ids)), "Duplicate question_ids found"

    def test_question_types_covered(self, records):
        types_present = {r["question_type"] for r in records}
        required_types = {"single-session-qa", "multi-session-qa", "temporal",
                          "adversarial", "entity-centric"}
        assert required_types.issubset(types_present), (
            f"Missing question types: {required_types - types_present}"
        )

    def test_sessions_structure(self, records):
        for i, rec in enumerate(records):
            sessions = rec.get("sessions", [])
            assert isinstance(sessions, list) and len(sessions) >= 1, (
                f"Case {i} ({rec['question_id']}) has no sessions"
            )
            for session in sessions:
                assert "session_id" in session, f"Case {i} session missing session_id"
                assert "conversation" in session, f"Case {i} session missing conversation"
                conv = session["conversation"]
                assert isinstance(conv, list) and len(conv) >= 1, (
                    f"Case {i} session has empty conversation"
                )

    def test_conversation_turns_have_role_and_content(self, records):
        for rec in records:
            for session in rec["sessions"]:
                for turn in session["conversation"]:
                    assert "role" in turn, f"Turn missing 'role' in {rec['question_id']}"
                    assert "content" in turn, f"Turn missing 'content' in {rec['question_id']}"
                    assert turn["role"] in {"user", "assistant"}, (
                        f"Invalid role {turn['role']!r} in {rec['question_id']}"
                    )
                    assert isinstance(turn["content"], str) and turn["content"].strip(), (
                        f"Empty content in {rec['question_id']}"
                    )

    def test_expected_answers_non_empty(self, records):
        for rec in records:
            assert rec.get("expected_answer", "").strip(), (
                f"Empty expected_answer in {rec['question_id']}"
            )


# ===========================================================================
# 2. TestSyntheticStructMemEvalDataset
# ===========================================================================

class TestSyntheticStructMemEvalDataset:
    """Validates the synthetic StructMemEval dataset (eval/data/structmemeval_suite_v1.jsonl)."""

    @pytest.fixture(scope="class")
    def records(self):
        assert _SME_DATASET.exists(), f"StructMemEval dataset not found: {_SME_DATASET}"
        return _load_jsonl(_SME_DATASET)

    def test_dataset_exists(self):
        assert _SME_DATASET.exists()

    def test_dataset_has_30_cases(self, records):
        assert len(records) == 30, f"Expected 30 cases, got {len(records)}"

    def test_required_fields_present(self, records):
        required = {"case_id", "question", "expected_answer", "answer_type",
                    "memories", "relevant_memory_index"}
        for i, rec in enumerate(records):
            missing = required - rec.keys()
            assert not missing, f"Case {i} missing fields: {missing}"

    def test_case_ids_unique(self, records):
        ids = [r["case_id"] for r in records]
        assert len(ids) == len(set(ids)), "Duplicate case_ids found"

    def test_answer_types_covered(self, records):
        types_present = {r["answer_type"] for r in records}
        required_types = {"string", "number", "entity", "list"}
        assert required_types.issubset(types_present), (
            f"Missing answer types: {required_types - types_present}"
        )

    def test_memories_non_empty_list(self, records):
        for i, rec in enumerate(records):
            mems = rec.get("memories", [])
            assert isinstance(mems, list) and len(mems) >= 1, (
                f"Case {i} ({rec['case_id']}) has no memories"
            )
            for mem in mems:
                assert isinstance(mem, str) and mem.strip(), (
                    f"Empty memory string in case {rec['case_id']}"
                )

    def test_relevant_memory_index_in_bounds(self, records):
        for rec in records:
            idx = rec["relevant_memory_index"]
            mems = rec["memories"]
            assert isinstance(idx, int) and 0 <= idx < len(mems), (
                f"relevant_memory_index {idx} out of bounds for case {rec['case_id']} "
                f"(memories: {len(mems)})"
            )

    def test_expected_answers_non_empty(self, records):
        for rec in records:
            assert rec.get("expected_answer", "").strip(), (
                f"Empty expected_answer in {rec['case_id']}"
            )


# ===========================================================================
# 3. TestBenchmarkRunnerImport
# ===========================================================================

class TestBenchmarkRunnerImport:
    """Tests that run_benchmark.py imports cleanly and exposes required symbols."""

    def test_module_importable(self):
        from eval import run_benchmark  # noqa: F401

    def test_run_benchmark_function_exists(self):
        from eval.run_benchmark import run_benchmark
        assert callable(run_benchmark)

    def test_build_parser_exists(self):
        from eval.run_benchmark import build_parser
        parser = build_parser()
        assert parser is not None

    def test_benchmark_run_report_dataclass(self):
        from eval.run_benchmark import BenchmarkRunReport
        # Must be constructable
        report = BenchmarkRunReport(
            run_id="test-id",
            mode="dry-run",
            timestamp_utc="2026-02-19T00:00:00Z",
            server_url=None,
            overall_passed=True,
            longmemeval=None,
            structmemeval=None,
            gates={},
            commit_sha=None,
            elapsed_total_seconds=1.0,
        )
        assert report.overall_passed is True
        assert report.mode == "dry-run"

    def test_get_commit_sha_returns_string_or_none(self):
        from eval.run_benchmark import _get_commit_sha
        sha = _get_commit_sha(_REPO_ROOT)
        # Either a valid 40-char hex SHA or None (no git in env)
        if sha is not None:
            assert isinstance(sha, str)
            assert len(sha) == 40
            assert all(c in "0123456789abcdef" for c in sha)


# ===========================================================================
# 4. TestBenchmarkRunnerDryRun
# ===========================================================================

class TestBenchmarkRunnerDryRun:
    """Tests dry-run mode of the benchmark pipeline (no server required)."""

    def test_dry_run_returns_report(self, tmp_path):
        from eval.run_benchmark import run_benchmark

        # Patch subprocess calls to simulate selftest pass
        with patch("eval.run_benchmark._run_longmemeval") as mock_lme, \
             patch("eval.run_benchmark._run_structmemeval") as mock_sme:
            from eval.run_benchmark import AdapterResult
            mock_lme.return_value = AdapterResult(
                adapter="longmemeval",
                mode="selftest",
                passed=True,
                metrics={"mean_ndcg_at_k": None, "mean_recall_at_k": None, "k": 10},
                dataset_path=None,
                case_count=3,
                elapsed_seconds=0.5,
            )
            mock_sme.return_value = AdapterResult(
                adapter="structmemeval",
                mode="selftest",
                passed=True,
                metrics={"mean_exact_match": None, "mean_token_f1": None, "mean_mrr_at_k": None},
                dataset_path=None,
                case_count=3,
                elapsed_seconds=0.3,
            )

            report = run_benchmark(mode="dry-run", output_path=tmp_path / "report.json")

        assert report is not None
        assert report.mode == "dry-run"

    def test_dry_run_overall_passed_when_adapters_pass(self, tmp_path):
        from eval.run_benchmark import run_benchmark, AdapterResult

        with patch("eval.run_benchmark._run_longmemeval") as mock_lme, \
             patch("eval.run_benchmark._run_structmemeval") as mock_sme:
            mock_lme.return_value = AdapterResult(
                adapter="longmemeval", mode="selftest", passed=True,
                metrics={}, dataset_path=None, case_count=3, elapsed_seconds=0.1,
            )
            mock_sme.return_value = AdapterResult(
                adapter="structmemeval", mode="selftest", passed=True,
                metrics={}, dataset_path=None, case_count=3, elapsed_seconds=0.1,
            )
            report = run_benchmark(mode="dry-run")

        assert report.overall_passed is True

    def test_dry_run_overall_failed_when_lme_fails(self, tmp_path):
        from eval.run_benchmark import run_benchmark, AdapterResult

        with patch("eval.run_benchmark._run_longmemeval") as mock_lme, \
             patch("eval.run_benchmark._run_structmemeval") as mock_sme:
            mock_lme.return_value = AdapterResult(
                adapter="longmemeval", mode="selftest", passed=False,
                metrics={}, dataset_path=None, case_count=0, elapsed_seconds=0.1,
                error="subprocess failed",
            )
            mock_sme.return_value = AdapterResult(
                adapter="structmemeval", mode="selftest", passed=True,
                metrics={}, dataset_path=None, case_count=3, elapsed_seconds=0.1,
            )
            report = run_benchmark(mode="dry-run")

        assert report.overall_passed is False

    def test_dry_run_writes_json_report(self, tmp_path):
        from eval.run_benchmark import run_benchmark, AdapterResult

        output = tmp_path / "out.json"
        with patch("eval.run_benchmark._run_longmemeval") as mock_lme, \
             patch("eval.run_benchmark._run_structmemeval") as mock_sme:
            mock_lme.return_value = AdapterResult(
                adapter="longmemeval", mode="selftest", passed=True,
                metrics={}, dataset_path=None, case_count=3, elapsed_seconds=0.1,
            )
            mock_sme.return_value = AdapterResult(
                adapter="structmemeval", mode="selftest", passed=True,
                metrics={}, dataset_path=None, case_count=3, elapsed_seconds=0.1,
            )
            run_benchmark(mode="dry-run", output_path=output)

        assert output.exists()
        data = json.loads(output.read_text())
        assert "run_id" in data
        assert "overall_passed" in data
        assert "gates" in data
        assert "adapters" in data

    def test_report_json_has_correct_schema(self, tmp_path):
        from eval.run_benchmark import run_benchmark, AdapterResult, _serialize_report

        with patch("eval.run_benchmark._run_longmemeval") as mock_lme, \
             patch("eval.run_benchmark._run_structmemeval") as mock_sme:
            mock_lme.return_value = AdapterResult(
                adapter="longmemeval", mode="selftest", passed=True,
                metrics={}, dataset_path=None, case_count=3, elapsed_seconds=0.1,
            )
            mock_sme.return_value = AdapterResult(
                adapter="structmemeval", mode="selftest", passed=True,
                metrics={}, dataset_path=None, case_count=3, elapsed_seconds=0.1,
            )
            report = run_benchmark(mode="dry-run")

        data = _serialize_report(report)
        assert isinstance(data["run_id"], str)
        assert isinstance(data["timestamp_utc"], str)
        assert isinstance(data["overall_passed"], bool)
        assert isinstance(data["elapsed_total_seconds"], float)
        assert isinstance(data["gates"], dict)
        assert "longmemeval" in data["gates"]
        assert "structmemeval" in data["gates"]

    def test_skip_lme_runs_only_sme(self, tmp_path):
        from eval.run_benchmark import run_benchmark, AdapterResult

        with patch("eval.run_benchmark._run_longmemeval") as mock_lme, \
             patch("eval.run_benchmark._run_structmemeval") as mock_sme:
            mock_sme.return_value = AdapterResult(
                adapter="structmemeval", mode="selftest", passed=True,
                metrics={}, dataset_path=None, case_count=3, elapsed_seconds=0.1,
            )
            report = run_benchmark(mode="dry-run", skip_lme=True)

        assert report.longmemeval is None
        mock_lme.assert_not_called()

    def test_skip_sme_runs_only_lme(self, tmp_path):
        from eval.run_benchmark import run_benchmark, AdapterResult

        with patch("eval.run_benchmark._run_longmemeval") as mock_lme, \
             patch("eval.run_benchmark._run_structmemeval") as mock_sme:
            mock_lme.return_value = AdapterResult(
                adapter="longmemeval", mode="selftest", passed=True,
                metrics={}, dataset_path=None, case_count=3, elapsed_seconds=0.1,
            )
            report = run_benchmark(mode="dry-run", skip_sme=True)

        assert report.structmemeval is None
        mock_sme.assert_not_called()

    def test_production_mode_fails_without_server(self, tmp_path):
        from eval.run_benchmark import run_benchmark

        with patch("eval.run_benchmark._check_server_health", return_value=False):
            report = run_benchmark(
                mode="production",
                server_url="http://localhost:19999",
                auth_token="fake-token",
            )

        assert report.overall_passed is False
        assert report.error is not None
        assert "not reachable" in report.error

    def test_run_id_is_unique_uuid(self, tmp_path):
        from eval.run_benchmark import run_benchmark, AdapterResult

        run_ids = set()
        with patch("eval.run_benchmark._run_longmemeval") as mock_lme, \
             patch("eval.run_benchmark._run_structmemeval") as mock_sme:
            mock_lme.return_value = AdapterResult(
                adapter="longmemeval", mode="selftest", passed=True,
                metrics={}, dataset_path=None, case_count=3, elapsed_seconds=0.1,
            )
            mock_sme.return_value = AdapterResult(
                adapter="structmemeval", mode="selftest", passed=True,
                metrics={}, dataset_path=None, case_count=3, elapsed_seconds=0.1,
            )
            for _ in range(3):
                r = run_benchmark(mode="dry-run")
                run_ids.add(r.run_id)

        assert len(run_ids) == 3, "run_id must be unique per invocation"

    def test_timestamp_is_iso8601(self, tmp_path):
        from eval.run_benchmark import run_benchmark, AdapterResult
        import datetime

        with patch("eval.run_benchmark._run_longmemeval") as mock_lme, \
             patch("eval.run_benchmark._run_structmemeval") as mock_sme:
            mock_lme.return_value = AdapterResult(
                adapter="longmemeval", mode="selftest", passed=True,
                metrics={}, dataset_path=None, case_count=3, elapsed_seconds=0.1,
            )
            mock_sme.return_value = AdapterResult(
                adapter="structmemeval", mode="selftest", passed=True,
                metrics={}, dataset_path=None, case_count=3, elapsed_seconds=0.1,
            )
            report = run_benchmark(mode="dry-run")

        # Must parse as datetime (will raise if malformed)
        dt = datetime.datetime.strptime(report.timestamp_utc, "%Y-%m-%dT%H:%M:%SZ")
        assert dt.year >= 2025


# ===========================================================================
# 5. TestBenchmarkRunnerGates
# ===========================================================================

class TestBenchmarkRunnerGates:
    """Tests gate evaluation logic in run_benchmark.py."""

    def test_lme_gate_passes_above_threshold(self):
        from eval.run_benchmark import _evaluate_lme_gate, AdapterResult

        lme = AdapterResult(
            adapter="longmemeval", mode="production", passed=True,
            metrics={"mean_ndcg_at_k": 0.72, "mean_recall_at_k": 0.70, "k": 10},
            dataset_path=None, case_count=30, elapsed_seconds=5.0,
        )
        result = _evaluate_lme_gate(lme, min_ndcg=0.60, min_recall=0.65, require=True)
        assert result["passed"] is True
        assert result["ndcg_ok"] is True
        assert result["recall_ok"] is True

    def test_lme_gate_fails_below_ndcg(self):
        from eval.run_benchmark import _evaluate_lme_gate, AdapterResult

        lme = AdapterResult(
            adapter="longmemeval", mode="production", passed=True,
            metrics={"mean_ndcg_at_k": 0.45, "mean_recall_at_k": 0.70, "k": 10},
            dataset_path=None, case_count=30, elapsed_seconds=5.0,
        )
        result = _evaluate_lme_gate(lme, min_ndcg=0.60, min_recall=0.65, require=True)
        assert result["passed"] is False
        assert result["ndcg_ok"] is False

    def test_lme_gate_fails_below_recall(self):
        from eval.run_benchmark import _evaluate_lme_gate, AdapterResult

        lme = AdapterResult(
            adapter="longmemeval", mode="production", passed=True,
            metrics={"mean_ndcg_at_k": 0.72, "mean_recall_at_k": 0.50, "k": 10},
            dataset_path=None, case_count=30, elapsed_seconds=5.0,
        )
        result = _evaluate_lme_gate(lme, min_ndcg=0.60, min_recall=0.65, require=True)
        assert result["passed"] is False
        assert result["recall_ok"] is False

    def test_lme_gate_not_required_when_missing(self):
        from eval.run_benchmark import _evaluate_lme_gate

        result = _evaluate_lme_gate(None, min_ndcg=0.60, min_recall=0.65, require=False)
        assert result["passed"] is True

    def test_lme_gate_required_fails_when_missing(self):
        from eval.run_benchmark import _evaluate_lme_gate

        result = _evaluate_lme_gate(None, min_ndcg=0.60, min_recall=0.65, require=True)
        assert result["passed"] is False

    def test_sme_gate_passes_above_threshold(self):
        from eval.run_benchmark import _evaluate_sme_gate, AdapterResult

        sme = AdapterResult(
            adapter="structmemeval", mode="production", passed=True,
            metrics={"mean_exact_match": 0.80, "mean_token_f1": 0.85, "mean_mrr_at_k": 0.90},
            dataset_path=None, case_count=30, elapsed_seconds=3.0,
        )
        result = _evaluate_sme_gate(sme, min_em=0.50, require=True)
        assert result["passed"] is True

    def test_sme_gate_fails_below_threshold(self):
        from eval.run_benchmark import _evaluate_sme_gate, AdapterResult

        sme = AdapterResult(
            adapter="structmemeval", mode="production", passed=True,
            metrics={"mean_exact_match": 0.30, "mean_token_f1": 0.40, "mean_mrr_at_k": 0.50},
            dataset_path=None, case_count=30, elapsed_seconds=3.0,
        )
        result = _evaluate_sme_gate(sme, min_em=0.50, require=True)
        assert result["passed"] is False

    def test_overall_passed_requires_mandatory_gates(self, tmp_path):
        from eval.run_benchmark import run_benchmark, AdapterResult

        with patch("eval.run_benchmark._run_longmemeval") as mock_lme, \
             patch("eval.run_benchmark._run_structmemeval") as mock_sme:
            mock_lme.return_value = AdapterResult(
                adapter="longmemeval", mode="production", passed=True,
                metrics={"mean_ndcg_at_k": 0.40, "mean_recall_at_k": 0.40, "k": 10},
                dataset_path=None, case_count=30, elapsed_seconds=1.0,
            )
            mock_sme.return_value = AdapterResult(
                adapter="structmemeval", mode="selftest", passed=True,
                metrics={}, dataset_path=None, case_count=3, elapsed_seconds=0.1,
            )
            # LME is required but below threshold → must fail overall
            with patch("eval.run_benchmark._check_server_health", return_value=True):
                report = run_benchmark(
                    mode="production",
                    auth_token="fake",
                    require_lme=True,
                    min_lme_ndcg=0.60,
                )

        assert report.overall_passed is False


# ===========================================================================
# 6. TestParserSandbox
# ===========================================================================

class TestParserSandbox:
    """Tests for muninn/ingestion/sandbox.py subprocess isolation."""

    def test_sandbox_module_importable(self):
        from muninn.ingestion.sandbox import sandboxed_parse_binary  # noqa: F401

    def test_invalid_source_type_raises_value_error(self, tmp_path):
        from muninn.ingestion.sandbox import sandboxed_parse_binary

        dummy = tmp_path / "test.txt"
        dummy.write_text("hello")
        with pytest.raises(ValueError, match="only supports 'pdf' and 'docx'"):
            sandboxed_parse_binary(dummy, "txt")

    def test_missing_file_raises_runtime_error(self, tmp_path):
        from muninn.ingestion.sandbox import sandboxed_parse_binary

        non_existent = tmp_path / "ghost.pdf"
        with pytest.raises(RuntimeError, match="File not found"):
            sandboxed_parse_binary(non_existent, "pdf")

    def test_subprocess_success_returns_text(self, tmp_path):
        from muninn.ingestion.sandbox import sandboxed_parse_binary

        dummy = tmp_path / "doc.pdf"
        dummy.write_bytes(b"%PDF-1.4 fake")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"text": "Hello from PDF"}).encode()
        mock_result.stderr = b""

        with patch("subprocess.run", return_value=mock_result):
            text = sandboxed_parse_binary(dummy, "pdf")

        assert text == "Hello from PDF"

    def test_subprocess_error_response_raises_runtime_error(self, tmp_path):
        from muninn.ingestion.sandbox import sandboxed_parse_binary

        dummy = tmp_path / "bad.pdf"
        dummy.write_bytes(b"not a pdf")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = json.dumps({"error": "PdfStreamError: unexpected EOF"}).encode()
        mock_result.stderr = b""

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="PdfStreamError"):
                sandboxed_parse_binary(dummy, "pdf")

    def test_subprocess_timeout_raises_runtime_error(self, tmp_path):
        import subprocess as sp
        from muninn.ingestion.sandbox import sandboxed_parse_binary

        dummy = tmp_path / "huge.pdf"
        dummy.write_bytes(b"%PDF fake content")

        with patch("subprocess.run", side_effect=sp.TimeoutExpired(cmd=[], timeout=1.0)):
            with pytest.raises(RuntimeError, match="timed out"):
                sandboxed_parse_binary(dummy, "pdf", timeout=1.0)

    def test_subprocess_invalid_json_raises_runtime_error(self, tmp_path):
        from muninn.ingestion.sandbox import sandboxed_parse_binary

        dummy = tmp_path / "weird.docx"
        dummy.write_bytes(b"PK\x03\x04 fake docx")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b"THIS IS NOT JSON"
        mock_result.stderr = b""

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="invalid JSON"):
                sandboxed_parse_binary(dummy, "docx")

    def test_subprocess_empty_stdout_exit_0_returns_empty_string(self, tmp_path):
        from muninn.ingestion.sandbox import sandboxed_parse_binary

        dummy = tmp_path / "empty.pdf"
        dummy.write_bytes(b"%PDF-1.4")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b""
        mock_result.stderr = b""

        with patch("subprocess.run", return_value=mock_result):
            text = sandboxed_parse_binary(dummy, "pdf")

        assert text == ""

    def test_fallback_in_process_disabled_by_default(self, tmp_path):
        """When subprocess raises FileNotFoundError and fallback=False, RuntimeError is raised."""
        from muninn.ingestion.sandbox import sandboxed_parse_binary

        dummy = tmp_path / "doc.pdf"
        dummy.write_bytes(b"%PDF fake")

        with patch("subprocess.run", side_effect=FileNotFoundError("python not found")):
            with pytest.raises(RuntimeError):
                sandboxed_parse_binary(dummy, "pdf", fallback_in_process=False)

    def test_max_stdout_bytes_cap(self, tmp_path):
        """Subprocess stdout exceeding MAX_STDOUT_BYTES is truncated before JSON parse."""
        from muninn.ingestion import sandbox

        dummy = tmp_path / "big.pdf"
        dummy.write_bytes(b"%PDF fake")

        # Build response larger than MAX_STDOUT_BYTES (4MB)
        large_text = "A" * (sandbox.MAX_STDOUT_BYTES + 1000)
        payload_bytes = json.dumps({"text": large_text}).encode()

        mock_result = MagicMock()
        mock_result.returncode = 0
        # Simulate OS returning truncated stdout (only first MAX_STDOUT_BYTES)
        mock_result.stdout = payload_bytes[:sandbox.MAX_STDOUT_BYTES]
        mock_result.stderr = b""

        # Should raise RuntimeError because truncation broke the JSON
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError):
                sandbox.sandboxed_parse_binary(dummy, "pdf")

    def test_sandbox_env_strips_sensitive_vars(self):
        """_make_sandbox_env must exclude secrets from the child process environment."""
        from muninn.ingestion.sandbox import _make_sandbox_env, _SANDBOX_ENV_ALLOWLIST
        import os

        sensitive_vars = {
            "MUNINN_AUTH_TOKEN": "super-secret-token",
            "OPENAI_API_KEY": "sk-fake-key",
            "ANTHROPIC_API_KEY": "ant-fake-key",
            "DATABASE_URL": "postgres://user:pass@host/db",
            "AWS_SECRET_ACCESS_KEY": "aws-secret-key",
        }

        with patch.dict(os.environ, sensitive_vars, clear=False):
            env = _make_sandbox_env()

        # None of the injected sensitive vars should appear
        for key in sensitive_vars:
            assert key not in env, (
                f"_make_sandbox_env leaked sensitive variable: {key}"
            )
        # Confirm the allowlist only contains benign vars
        assert "MUNINN_AUTH_TOKEN" not in _SANDBOX_ENV_ALLOWLIST
        assert "OPENAI_API_KEY" not in _SANDBOX_ENV_ALLOWLIST

    def test_sandbox_env_preserves_path(self):
        """_make_sandbox_env must keep PATH so the child can find system libs."""
        from muninn.ingestion.sandbox import _make_sandbox_env
        import os

        test_path = "/usr/local/bin:/usr/bin"
        with patch.dict(os.environ, {"PATH": test_path}, clear=False):
            env = _make_sandbox_env()

        assert "PATH" in env
        assert env["PATH"] == test_path

    def test_subprocess_call_uses_sandboxed_env(self, tmp_path):
        """subprocess.run must be called with env= kwarg (not None) for isolation."""
        from muninn.ingestion.sandbox import sandboxed_parse_binary

        dummy = tmp_path / "doc.pdf"
        dummy.write_bytes(b"%PDF-1.4 fake")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"text": "clean"}).encode()
        mock_result.stderr = b""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            sandboxed_parse_binary(dummy, "pdf")

        call_kwargs = mock_run.call_args.kwargs
        assert "env" in call_kwargs, "subprocess.run must receive explicit env= for isolation"
        assert call_kwargs["env"] is not None


# ===========================================================================
# 7. TestParserSubprocessWorker
# ===========================================================================

class TestParserSubprocessWorker:
    """Tests for muninn/ingestion/_parser_subprocess.py worker logic."""

    def test_worker_importable(self):
        from muninn.ingestion import _parser_subprocess  # noqa: F401

    def test_wrong_argc_returns_exit_2(self, capsys):
        """Worker exits 2 when called with wrong number of arguments."""
        from muninn.ingestion._parser_subprocess import main

        with patch.object(sys, "argv", ["_parser_subprocess.py"]):
            exit_code = main()
        assert exit_code == 2

    def test_unsupported_source_type_returns_exit_1(self, tmp_path, capsys):
        from muninn.ingestion._parser_subprocess import main

        dummy = tmp_path / "test.txt"
        dummy.write_text("hello")
        with patch.object(sys, "argv", ["_parser_subprocess.py", "html", str(dummy)]):
            exit_code = main()
        assert exit_code == 1
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "error" in data
        assert "unsupported" in data["error"]

    def test_missing_file_returns_exit_1(self, tmp_path, capsys):
        from muninn.ingestion._parser_subprocess import main

        missing = tmp_path / "ghost.pdf"
        with patch.object(sys, "argv", ["_parser_subprocess.py", "pdf", str(missing)]):
            exit_code = main()
        assert exit_code == 1
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "error" in data
        assert "not found" in data["error"]

    def test_pdf_parse_success_emits_json_text(self, tmp_path, capsys):
        from muninn.ingestion._parser_subprocess import main

        dummy = tmp_path / "doc.pdf"
        dummy.write_bytes(b"%PDF fake")

        with patch.object(sys, "argv", ["_parser_subprocess.py", "pdf", str(dummy)]):
            with patch("muninn.ingestion._parser_subprocess._parse_pdf",
                       return_value="Extracted PDF text"):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data.get("text") == "Extracted PDF text"

    def test_docx_parse_success_emits_json_text(self, tmp_path, capsys):
        from muninn.ingestion._parser_subprocess import main

        dummy = tmp_path / "report.docx"
        dummy.write_bytes(b"PK fake docx")

        with patch.object(sys, "argv", ["_parser_subprocess.py", "docx", str(dummy)]):
            with patch("muninn.ingestion._parser_subprocess._parse_docx",
                       return_value="Extracted DOCX content"):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data.get("text") == "Extracted DOCX content"

    def test_parse_exception_emits_error_json(self, tmp_path, capsys):
        from muninn.ingestion._parser_subprocess import main

        dummy = tmp_path / "corrupt.pdf"
        dummy.write_bytes(b"%PDF bad data")

        with patch.object(sys, "argv", ["_parser_subprocess.py", "pdf", str(dummy)]):
            with patch("muninn.ingestion._parser_subprocess._parse_pdf",
                       side_effect=Exception("CryptFilter error")):
                exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "error" in data
        assert "CryptFilter" in data["error"]

    def test_output_truncated_at_max_chars(self, tmp_path, capsys):
        from muninn.ingestion import _parser_subprocess
        from muninn.ingestion._parser_subprocess import main

        dummy = tmp_path / "huge.pdf"
        dummy.write_bytes(b"%PDF huge")

        large_text = "X" * (_parser_subprocess.MAX_OUTPUT_CHARS + 500)

        with patch.object(sys, "argv", ["_parser_subprocess.py", "pdf", str(dummy)]):
            with patch("muninn.ingestion._parser_subprocess._parse_pdf",
                       return_value=large_text):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data["text"]) <= _parser_subprocess.MAX_OUTPUT_CHARS + len("\n\n[TRUNCATED]")
        assert data["text"].endswith("[TRUNCATED]")

    def test_import_error_in_pdf_parse_raises_runtime_error(self, tmp_path):
        from muninn.ingestion._parser_subprocess import _parse_pdf

        dummy = tmp_path / "doc.pdf"
        dummy.write_bytes(b"%PDF")

        # Simulate pypdf not installed
        with patch.dict(sys.modules, {"pypdf": None}):
            with pytest.raises((RuntimeError, ImportError)):
                _parse_pdf(dummy)


# ===========================================================================
# 8. TestVersionBump314
# ===========================================================================

class TestVersionBump314:
    """Verifies version was bumped to 3.14.0 across all version files."""

    def test_version_module_is_314(self):
        from muninn.version import __version__
        parts = tuple(int(x) for x in __version__.split("."))
        assert parts >= (3, 14, 0), (
            f"Expected >= 3.14.0, got {__version__}."
        )

    def test_pyproject_toml_version_matches(self):
        import re
        pyproject = _REPO_ROOT / "pyproject.toml"
        assert pyproject.exists()
        content = pyproject.read_text(encoding="utf-8")
        m = re.search(r'^version\s*=\s*"([\d.]+)"', content, re.MULTILINE)
        assert m is not None
        from muninn.version import __version__
        assert m.group(1) == __version__, (
            f"pyproject.toml ({m.group(1)}) must match version.py ({__version__})"
        )
