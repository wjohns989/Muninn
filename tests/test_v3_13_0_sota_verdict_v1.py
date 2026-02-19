"""
Tests for Muninn v3.13.0 — SOTA+ Signed Verdict v1 (Phase 16).

Coverage:
  1. Provenance helpers — commit SHA, file hashing, HMAC signature
  2. LongMemEval external benchmark gate — pass/fail thresholds, missing-report handling
  3. sota-verdict payload — provenance block, gates.external_benchmarks, input_file_hashes
  4. StructMemEval adapter — parsing, metrics, evaluate_case, run aggregation, selftest
  5. Version bump — 3.13.0
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — import the production modules under test
# ---------------------------------------------------------------------------

from eval.ollama_local_benchmark import (
    _compute_hmac_signature,
    _evaluate_longmemeval_gate,
    _get_commit_sha,
    _sha256_file,
)
from eval.structmemeval_adapter import (
    AdapterReport,
    StructCase,
    StructMemEvalAdapter,
    exact_match,
    mrr_at_k,
    parse_dataset,
    run_selftest,
    token_f1,
)


# ===========================================================================
# 1. TestProvenanceHelpers
# ===========================================================================


class TestProvenanceHelpers:
    """Unit tests for _get_commit_sha, _sha256_file, _compute_hmac_signature."""

    # --- _sha256_file ---

    def test_sha256_file_matches_hashlib(self, tmp_path: Path) -> None:
        p = tmp_path / "data.bin"
        content = b"muninn v3.13.0 benchmark evidence"
        p.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert _sha256_file(p) == expected

    def test_sha256_file_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.json"
        p.write_bytes(b"")
        assert _sha256_file(p) == hashlib.sha256(b"").hexdigest()

    def test_sha256_file_large_file(self, tmp_path: Path) -> None:
        # >1 MiB to exercise the chunked read path
        p = tmp_path / "large.bin"
        data = b"x" * (2 * 1024 * 1024)
        p.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert _sha256_file(p) == expected

    # --- _get_commit_sha ---

    def test_get_commit_sha_returns_40_hex_on_success(self, tmp_path: Path) -> None:
        fake_sha = "a" * 40
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = fake_sha + "\n"
        with patch("eval.ollama_local_benchmark.subprocess.run", return_value=fake_result):
            result = _get_commit_sha(tmp_path)
        assert result == fake_sha

    def test_get_commit_sha_returns_none_on_nonzero_returncode(self, tmp_path: Path) -> None:
        fake_result = MagicMock()
        fake_result.returncode = 1
        fake_result.stdout = ""
        with patch("eval.ollama_local_benchmark.subprocess.run", return_value=fake_result):
            result = _get_commit_sha(tmp_path)
        assert result is None

    def test_get_commit_sha_returns_none_on_exception(self, tmp_path: Path) -> None:
        with patch("eval.ollama_local_benchmark.subprocess.run", side_effect=FileNotFoundError("git not found")):
            result = _get_commit_sha(tmp_path)
        assert result is None

    def test_get_commit_sha_rejects_non_40_char_output(self, tmp_path: Path) -> None:
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = "HEAD"  # not a valid SHA
        with patch("eval.ollama_local_benchmark.subprocess.run", return_value=fake_result):
            result = _get_commit_sha(tmp_path)
        assert result is None

    # --- _compute_hmac_signature ---

    def test_hmac_signature_format(self) -> None:
        data = {"run_id": "20260219T120000Z", "passed": True, "commit_sha": "a" * 40, "input_file_hashes": {}}
        sig = _compute_hmac_signature(data, "test-signing-key")
        assert sig.startswith("hmac-sha256=")
        hex_part = sig.split("=", 1)[1]
        assert len(hex_part) == 64  # SHA256 hex = 64 chars
        assert all(c in "0123456789abcdef" for c in hex_part)

    def test_hmac_signature_deterministic(self) -> None:
        data = {"run_id": "R1", "passed": False, "commit_sha": None, "input_file_hashes": {"f": "abc"}}
        sig1 = _compute_hmac_signature(data, "key")
        sig2 = _compute_hmac_signature(data, "key")
        assert sig1 == sig2

    def test_hmac_signature_different_keys_differ(self) -> None:
        data = {"run_id": "R1", "passed": True, "commit_sha": None, "input_file_hashes": {}}
        sig1 = _compute_hmac_signature(data, "key-alpha")
        sig2 = _compute_hmac_signature(data, "key-beta")
        assert sig1 != sig2

    def test_hmac_signature_verifiable_externally(self) -> None:
        """The signature can be recomputed independently using stdlib hmac."""
        data = {"run_id": "R2", "passed": True, "commit_sha": "b" * 40, "input_file_hashes": {"f.json": "deadbeef"}}
        key = "verifiable-key"
        sig = _compute_hmac_signature(data, key)
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        expected_hex = hmac.new(key.encode("utf-8"), canonical, hashlib.sha256).hexdigest()
        assert sig == f"hmac-sha256={expected_hex}"


# ===========================================================================
# 2. TestLongMemEvalGate
# ===========================================================================


class TestLongMemEvalGate:
    """Unit tests for _evaluate_longmemeval_gate threshold logic."""

    def _make_report(self, ndcg: float, recall: float) -> dict:
        return {"ndcg_at_10": ndcg, "recall_at_10": recall}

    def test_gate_passes_when_both_thresholds_met(self) -> None:
        r = _evaluate_longmemeval_gate(self._make_report(0.70, 0.75), min_ndcg_at_10=0.60, min_recall_at_10=0.65)
        assert r["passed"] is True
        assert r["violations"] == []
        assert r["ndcg_at_10"] == pytest.approx(0.70)
        assert r["recall_at_10"] == pytest.approx(0.75)

    def test_gate_fails_when_ndcg_below_threshold(self) -> None:
        r = _evaluate_longmemeval_gate(self._make_report(0.50, 0.75), min_ndcg_at_10=0.60, min_recall_at_10=0.65)
        assert r["passed"] is False
        assert any("ndcg" in v for v in r["violations"])

    def test_gate_fails_when_recall_below_threshold(self) -> None:
        r = _evaluate_longmemeval_gate(self._make_report(0.70, 0.60), min_ndcg_at_10=0.60, min_recall_at_10=0.65)
        assert r["passed"] is False
        assert any("recall" in v for v in r["violations"])

    def test_gate_fails_when_both_below_threshold(self) -> None:
        r = _evaluate_longmemeval_gate(self._make_report(0.40, 0.40), min_ndcg_at_10=0.60, min_recall_at_10=0.65)
        assert r["passed"] is False
        assert len(r["violations"]) == 2

    def test_gate_missing_ndcg_key(self) -> None:
        r = _evaluate_longmemeval_gate({"recall_at_10": 0.70}, min_ndcg_at_10=0.60, min_recall_at_10=0.65)
        assert r["passed"] is False
        assert any("missing" in v for v in r["violations"])

    def test_gate_accepts_mean_ndcg_at_10_variant(self) -> None:
        """Adapter emits mean_ndcg_at_10 rather than ndcg_at_10."""
        r = _evaluate_longmemeval_gate(
            {"mean_ndcg_at_10": 0.65, "mean_recall_at_10": 0.70},
            min_ndcg_at_10=0.60,
            min_recall_at_10=0.65,
        )
        assert r["passed"] is True

    def test_gate_accepts_cutoffs_at10_format(self) -> None:
        """eval.run report format: cutoffs.@10.{ndcg, recall}"""
        report = {"cutoffs": {"@10": {"ndcg": 0.72, "recall": 0.68}}}
        r = _evaluate_longmemeval_gate(report, min_ndcg_at_10=0.60, min_recall_at_10=0.65)
        assert r["passed"] is True

    def test_gate_boundary_exactly_at_threshold(self) -> None:
        """At exactly the threshold the gate must pass (not strictly greater)."""
        r = _evaluate_longmemeval_gate(self._make_report(0.60, 0.65), min_ndcg_at_10=0.60, min_recall_at_10=0.65)
        assert r["passed"] is True


# ===========================================================================
# 3. TestSotaVerdictPayload
# ===========================================================================


class TestSotaVerdictPayload:
    """
    Integration-level tests for the cmd_sota_verdict payload structure,
    verifying provenance and external_benchmarks gates are present.
    """

    def _minimal_eval_report(self, primary: float = 0.80) -> dict:
        return {
            "primary_score": primary,
            "tracks": {},
            "latency_ms": {"p95": 100.0, "count": 50},
        }

    def _write_json(self, tmp_path: Path, name: str, data: dict) -> Path:
        p = tmp_path / name
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    def _run_verdict(self, tmp_path: Path, extra_args: list[str] | None = None) -> dict:
        """
        Build a minimal verdict via argparse and cmd_sota_verdict,
        return the parsed payload dict.
        """
        from eval.ollama_local_benchmark import build_parser, cmd_sota_verdict

        candidate = self._write_json(tmp_path, "candidate.json", self._minimal_eval_report(0.90))
        baseline = self._write_json(tmp_path, "baseline.json", self._minimal_eval_report(0.80))
        output = tmp_path / "verdict.json"

        argv = [
            "sota-verdict",
            "--candidate-eval-report", str(candidate),
            "--baseline-eval-report", str(baseline),
            "--output", str(output),
            "--no-require-profile-gate",
            "--no-require-transport-reports",
            "--no-require-stat-significance",
            "--no-verify-artifacts",
            "--no-require-artifact-verification",
            "--no-require-longmemeval",
        ] + (extra_args or [])

        parser = build_parser()
        args = parser.parse_args(argv)
        # Patch git to return a stable SHA for deterministic tests
        with patch("eval.ollama_local_benchmark._get_commit_sha", return_value="c" * 40):
            cmd_sota_verdict(args)

        return json.loads(output.read_text(encoding="utf-8"))

    def test_payload_contains_provenance_key(self, tmp_path: Path) -> None:
        payload = self._run_verdict(tmp_path)
        assert "provenance" in payload

    def test_provenance_has_commit_sha(self, tmp_path: Path) -> None:
        payload = self._run_verdict(tmp_path)
        assert payload["provenance"]["commit_sha"] == "c" * 40

    def test_provenance_has_input_file_hashes(self, tmp_path: Path) -> None:
        payload = self._run_verdict(tmp_path)
        hashes = payload["provenance"]["input_file_hashes"]
        assert isinstance(hashes, dict)
        # candidate and baseline files must both be hashed
        keys = list(hashes.keys())
        assert any("candidate" in k for k in keys)
        assert any("baseline" in k for k in keys)

    def test_provenance_signature_none_when_no_key(self, tmp_path: Path) -> None:
        payload = self._run_verdict(tmp_path)
        assert payload["provenance"]["promotion_signature"] is None

    def test_provenance_signature_present_when_key_given(self, tmp_path: Path) -> None:
        payload = self._run_verdict(tmp_path, extra_args=["--signing-key", "my-secret"])
        sig = payload["provenance"]["promotion_signature"]
        assert sig is not None
        assert sig.startswith("hmac-sha256=")

    def test_gates_contains_external_benchmarks_key(self, tmp_path: Path) -> None:
        payload = self._run_verdict(tmp_path)
        assert "external_benchmarks" in payload["gates"]
        assert "longmemeval" in payload["gates"]["external_benchmarks"]

    def test_longmemeval_gate_absent_but_not_required_passes(self, tmp_path: Path) -> None:
        """No --longmemeval-report + --no-require-longmemeval → gate passes."""
        payload = self._run_verdict(tmp_path)
        lme_gate = payload["gates"]["external_benchmarks"]["longmemeval"]
        assert lme_gate["passed"] is True
        assert lme_gate["violations"] == []

    def test_longmemeval_gate_passes_with_valid_report(self, tmp_path: Path) -> None:
        lme_report = self._write_json(tmp_path, "lme.json", {"ndcg_at_10": 0.72, "recall_at_10": 0.68})
        payload = self._run_verdict(
            tmp_path,
            extra_args=[
                "--longmemeval-report", str(lme_report),
                "--min-longmemeval-ndcg", "0.60",
                "--min-longmemeval-recall", "0.65",
            ],
        )
        lme_gate = payload["gates"]["external_benchmarks"]["longmemeval"]
        assert lme_gate["passed"] is True

    def test_longmemeval_gate_fails_below_ndcg_threshold(self, tmp_path: Path) -> None:
        lme_report = self._write_json(tmp_path, "lme_low.json", {"ndcg_at_10": 0.50, "recall_at_10": 0.70})
        payload = self._run_verdict(
            tmp_path,
            extra_args=[
                "--longmemeval-report", str(lme_report),
                "--min-longmemeval-ndcg", "0.60",
            ],
        )
        lme_gate = payload["gates"]["external_benchmarks"]["longmemeval"]
        assert lme_gate["passed"] is False
        assert payload["passed"] is False

    def test_verdict_schema_version_is_1_0(self, tmp_path: Path) -> None:
        payload = self._run_verdict(tmp_path)
        assert payload["provenance"]["verdict_schema_version"] == "1.0"


# ===========================================================================
# 4. TestStructMemEvalMetrics
# ===========================================================================


class TestStructMemEvalMetrics:
    """Unit tests for StructMemEval metric functions."""

    # exact_match
    def test_exact_match_present(self) -> None:
        assert exact_match("694", "Phase 14 completed with 694 tests passing") == 1.0

    def test_exact_match_absent(self) -> None:
        assert exact_match("694", "Phase 13 completed") == 0.0

    def test_exact_match_case_insensitive(self) -> None:
        assert exact_match("Qdrant", "uses qdrant for vector storage") == 1.0

    def test_exact_match_list_all_present(self) -> None:
        assert exact_match("project, global", "scope can be project or global") == 1.0

    def test_exact_match_list_partial_missing(self) -> None:
        assert exact_match("project, global", "only project scope supported") == 0.0

    # token_f1
    def test_token_f1_perfect(self) -> None:
        assert token_f1("hello world", "hello world") == pytest.approx(1.0)

    def test_token_f1_no_overlap(self) -> None:
        assert token_f1("alpha beta", "gamma delta") == pytest.approx(0.0)

    def test_token_f1_partial_overlap(self) -> None:
        score = token_f1("foo bar baz", "foo bar")
        assert 0.0 < score < 1.0

    def test_token_f1_empty_expected(self) -> None:
        assert token_f1("", "") == pytest.approx(1.0)

    # mrr_at_k
    def test_mrr_rank_1(self) -> None:
        assert mrr_at_k(1, 10) == pytest.approx(1.0)

    def test_mrr_rank_2(self) -> None:
        assert mrr_at_k(2, 10) == pytest.approx(0.5)

    def test_mrr_rank_beyond_k(self) -> None:
        assert mrr_at_k(11, 10) == pytest.approx(0.0)

    def test_mrr_rank_zero_invalid(self) -> None:
        assert mrr_at_k(0, 10) == pytest.approx(0.0)


# ===========================================================================
# 5. TestStructMemEvalDatasetParser
# ===========================================================================


class TestStructMemEvalDatasetParser:
    def _make_case(self, case_id: str, relevant_idx: int = 0, n_memories: int = 2) -> dict:
        return {
            "case_id": case_id,
            "question": f"What is the answer for {case_id}?",
            "expected_answer": "42",
            "answer_type": "number",
            "memories": [f"Memory {i} for {case_id}" for i in range(n_memories)],
            "relevant_memory_index": relevant_idx,
        }

    def test_parse_dataset_from_jsonl(self, tmp_path: Path) -> None:
        lines = [json.dumps(self._make_case(f"c-{i:03d}")) for i in range(5)]
        p = tmp_path / "ds.jsonl"
        p.write_text("\n".join(lines), encoding="utf-8")
        cases = parse_dataset(p)
        assert len(cases) == 5
        assert all(isinstance(c, StructCase) for c in cases)

    def test_parse_dataset_skips_malformed_json(self, tmp_path: Path) -> None:
        lines = [
            json.dumps(self._make_case("c-001")),
            "NOT JSON {{{{",
            json.dumps(self._make_case("c-002")),
        ]
        p = tmp_path / "ds2.jsonl"
        p.write_text("\n".join(lines), encoding="utf-8")
        cases = parse_dataset(p)
        assert len(cases) == 2

    def test_parse_dataset_skips_out_of_range_index(self, tmp_path: Path) -> None:
        bad = self._make_case("c-bad", relevant_idx=99, n_memories=2)
        good = self._make_case("c-ok", relevant_idx=1, n_memories=2)
        p = tmp_path / "ds3.jsonl"
        p.write_text(json.dumps(bad) + "\n" + json.dumps(good), encoding="utf-8")
        cases = parse_dataset(p)
        assert len(cases) == 1
        assert cases[0].case_id == "c-ok"

    def test_parse_dataset_answer_type_default(self, tmp_path: Path) -> None:
        obj = self._make_case("c-001")
        del obj["answer_type"]
        p = tmp_path / "ds4.jsonl"
        p.write_text(json.dumps(obj), encoding="utf-8")
        cases = parse_dataset(p)
        assert cases[0].answer_type == "string"


# ===========================================================================
# 6. TestStructMemEvalAdapter
# ===========================================================================


class TestStructMemEvalAdapter:
    """Tests for StructMemEvalAdapter.evaluate_case and run."""

    def _make_case(self, case_id: str = "t-001") -> StructCase:
        return StructCase(
            case_id=case_id,
            question="What port does the server use?",
            expected_answer="42069",
            answer_type="number",
            memories=[
                "The authentication module uses HMAC-SHA256.",
                "Muninn FastAPI server listens on port 42069 by default.",
                "Qdrant stores vectors in the qdrant_v8 directory.",
            ],
            relevant_memory_index=1,
        )

    def _make_perfect_client(self, case: StructCase) -> MagicMock:
        """Client that always returns the relevant memory first."""
        client = MagicMock()
        client.add.return_value = {"memory_id": "mid", "status": "ok"}
        relevant_text = case.memories[case.relevant_memory_index]
        client.search.return_value = [
            {"id": "some-id", "content": relevant_text},
            {"id": "other-id", "content": case.memories[0]},
        ]
        client.delete_all.return_value = None
        return client

    def test_evaluate_case_exact_match_when_relevant_returned_first(self) -> None:
        case = self._make_case()
        client = self._make_perfect_client(case)
        adapter = StructMemEvalAdapter(client, k=10, cleanup=True)
        result = adapter.evaluate_case(case)
        assert result.exact_match == pytest.approx(1.0)
        assert result.error is None

    def test_evaluate_case_mrr_rank_1_when_relevant_first(self) -> None:
        case = self._make_case()
        # Make the returned IDs match the stable_memory_id for index 1
        adapter = StructMemEvalAdapter(MagicMock(), k=10, cleanup=True)
        relevant_mid = adapter._stable_memory_id(case.case_id, case.relevant_memory_index)
        client = MagicMock()
        client.add.return_value = {"memory_id": "mid", "status": "ok"}
        relevant_text = case.memories[case.relevant_memory_index]
        client.search.return_value = [{"id": relevant_mid, "content": relevant_text}]
        client.delete_all.return_value = None
        adapter.client = client
        result = adapter.evaluate_case(case)
        assert result.rank_of_relevant == 1
        assert result.mrr_at_k == pytest.approx(1.0)

    def test_evaluate_case_error_is_captured(self) -> None:
        case = self._make_case()
        client = MagicMock()
        client.add.side_effect = ConnectionError("server unreachable")
        adapter = StructMemEvalAdapter(client, k=10, cleanup=True)
        result = adapter.evaluate_case(case)
        assert result.error is not None
        assert "server unreachable" in result.error

    def test_evaluate_case_calls_delete_all_when_cleanup_true(self) -> None:
        case = self._make_case()
        client = self._make_perfect_client(case)
        adapter = StructMemEvalAdapter(client, k=10, cleanup=True)
        adapter.evaluate_case(case)
        client.delete_all.assert_called_once()

    def test_evaluate_case_skips_delete_all_when_cleanup_false(self) -> None:
        case = self._make_case()
        client = self._make_perfect_client(case)
        adapter = StructMemEvalAdapter(client, k=10, cleanup=False)
        adapter.evaluate_case(case)
        client.delete_all.assert_not_called()

    def test_run_aggregates_metrics(self) -> None:
        cases = [self._make_case(f"c-{i:03d}") for i in range(4)]
        client = self._make_perfect_client(cases[0])  # template client
        client.add.return_value = {"memory_id": "mid", "status": "ok"}
        client.search.return_value = [{"id": "x", "content": "port 42069 default"}]
        adapter = StructMemEvalAdapter(client, k=10, cleanup=True)
        report = adapter.run(cases)
        assert report.total_cases == 4
        assert report.evaluated_cases == 4
        assert report.skipped_cases == 0
        assert isinstance(report.mean_exact_match, float)
        assert isinstance(report.mean_mrr_at_k, float)

    def test_run_skips_case_with_empty_memories(self) -> None:
        empty_case = StructCase(
            case_id="empty",
            question="irrelevant",
            expected_answer="x",
            answer_type="string",
            memories=[],
            relevant_memory_index=0,
        )
        normal_case = self._make_case("normal")
        client = self._make_perfect_client(normal_case)
        adapter = StructMemEvalAdapter(client, k=10, cleanup=True)
        report = adapter.run([empty_case, normal_case])
        assert report.skipped_cases == 1
        assert report.evaluated_cases == 1

    def test_run_produces_by_answer_type_breakdown(self) -> None:
        cases = [
            StructCase(
                case_id="num-1",
                question="q",
                expected_answer="42",
                answer_type="number",
                memories=["The answer is 42"],
                relevant_memory_index=0,
            ),
            StructCase(
                case_id="str-1",
                question="q2",
                expected_answer="foo",
                answer_type="string",
                memories=["The value is foo"],
                relevant_memory_index=0,
            ),
        ]
        client = MagicMock()
        client.add.return_value = {"memory_id": "m", "status": "ok"}
        client.search.return_value = [{"id": "x", "content": "42 foo"}]
        client.delete_all.return_value = None
        adapter = StructMemEvalAdapter(client, k=10, cleanup=True)
        report = adapter.run(cases)
        assert "number" in report.by_answer_type
        assert "string" in report.by_answer_type


# ===========================================================================
# 7. TestStructMemEvalSelftest
# ===========================================================================


class TestStructMemEvalSelftest:
    """Verify the selftest oracle produces correct results."""

    def test_selftest_runs_without_server(self) -> None:
        report = run_selftest(k=3)
        assert isinstance(report, AdapterReport)
        assert report.total_cases == 3
        assert report.evaluated_cases == 3
        assert report.skipped_cases == 0

    def test_selftest_all_cases_pass_exact_match(self) -> None:
        report = run_selftest(k=3)
        assert report.mean_exact_match == pytest.approx(1.0)

    def test_selftest_all_cases_mrr_at_1(self) -> None:
        """Oracle client sorts by keyword overlap — relevant memories should rank first."""
        report = run_selftest(k=3)
        assert report.mean_mrr_at_k == pytest.approx(1.0)

    def test_selftest_by_answer_type_covers_three_types(self) -> None:
        report = run_selftest(k=3)
        assert "string" in report.by_answer_type
        assert "number" in report.by_answer_type
        assert "list" in report.by_answer_type

    def test_selftest_no_errors(self) -> None:
        report = run_selftest(k=3)
        assert report.errors == []


# ===========================================================================
# 8. TestVersionBump
# ===========================================================================


class TestVersionBump313:
    def test_version_is_3_13_0(self) -> None:
        from muninn.version import __version__
        assert __version__ == "3.13.0"

    def test_pyproject_version_matches(self) -> None:
        """pyproject.toml version field matches muninn.version.__version__."""
        from muninn.version import __version__

        root = Path(__file__).parent.parent
        toml_text = (root / "pyproject.toml").read_text(encoding="utf-8")
        import re
        match = re.search(r'^version\s*=\s*"([^"]+)"', toml_text, re.MULTILINE)
        assert match is not None, "version field not found in pyproject.toml"
        assert match.group(1) == __version__
