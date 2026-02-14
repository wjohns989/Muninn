import json
import sys
from pathlib import Path

import pytest

from eval import run as eval_run


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_parse_required_track_args():
    parsed = eval_run._parse_required_track_args(
        ["accurate_retrieval:20", "test_time_learning:6"]
    )
    assert parsed == {"accurate_retrieval": 20, "test_time_learning": 6}


def test_parse_required_track_args_invalid():
    with pytest.raises(ValueError):
        eval_run._parse_required_track_args(["bad-format"])
    with pytest.raises(ValueError):
        eval_run._parse_required_track_args(["track:not-an-int"])


def test_significance_correction_family_all_vs_by_track():
    base_significance = {
        "global": {"@1": {"recall": {"p_value": 0.04, "mean_delta": -0.2, "ci_high": -0.01}}},
        "tracks": {
            "accurate_retrieval": {
                "cutoffs": {"@1": {"recall": {"p_value": 0.04, "mean_delta": -0.2, "ci_high": -0.01}}}
            }
        },
    }

    sig_all = json.loads(json.dumps(base_significance))
    eval_run._apply_significance_correction(
        sig_all,
        alpha=0.05,
        method="bonferroni",
        family="all",
    )
    assert sig_all["global"]["@1"]["recall"]["significant"] is False
    assert (
        sig_all["tracks"]["accurate_retrieval"]["cutoffs"]["@1"]["recall"]["significant"]
        is False
    )

    sig_by_track = json.loads(json.dumps(base_significance))
    eval_run._apply_significance_correction(
        sig_by_track,
        alpha=0.05,
        method="bonferroni",
        family="by_track",
    )
    assert sig_by_track["global"]["@1"]["recall"]["significant"] is True
    assert (
        sig_by_track["tracks"]["accurate_retrieval"]["cutoffs"]["@1"]["recall"]["significant"]
        is True
    )


def test_main_with_preset_and_track_coverage(tmp_path, monkeypatch):
    dataset_path = tmp_path / "dataset.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {"query_id": "q1", "relevant_ids": ["a"], "track": "accurate_retrieval"},
            {"query_id": "q2", "relevant_ids": ["b"], "track": "test_time_learning"},
            {"query_id": "q3", "relevant_ids": ["c"], "track": "long_range_understanding"},
            {"query_id": "q4", "relevant_ids": ["d"], "track": "conflict_resolution"},
        ],
    )
    _write_jsonl(
        predictions_path,
        [
            {"query_id": "q1", "ranked_ids": ["a"], "latency_ms": 10.0},
            {"query_id": "q2", "ranked_ids": ["b"], "latency_ms": 10.0},
            {"query_id": "q3", "ranked_ids": ["c"], "latency_ms": 10.0},
            {"query_id": "q4", "ranked_ids": ["d"], "latency_ms": 10.0},
        ],
    )

    argv = [
        "eval.run",
        "--preset",
        "vibecoder_memoryagentbench_v1",
        "--dataset",
        str(dataset_path),
        "--predictions",
        str(predictions_path),
        "--required-track",
        "accurate_retrieval:1",
        "--required-track",
        "test_time_learning:1",
        "--required-track",
        "long_range_understanding:1",
        "--required-track",
        "conflict_resolution:1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    exit_code = eval_run.main()
    assert exit_code == 0


def test_main_significance_gate_detects_regression(tmp_path, monkeypatch):
    dataset_path = tmp_path / "dataset.jsonl"
    current_predictions = tmp_path / "current.jsonl"
    baseline_predictions = tmp_path / "baseline.jsonl"
    output_path = tmp_path / "report.json"

    _write_jsonl(
        dataset_path,
        [
            {"query_id": f"q{i}", "relevant_ids": [f"m{i}"], "track": "accurate_retrieval"}
            for i in range(1, 13)
        ],
    )
    # Baseline retrieves correctly; current always misses.
    _write_jsonl(
        baseline_predictions,
        [
            {"query_id": f"q{i}", "ranked_ids": [f"m{i}", "x"], "latency_ms": 12.0}
            for i in range(1, 13)
        ],
    )
    _write_jsonl(
        current_predictions,
        [
            {"query_id": f"q{i}", "ranked_ids": ["x", "y"], "latency_ms": 12.0}
            for i in range(1, 13)
        ],
    )

    argv = [
        "eval.run",
        "--dataset",
        str(dataset_path),
        "--predictions",
        str(current_predictions),
        "--baseline-predictions",
        str(baseline_predictions),
        "--ks",
        "1",
        "--gate-significant-regressions",
        "--output",
        str(output_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    exit_code = eval_run.main()
    assert exit_code == 2

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert "significance" in report
    assert report["gate_config"]["significance_correction"] == "none"
    assert report["gate_config"]["significance_correction_family"] == "all"
    assert report["gates"]["passed"] is False
    assert any("significant regression" in v for v in report["gates"]["violations"])


def test_skip_baseline_compare_ignores_bad_baseline_report(tmp_path, monkeypatch):
    dataset_path = tmp_path / "dataset.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    bogus_baseline_report = tmp_path / "baseline.json"

    _write_jsonl(
        dataset_path,
        [{"query_id": "q1", "relevant_ids": ["m1"], "track": "accurate_retrieval"}],
    )
    _write_jsonl(
        predictions_path,
        [{"query_id": "q1", "ranked_ids": ["m1"], "latency_ms": 10.0}],
    )
    bogus_baseline_report.write_text(
        json.dumps({"cutoffs": {"@5": {"recall": 1.0, "mrr": 1.0, "ndcg": 1.0}}}, indent=2) + "\n",
        encoding="utf-8",
    )

    argv = [
        "eval.run",
        "--dataset",
        str(dataset_path),
        "--predictions",
        str(predictions_path),
        "--baseline-report",
        str(bogus_baseline_report),
        "--skip-baseline-compare",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    exit_code = eval_run.main()
    assert exit_code == 0
