import hashlib
import json
from pathlib import Path

from eval.artifacts import (
    _build_expected_baseline_report,
    verify_all_preset_artifacts,
    verify_preset_artifacts,
)
from eval.presets import EvalPreset, PRESETS


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_verify_preset_artifacts_ok(tmp_path):
    preset_name = "test_eval_bundle"
    artifact_root = tmp_path / "artifacts"
    artifact_dir = artifact_root / preset_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = artifact_dir / "dataset.jsonl"
    predictions_path = artifact_dir / "baseline_predictions.jsonl"
    baseline_report_path = artifact_dir / "baseline_report.json"

    dataset_rows = [
        {"query_id": "q1", "track": "accurate_retrieval", "relevant_ids": ["m1"]},
        {"query_id": "q2", "track": "accurate_retrieval", "relevant_ids": ["m2"]},
    ]
    prediction_rows = [
        {"query_id": "q1", "ranked_ids": ["m1", "x"], "latency_ms": 20.0},
        {"query_id": "q2", "ranked_ids": ["m2", "x"], "latency_ms": 30.0},
    ]
    _write_jsonl(dataset_path, dataset_rows)
    _write_jsonl(predictions_path, prediction_rows)

    PRESETS[preset_name] = EvalPreset(
        name=preset_name,
        description="test bundle",
        ks="5,10",
        dataset_path=str(dataset_path),
        predictions_path=str(predictions_path),
        baseline_report_path=str(baseline_report_path),
        required_track_cases={"accurate_retrieval": 2},
        gate_significant_regressions=False,
    )
    try:
        baseline_report = _build_expected_baseline_report(
            preset_name=preset_name,
            dataset_rows=dataset_rows,
            prediction_rows=prediction_rows,
        )
        baseline_report_path.write_text(json.dumps(baseline_report, indent=2) + "\n", encoding="utf-8")

        manifest = {
            "manifest_version": 1,
            "preset": preset_name,
            "files": {
                "dataset.jsonl": _sha256(dataset_path),
                "baseline_predictions.jsonl": _sha256(predictions_path),
                "baseline_report.json": _sha256(baseline_report_path),
            },
            "dataset": {"rows": 2, "track_counts": {"accurate_retrieval": 2}},
        }
        (artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        result = verify_preset_artifacts(preset_name=preset_name, artifact_root=artifact_root)
        assert result["valid"] is True
        assert result["errors"] == []
    finally:
        del PRESETS[preset_name]


def test_verify_preset_artifacts_detects_tamper(tmp_path):
    preset_name = "test_eval_bundle_tamper"
    artifact_root = tmp_path / "artifacts"
    artifact_dir = artifact_root / preset_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = artifact_dir / "dataset.jsonl"
    predictions_path = artifact_dir / "baseline_predictions.jsonl"
    baseline_report_path = artifact_dir / "baseline_report.json"

    dataset_rows = [{"query_id": "q1", "track": "accurate_retrieval", "relevant_ids": ["m1"]}]
    prediction_rows = [{"query_id": "q1", "ranked_ids": ["m1"], "latency_ms": 20.0}]
    _write_jsonl(dataset_path, dataset_rows)
    _write_jsonl(predictions_path, prediction_rows)

    PRESETS[preset_name] = EvalPreset(
        name=preset_name,
        description="test bundle tamper",
        dataset_path=str(dataset_path),
        predictions_path=str(predictions_path),
        baseline_report_path=str(baseline_report_path),
        required_track_cases={"accurate_retrieval": 1},
        gate_significant_regressions=False,
    )
    try:
        baseline_report = _build_expected_baseline_report(
            preset_name=preset_name,
            dataset_rows=dataset_rows,
            prediction_rows=prediction_rows,
        )
        baseline_report_path.write_text(json.dumps(baseline_report, indent=2) + "\n", encoding="utf-8")
        manifest = {
            "manifest_version": 1,
            "preset": preset_name,
            "files": {
                "dataset.jsonl": _sha256(dataset_path),
                "baseline_predictions.jsonl": _sha256(predictions_path),
                "baseline_report.json": _sha256(baseline_report_path),
            },
            "dataset": {"rows": 1, "track_counts": {"accurate_retrieval": 1}},
        }
        (artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        # Tamper after manifest generation.
        dataset_path.write_text(
            json.dumps({"query_id": "q1", "track": "accurate_retrieval", "relevant_ids": ["DIFFERENT"]}) + "\n",
            encoding="utf-8",
        )

        result = verify_preset_artifacts(preset_name=preset_name, artifact_root=artifact_root)
        assert result["valid"] is False
        assert any("Checksum mismatch" in msg for msg in result["errors"])
    finally:
        del PRESETS[preset_name]


def test_verify_all_preset_artifacts_aggregates_results(tmp_path):
    artifact_root = tmp_path / "artifacts"

    ok_name = "test_eval_bundle_ok_all"
    bad_name = "test_eval_bundle_bad_all"

    ok_dir = artifact_root / ok_name
    bad_dir = artifact_root / bad_name
    ok_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    ok_dataset = ok_dir / "dataset.jsonl"
    ok_predictions = ok_dir / "baseline_predictions.jsonl"
    ok_report = ok_dir / "baseline_report.json"
    bad_dataset = bad_dir / "dataset.jsonl"
    bad_predictions = bad_dir / "baseline_predictions.jsonl"
    bad_report = bad_dir / "baseline_report.json"

    ok_rows = [{"query_id": "q1", "track": "accurate_retrieval", "relevant_ids": ["m1"]}]
    ok_preds = [{"query_id": "q1", "ranked_ids": ["m1"], "latency_ms": 20.0}]
    _write_jsonl(ok_dataset, ok_rows)
    _write_jsonl(ok_predictions, ok_preds)

    bad_rows = [{"query_id": "q2", "track": "accurate_retrieval", "relevant_ids": ["m2"]}]
    bad_preds = [{"query_id": "q2", "ranked_ids": ["m2"], "latency_ms": 25.0}]
    _write_jsonl(bad_dataset, bad_rows)
    _write_jsonl(bad_predictions, bad_preds)

    PRESETS[ok_name] = EvalPreset(
        name=ok_name,
        description="ok aggregate bundle",
        dataset_path=str(ok_dataset),
        predictions_path=str(ok_predictions),
        baseline_report_path=str(ok_report),
        required_track_cases={"accurate_retrieval": 1},
        gate_significant_regressions=False,
    )
    PRESETS[bad_name] = EvalPreset(
        name=bad_name,
        description="bad aggregate bundle",
        dataset_path=str(bad_dataset),
        predictions_path=str(bad_predictions),
        baseline_report_path=str(bad_report),
        required_track_cases={"accurate_retrieval": 1},
        gate_significant_regressions=False,
    )

    try:
        ok_expected = _build_expected_baseline_report(
            preset_name=ok_name,
            dataset_rows=ok_rows,
            prediction_rows=ok_preds,
        )
        ok_report.write_text(json.dumps(ok_expected, indent=2) + "\n", encoding="utf-8")
        ok_manifest = {
            "manifest_version": 1,
            "preset": ok_name,
            "files": {
                "dataset.jsonl": _sha256(ok_dataset),
                "baseline_predictions.jsonl": _sha256(ok_predictions),
                "baseline_report.json": _sha256(ok_report),
            },
            "dataset": {"rows": 1, "track_counts": {"accurate_retrieval": 1}},
        }
        (ok_dir / "manifest.json").write_text(json.dumps(ok_manifest, indent=2) + "\n", encoding="utf-8")

        bad_expected = _build_expected_baseline_report(
            preset_name=bad_name,
            dataset_rows=bad_rows,
            prediction_rows=bad_preds,
        )
        bad_report.write_text(json.dumps(bad_expected, indent=2) + "\n", encoding="utf-8")
        bad_manifest = {
            "manifest_version": 1,
            "preset": bad_name,
            "files": {
                "dataset.jsonl": _sha256(bad_dataset),
                "baseline_predictions.jsonl": _sha256(bad_predictions),
                "baseline_report.json": "0000badchecksum",
            },
            "dataset": {"rows": 1, "track_counts": {"accurate_retrieval": 1}},
        }
        (bad_dir / "manifest.json").write_text(json.dumps(bad_manifest, indent=2) + "\n", encoding="utf-8")

        result = verify_all_preset_artifacts(
            artifact_root=artifact_root,
            preset_names=[ok_name, bad_name],
        )
        assert result["valid"] is False
        assert result["checked_presets"] == [ok_name, bad_name]
        by_name = {entry["preset"]: entry for entry in result["results"]}
        assert by_name[ok_name]["valid"] is True
        assert by_name[bad_name]["valid"] is False
    finally:
        del PRESETS[ok_name]
        del PRESETS[bad_name]
