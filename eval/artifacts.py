"""Artifact integrity and reproducibility checks for eval benchmark presets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from eval.metrics import evaluate_batch
from eval.presets import PRESETS, get_preset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(obj)
    return rows


def _merge_cases(
    dataset_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_query_id = {row["query_id"]: row for row in prediction_rows if "query_id" in row}
    merged: list[dict[str, Any]] = []
    for row in dataset_rows:
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            continue
        pred = by_query_id.get(query_id, {})
        ranked_ids = pred.get("ranked_ids", [])
        if not isinstance(ranked_ids, list):
            ranked_ids = []
        latency_ms = pred.get("latency_ms")
        if not isinstance(latency_ms, (int, float)):
            latency_ms = None
        relevant_ids = row.get("relevant_ids", [])
        if not isinstance(relevant_ids, list):
            relevant_ids = []
        track = row.get("track")
        if not isinstance(track, str):
            track = None
        merged.append(
            {
                "query_id": query_id,
                "relevant_ids": relevant_ids,
                "ranked_ids": ranked_ids,
                "latency_ms": latency_ms,
                "track": track,
            }
        )
    return merged


def _float_equal(a: Any, b: Any, tol: float = 1e-12) -> bool:
    try:
        af = float(a)
        bf = float(b)
    except (TypeError, ValueError):
        return a == b
    return abs(af - bf) <= tol


def _iter_mismatches(
    expected: Any,
    actual: Any,
    path: str = "",
) -> Iterable[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        keys = sorted(set(expected).union(actual))
        for key in keys:
            next_path = f"{path}.{key}" if path else str(key)
            if key not in expected:
                yield f"Unexpected key: {next_path}"
                continue
            if key not in actual:
                yield f"Missing key: {next_path}"
                continue
            yield from _iter_mismatches(expected[key], actual[key], next_path)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            yield f"Length mismatch at {path}: expected {len(expected)}, got {len(actual)}"
            return
        for idx, (left, right) in enumerate(zip(expected, actual)):
            yield from _iter_mismatches(left, right, f"{path}[{idx}]")
        return
    if not _float_equal(expected, actual):
        yield f"Value mismatch at {path}: expected {expected!r}, got {actual!r}"


def _build_expected_baseline_report(
    *,
    preset_name: str,
    dataset_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    preset = get_preset(preset_name)
    ks = [int(x.strip()) for x in preset.ks.split(",") if x.strip()]
    cases = _merge_cases(dataset_rows, prediction_rows)
    report = evaluate_batch(cases, ks=ks)
    report["dataset_size"] = len(dataset_rows)
    report["predictions_size"] = len(prediction_rows)
    report["matched_queries"] = len(cases)
    report["gates"] = {"passed": True, "violations": []}
    report["gate_config"] = {
        "preset": preset.name,
        "max_metric_regression": preset.max_metric_regression,
        "max_track_metric_regression": preset.max_track_metric_regression,
        "max_p95_latency_ms": preset.max_p95_latency_ms,
        "required_track_cases": preset.required_track_cases,
        "skip_baseline_compare": True,
        "baseline_predictions": None,
        "significance_alpha": preset.significance_alpha,
        "bootstrap_samples": preset.bootstrap_samples,
        "permutation_rounds": preset.permutation_rounds,
        "significance_seed": 42,
        "gate_significant_regressions": preset.gate_significant_regressions,
        "significance_correction": preset.significance_correction,
        "significance_correction_family": preset.significance_correction_family,
    }
    return report


def verify_preset_artifacts(
    *,
    preset_name: str,
    artifact_root: Path = Path("eval/artifacts"),
) -> dict[str, Any]:
    """Verify checksums, dataset contract, and baseline-report reproducibility."""
    preset = get_preset(preset_name)
    artifact_dir = artifact_root / preset_name
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = _load_json(manifest_path)

    result: Dict[str, Any] = {
        "preset": preset_name,
        "artifact_dir": str(artifact_dir),
        "valid": True,
        "checks": [],
        "errors": [],
    }

    # Checksum validation
    manifest_files = manifest.get("files", {})
    if not isinstance(manifest_files, dict) or not manifest_files:
        result["valid"] = False
        result["errors"].append("manifest.files is missing or empty")
        return result

    for rel_name, expected_sha in manifest_files.items():
        file_path = artifact_dir / rel_name
        if not file_path.exists():
            result["valid"] = False
            result["errors"].append(f"Missing artifact file: {file_path}")
            continue
        actual_sha = _sha256(file_path)
        if actual_sha != expected_sha:
            result["valid"] = False
            result["errors"].append(
                f"Checksum mismatch for {rel_name}: expected {expected_sha}, got {actual_sha}"
            )
    result["checks"].append("checksums")

    # Dataset contract validation
    dataset_path = Path(preset.dataset_path or "")
    if not dataset_path.exists():
        result["valid"] = False
        result["errors"].append(f"Missing dataset path from preset: {dataset_path}")
        return result
    dataset_rows = _load_jsonl(dataset_path)
    observed_track_counts: Dict[str, int] = {}
    query_ids = set()
    for row in dataset_rows:
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            result["valid"] = False
            result["errors"].append("Dataset row missing non-empty query_id")
            continue
        if query_id in query_ids:
            result["valid"] = False
            result["errors"].append(f"Duplicate query_id in dataset: {query_id}")
        query_ids.add(query_id)

        track = row.get("track")
        if isinstance(track, str) and track:
            observed_track_counts[track] = observed_track_counts.get(track, 0) + 1
        rel = row.get("relevant_ids")
        if not isinstance(rel, list) or not rel:
            result["valid"] = False
            result["errors"].append(f"Dataset row {query_id} has empty relevant_ids")

    declared = manifest.get("dataset", {})
    declared_rows = int(declared.get("rows", -1))
    if declared_rows != len(dataset_rows):
        result["valid"] = False
        result["errors"].append(
            f"Manifest dataset.rows mismatch: expected {declared_rows}, got {len(dataset_rows)}"
        )
    declared_track_counts = declared.get("track_counts", {})
    if isinstance(declared_track_counts, dict) and declared_track_counts != observed_track_counts:
        result["valid"] = False
        result["errors"].append(
            f"Manifest track_counts mismatch: expected {declared_track_counts}, got {observed_track_counts}"
        )
    for track_name, minimum in preset.required_track_cases.items():
        observed = observed_track_counts.get(track_name, 0)
        if observed < int(minimum):
            result["valid"] = False
            result["errors"].append(
                f"Track coverage below preset requirement: {track_name} has {observed}, requires {minimum}"
            )
    result["checks"].append("dataset_contract")

    # Baseline-report reproducibility validation
    predictions_path = Path(preset.predictions_path or "")
    baseline_report_path = Path(preset.baseline_report_path or "")
    if not predictions_path.exists():
        result["valid"] = False
        result["errors"].append(f"Missing predictions path from preset: {predictions_path}")
        return result
    if not baseline_report_path.exists():
        result["valid"] = False
        result["errors"].append(f"Missing baseline report path from preset: {baseline_report_path}")
        return result
    prediction_rows = _load_jsonl(predictions_path)
    actual_report = _load_json(baseline_report_path)
    expected_report = _build_expected_baseline_report(
        preset_name=preset_name,
        dataset_rows=dataset_rows,
        prediction_rows=prediction_rows,
    )
    mismatches = list(_iter_mismatches(expected_report, actual_report))
    if mismatches:
        result["valid"] = False
        result["errors"].extend(mismatches[:25])
        if len(mismatches) > 25:
            result["errors"].append(f"... {len(mismatches) - 25} additional mismatches omitted")
    result["checks"].append("baseline_reproducibility")

    return result


def verify_all_preset_artifacts(
    *,
    artifact_root: Path = Path("eval/artifacts"),
    preset_names: list[str] | None = None,
) -> dict[str, Any]:
    """Verify all preset artifact bundles and return aggregate status."""
    names = preset_names or sorted(PRESETS.keys())
    results: list[dict[str, Any]] = []
    overall_valid = True
    for name in names:
        try:
            result = verify_preset_artifacts(
                preset_name=name,
                artifact_root=artifact_root,
            )
        except Exception as exc:
            result = {
                "preset": name,
                "artifact_dir": str(artifact_root / name),
                "valid": False,
                "checks": [],
                "errors": [str(exc)],
            }
        results.append(result)
        if not bool(result.get("valid")):
            overall_valid = False
    return {
        "valid": overall_valid,
        "checked_presets": names,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify eval artifact integrity and reproducibility.")
    parser.add_argument(
        "command",
        choices=["verify"],
        help="Verification command.",
    )
    parser.add_argument(
        "--preset",
        default="vibecoder_memoryagentbench_v1",
        help="Preset/artifact bundle name.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Verify all known preset artifact bundles.",
    )
    args = parser.parse_args()

    if args.command == "verify":
        if args.all:
            result = verify_all_preset_artifacts()
        else:
            result = verify_preset_artifacts(preset_name=args.preset)
        print(json.dumps(result, indent=2))
        return 0 if result.get("valid") else 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
