"""CLI for Muninn retrieval evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.gates import (
    check_latency_budget,
    check_track_coverage,
    compare_reports,
    compare_track_reports,
)
from eval.metrics import evaluate_batch, evaluate_case
from eval.presets import PRESETS, get_preset, resolve_optional_path
from eval.statistics import (
    CorrectionMethod,
    apply_multiple_testing_correction,
    summarize_paired_deltas,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _merge_cases(
    dataset_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_query_id = {row["query_id"]: row for row in prediction_rows if "query_id" in row}
    cases: list[dict[str, Any]] = []

    for row in dataset_rows:
        query_id = row.get("query_id")
        if not query_id:
            continue
        prediction = by_query_id.get(query_id, {})
        ranked_ids = prediction.get("ranked_ids", [])
        if not isinstance(ranked_ids, list):
            ranked_ids = []
        latency_ms = prediction.get("latency_ms")
        if not isinstance(latency_ms, (int, float)):
            latency_ms = None
        relevant_ids = row.get("relevant_ids", [])
        if not isinstance(relevant_ids, list):
            relevant_ids = []
        track = row.get("track")
        if not isinstance(track, str):
            track = None
        cases.append(
            {
                "query_id": query_id,
                "relevant_ids": relevant_ids,
                "ranked_ids": ranked_ids,
                "latency_ms": latency_ms,
                "track": track,
            }
        )
    return cases


def _parse_required_track_args(values: list[str]) -> dict[str, int]:
    """
    Parse repeatable TRACK:MIN_CASES CLI args into a dictionary.
    """
    required: dict[str, int] = {}
    for raw in values:
        value = (raw or "").strip()
        if not value:
            continue
        if ":" not in value:
            raise ValueError(
                f"Invalid --required-track '{value}'. Expected format TRACK:MIN_CASES"
            )
        track, min_cases_str = value.split(":", 1)
        track_name = track.strip()
        if not track_name:
            raise ValueError(
                f"Invalid --required-track '{value}'. Track name cannot be empty"
            )
        try:
            min_cases = int(min_cases_str.strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid --required-track '{value}'. MIN_CASES must be an integer"
            ) from exc
        if min_cases <= 0:
            raise ValueError(
                f"Invalid --required-track '{value}'. MIN_CASES must be > 0"
            )
        required[track_name] = min_cases
    return required


def _build_metric_map(
    cases: list[dict[str, Any]],
    *,
    ks: list[int],
) -> dict[str, dict[int, dict[str, float]]]:
    """
    Build query_id -> k -> metric_name map for paired significance comparisons.
    """
    metric_map: dict[str, dict[int, dict[str, float]]] = {}
    for case in cases:
        query_id = case.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            continue
        relevant_ids = case.get("relevant_ids", [])
        ranked_ids = case.get("ranked_ids", [])
        by_k: dict[int, dict[str, float]] = {}
        for k in ks:
            by_k[k] = evaluate_case(relevant_ids, ranked_ids, k)
        metric_map[query_id] = by_k
    return metric_map


def _compute_significance(
    *,
    current_cases: list[dict[str, Any]],
    baseline_cases: list[dict[str, Any]],
    ks: list[int],
    alpha: float,
    bootstrap_samples: int,
    permutation_rounds: int,
    seed: int,
) -> dict[str, Any]:
    """
    Compute paired significance/effect-size summary for global and track slices.
    """
    current_metric_map = _build_metric_map(current_cases, ks=ks)
    baseline_metric_map = _build_metric_map(baseline_cases, ks=ks)

    shared_query_ids = sorted(set(current_metric_map).intersection(baseline_metric_map))
    track_by_query_id = {
        case["query_id"]: case.get("track")
        for case in current_cases
        if isinstance(case.get("query_id"), str)
    }

    significance: dict[str, Any] = {
        "shared_queries": len(shared_query_ids),
        "alpha": alpha,
        "bootstrap_samples": bootstrap_samples,
        "permutation_rounds": permutation_rounds,
        "seed": seed,
        "global": {},
        "tracks": {},
    }

    for k in ks:
        cutoff_key = f"@{k}"
        significance["global"][cutoff_key] = {}
        for metric_name in ("recall", "mrr", "ndcg"):
            deltas = []
            for query_id in shared_query_ids:
                cur = current_metric_map[query_id][k].get(metric_name, 0.0)
                base = baseline_metric_map[query_id][k].get(metric_name, 0.0)
                deltas.append(float(cur) - float(base))
            significance["global"][cutoff_key][metric_name] = summarize_paired_deltas(
                deltas,
                alpha=alpha,
                bootstrap_samples=bootstrap_samples,
                permutation_rounds=permutation_rounds,
                seed=seed + (k * 10),
            )

    # Track-level paired deltas
    track_names = sorted(
        {
            track_by_query_id.get(query_id)
            for query_id in shared_query_ids
            if isinstance(track_by_query_id.get(query_id), str) and track_by_query_id.get(query_id)
        }
    )
    for track_name in track_names:
        track_query_ids = [qid for qid in shared_query_ids if track_by_query_id.get(qid) == track_name]
        significance["tracks"][track_name] = {"queries": len(track_query_ids), "cutoffs": {}}
        for k in ks:
            cutoff_key = f"@{k}"
            significance["tracks"][track_name]["cutoffs"][cutoff_key] = {}
            for metric_name in ("recall", "mrr", "ndcg"):
                deltas = []
                for query_id in track_query_ids:
                    cur = current_metric_map[query_id][k].get(metric_name, 0.0)
                    base = baseline_metric_map[query_id][k].get(metric_name, 0.0)
                    deltas.append(float(cur) - float(base))
                significance["tracks"][track_name]["cutoffs"][cutoff_key][metric_name] = summarize_paired_deltas(
                    deltas,
                    alpha=alpha,
                    bootstrap_samples=bootstrap_samples,
                    permutation_rounds=permutation_rounds,
                    seed=seed + (k * 100),
                )

    return significance


def _collect_global_significance_summaries(significance: dict[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for metric_summary in significance.get("global", {}).values():
        if not isinstance(metric_summary, dict):
            continue
        for summary in metric_summary.values():
            if isinstance(summary, dict):
                summaries.append(summary)
    return summaries


def _collect_track_significance_summaries(
    significance: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    tracks = significance.get("tracks", {})
    if not isinstance(tracks, dict):
        return grouped
    for track_name, track_data in tracks.items():
        if not isinstance(track_name, str) or not isinstance(track_data, dict):
            continue
        cutoffs = track_data.get("cutoffs", {})
        if not isinstance(cutoffs, dict):
            continue
        bucket: list[dict[str, Any]] = []
        for metric_summary in cutoffs.values():
            if not isinstance(metric_summary, dict):
                continue
            for summary in metric_summary.values():
                if isinstance(summary, dict):
                    bucket.append(summary)
        grouped[track_name] = bucket
    return grouped


def _apply_significance_correction(
    significance: dict[str, Any],
    *,
    alpha: float,
    method: CorrectionMethod,
    family: str,
) -> None:
    """
    Apply multiple-comparison correction to significance summaries in-place.

    family:
      - all: one family across global + all tracks
      - by_track: one family for global + one per track
    """
    global_summaries = _collect_global_significance_summaries(significance)
    grouped_tracks = _collect_track_significance_summaries(significance)

    if method == "none":
        apply_multiple_testing_correction(global_summaries, alpha=alpha, method=method)
        for bucket in grouped_tracks.values():
            apply_multiple_testing_correction(bucket, alpha=alpha, method=method)
        families_applied = {"global": len(global_summaries), "tracks": {k: len(v) for k, v in grouped_tracks.items()}}
    elif family == "all":
        combined = list(global_summaries)
        for bucket in grouped_tracks.values():
            combined.extend(bucket)
        apply_multiple_testing_correction(combined, alpha=alpha, method=method)
        families_applied = {"all": len(combined)}
    elif family == "by_track":
        apply_multiple_testing_correction(global_summaries, alpha=alpha, method=method)
        for bucket in grouped_tracks.values():
            apply_multiple_testing_correction(bucket, alpha=alpha, method=method)
        families_applied = {"global": len(global_summaries), "tracks": {k: len(v) for k, v in grouped_tracks.items()}}
    else:
        raise ValueError(f"Unsupported significance correction family: {family}")

    significance["multiple_testing"] = {
        "method": method,
        "family": family,
        "families_applied": families_applied,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Muninn retrieval output")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        help="Optional preset profile for default gates/coverage requirements.",
    )
    parser.add_argument("--dataset", help="Ground-truth JSONL with query_id + relevant_ids")
    parser.add_argument("--predictions", help="Prediction JSONL with query_id + ranked_ids")
    parser.add_argument("--ks", help="Comma-separated cutoffs, e.g. 5,10,20")
    parser.add_argument("--output", help="Optional output JSON path")
    parser.add_argument("--baseline-report", help="Optional baseline JSON report for regression checks")
    parser.add_argument(
        "--skip-baseline-compare",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip baseline report comparison even if a preset defines baseline_report_path.",
    )
    parser.add_argument(
        "--baseline-predictions",
        help=(
            "Optional baseline predictions JSONL for paired significance/effect-size analysis. "
            "Must share query_id keys with --dataset."
        ),
    )
    parser.add_argument(
        "--max-metric-regression",
        type=float,
        help="Allowed absolute regression for recall/mrr/ndcg per cutoff.",
    )
    parser.add_argument(
        "--max-track-metric-regression",
        type=float,
        help="Allowed absolute regression for recall/mrr/ndcg at track level.",
    )
    parser.add_argument(
        "--max-p95-latency-ms",
        type=float,
        help="Optional p95 latency budget in milliseconds.",
    )
    parser.add_argument(
        "--required-track",
        action="append",
        default=[],
        help="Repeatable required track coverage gate in TRACK:MIN_CASES format.",
    )
    parser.add_argument(
        "--significance-alpha",
        type=float,
        default=None,
        help="Alpha level for paired significance tests.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=None,
        help="Bootstrap sample count for confidence intervals.",
    )
    parser.add_argument(
        "--permutation-rounds",
        type=int,
        default=None,
        help="Permutation rounds for paired randomization p-values.",
    )
    parser.add_argument(
        "--significance-seed",
        type=int,
        default=42,
        help="Deterministic random seed for significance calculations.",
    )
    parser.add_argument(
        "--gate-significant-regressions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fail gate when significant regressions are detected from paired significance analysis.",
    )
    parser.add_argument(
        "--significance-correction",
        choices=["none", "bonferroni", "holm", "bh"],
        default=None,
        help="Multiple-comparison correction method applied to significance p-values.",
    )
    parser.add_argument(
        "--significance-correction-family",
        choices=["all", "by_track"],
        default=None,
        help="Family scope for multiple-comparison correction.",
    )
    args = parser.parse_args()

    preset = get_preset(args.preset) if args.preset else None

    dataset_path = Path(args.dataset) if args.dataset else resolve_optional_path(preset.dataset_path if preset else None)
    prediction_path = Path(args.predictions) if args.predictions else resolve_optional_path(
        preset.predictions_path if preset else None
    )
    if dataset_path is None or prediction_path is None:
        raise ValueError(
            "Both --dataset and --predictions are required unless provided by the selected preset."
        )

    dataset_rows = _load_jsonl(dataset_path)
    prediction_rows = _load_jsonl(prediction_path)

    ks_source = args.ks or (preset.ks if preset else "5,10")
    ks = [int(x.strip()) for x in ks_source.split(",") if x.strip()]
    cases = _merge_cases(dataset_rows, prediction_rows)
    report = evaluate_batch(cases, ks=ks)
    report["dataset_size"] = len(dataset_rows)
    report["predictions_size"] = len(prediction_rows)
    report["matched_queries"] = len(cases)
    report["gates"] = {"passed": True, "violations": []}

    required_track_cases = {}
    if preset:
        required_track_cases.update(preset.required_track_cases)
    required_track_cases.update(_parse_required_track_args(args.required_track))

    max_metric_regression = (
        args.max_metric_regression
        if args.max_metric_regression is not None
        else (preset.max_metric_regression if preset else 0.0)
    )
    max_track_metric_regression = (
        args.max_track_metric_regression
        if args.max_track_metric_regression is not None
        else (preset.max_track_metric_regression if preset else max_metric_regression)
    )
    max_p95_latency_ms = (
        args.max_p95_latency_ms
        if args.max_p95_latency_ms is not None
        else (preset.max_p95_latency_ms if preset else None)
    )
    significance_alpha = (
        args.significance_alpha
        if args.significance_alpha is not None
        else (preset.significance_alpha if preset else 0.05)
    )
    bootstrap_samples = (
        args.bootstrap_samples
        if args.bootstrap_samples is not None
        else (preset.bootstrap_samples if preset else 2000)
    )
    permutation_rounds = (
        args.permutation_rounds
        if args.permutation_rounds is not None
        else (preset.permutation_rounds if preset else 4000)
    )
    gate_significant_regressions = (
        args.gate_significant_regressions
        if args.gate_significant_regressions is not None
        else (preset.gate_significant_regressions if preset else False)
    )
    significance_correction: CorrectionMethod = (
        args.significance_correction
        if args.significance_correction is not None
        else (preset.significance_correction if preset else "none")
    )
    significance_correction_family = (
        args.significance_correction_family
        if args.significance_correction_family is not None
        else (preset.significance_correction_family if preset else "all")
    )

    baseline_path: Path | None = None
    if args.skip_baseline_compare:
        baseline_path = None
    elif args.baseline_report:
        baseline_path = Path(args.baseline_report)
    elif preset and preset.baseline_report_path:
        baseline_path = Path(preset.baseline_report_path)

    violations: list[str] = []
    if baseline_path is not None:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        violations.extend(
            compare_reports(
                report,
                baseline,
                max_metric_regression=max_metric_regression,
            )
        )
        violations.extend(
            compare_track_reports(
                report,
                baseline,
                max_metric_regression=max_track_metric_regression,
            )
        )

    baseline_predictions_path: Path | None = None
    if args.baseline_predictions:
        baseline_predictions_path = Path(args.baseline_predictions)
    baseline_significance: dict[str, Any] | None = None
    if baseline_predictions_path is not None:
        baseline_prediction_rows = _load_jsonl(baseline_predictions_path)
        baseline_cases = _merge_cases(dataset_rows, baseline_prediction_rows)
        baseline_significance = _compute_significance(
            current_cases=cases,
            baseline_cases=baseline_cases,
            ks=ks,
            alpha=max(1e-6, min(0.5, float(significance_alpha))),
            bootstrap_samples=max(200, int(bootstrap_samples)),
            permutation_rounds=max(500, int(permutation_rounds)),
            seed=int(args.significance_seed),
        )
        _apply_significance_correction(
            baseline_significance,
            alpha=max(1e-6, min(0.5, float(significance_alpha))),
            method=significance_correction,
            family=significance_correction_family,
        )
        report["significance"] = baseline_significance

        if gate_significant_regressions:
            for cutoff, metric_summary in baseline_significance.get("global", {}).items():
                if not isinstance(metric_summary, dict):
                    continue
                for metric_name, summary in metric_summary.items():
                    if not isinstance(summary, dict):
                        continue
                    if bool(summary.get("significant_regression")):
                        violations.append(
                            (
                                f"global {cutoff}.{metric_name} significant regression: "
                                f"delta={summary.get('mean_delta', 0.0):.4f}, "
                                f"p={summary.get('p_value', 1.0):.4f}, "
                                f"ci=[{summary.get('ci_low', 0.0):.4f},{summary.get('ci_high', 0.0):.4f}]"
                            )
                        )
            for track_name, track_data in baseline_significance.get("tracks", {}).items():
                if not isinstance(track_data, dict):
                    continue
                cutoffs = track_data.get("cutoffs", {})
                if not isinstance(cutoffs, dict):
                    continue
                for cutoff, metric_summary in cutoffs.items():
                    if not isinstance(metric_summary, dict):
                        continue
                    for metric_name, summary in metric_summary.items():
                        if not isinstance(summary, dict):
                            continue
                        if bool(summary.get("significant_regression")):
                            violations.append(
                                (
                                    f"track {track_name} {cutoff}.{metric_name} significant regression: "
                                    f"delta={summary.get('mean_delta', 0.0):.4f}, "
                                    f"p={summary.get('p_value', 1.0):.4f}, "
                                    f"ci=[{summary.get('ci_low', 0.0):.4f},{summary.get('ci_high', 0.0):.4f}]"
                                )
                            )

    if max_p95_latency_ms is not None:
        violations.extend(check_latency_budget(report, max_p95_ms=max_p95_latency_ms))

    violations.extend(
        check_track_coverage(
            report,
            required_track_cases=required_track_cases,
        )
    )

    report["gate_config"] = {
        "preset": preset.name if preset else None,
        "max_metric_regression": max_metric_regression,
        "max_track_metric_regression": max_track_metric_regression,
        "max_p95_latency_ms": max_p95_latency_ms,
        "required_track_cases": required_track_cases,
        "skip_baseline_compare": bool(args.skip_baseline_compare),
        "baseline_predictions": str(baseline_predictions_path) if baseline_predictions_path else None,
        "significance_alpha": max(1e-6, min(0.5, float(significance_alpha))),
        "bootstrap_samples": max(200, int(bootstrap_samples)),
        "permutation_rounds": max(500, int(permutation_rounds)),
        "significance_seed": int(args.significance_seed),
        "gate_significant_regressions": bool(gate_significant_regressions),
        "significance_correction": significance_correction,
        "significance_correction_family": significance_correction_family,
    }

    if violations:
        report["gates"]["passed"] = False
        report["gates"]["violations"] = violations

    rendered = json.dumps(report, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0 if report["gates"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
