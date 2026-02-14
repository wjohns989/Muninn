"""Quality gate checks for eval reports."""

from __future__ import annotations

from typing import Any, Dict, List


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compare_reports(
    current: Dict[str, Any],
    baseline: Dict[str, Any],
    *,
    max_metric_regression: float = 0.0,
) -> List[str]:
    """
    Compare current report to baseline.

    Returns a list of violations. Empty list means pass.
    """
    violations: List[str] = []
    current_cutoffs = current.get("cutoffs", {})
    baseline_cutoffs = baseline.get("cutoffs", {})

    for cutoff, baseline_metrics in baseline_cutoffs.items():
        current_metrics = current_cutoffs.get(cutoff)
        if not isinstance(current_metrics, dict):
            violations.append(f"Missing cutoff {cutoff} in current report")
            continue
        for metric in ("recall", "mrr", "ndcg"):
            base = _as_float(baseline_metrics.get(metric))
            cur = _as_float(current_metrics.get(metric))
            delta = cur - base
            if delta < -abs(max_metric_regression):
                violations.append(
                    f"{cutoff}.{metric} regressed by {delta:.4f} (baseline={base:.4f}, current={cur:.4f})"
                )
    return violations


def compare_track_reports(
    current: Dict[str, Any],
    baseline: Dict[str, Any],
    *,
    max_metric_regression: float = 0.0,
) -> List[str]:
    """
    Compare per-track metrics in current report against baseline report.

    Returns violations for missing tracks/cutoffs and over-budget regressions.
    """
    violations: List[str] = []
    current_tracks = current.get("tracks", {})
    baseline_tracks = baseline.get("tracks", {})
    if not isinstance(baseline_tracks, dict) or not baseline_tracks:
        return violations
    if not isinstance(current_tracks, dict):
        violations.append("Missing tracks section in current report")
        return violations

    for track_name, baseline_track in baseline_tracks.items():
        current_track = current_tracks.get(track_name)
        if not isinstance(current_track, dict):
            violations.append(f"Missing track {track_name} in current report")
            continue
        baseline_cutoffs = baseline_track.get("cutoffs", {})
        current_cutoffs = current_track.get("cutoffs", {})
        for cutoff, baseline_metrics in baseline_cutoffs.items():
            current_metrics = current_cutoffs.get(cutoff)
            if not isinstance(current_metrics, dict):
                violations.append(f"Missing track cutoff {track_name}.{cutoff} in current report")
                continue
            for metric in ("recall", "mrr", "ndcg"):
                base = _as_float(baseline_metrics.get(metric))
                cur = _as_float(current_metrics.get(metric))
                delta = cur - base
                if delta < -abs(max_metric_regression):
                    violations.append(
                        (
                            f"track {track_name} {cutoff}.{metric} regressed by {delta:.4f} "
                            f"(baseline={base:.4f}, current={cur:.4f})"
                        )
                    )
    return violations


def check_latency_budget(report: Dict[str, Any], *, max_p95_ms: float) -> List[str]:
    """Validate p95 latency budget."""
    violations: List[str] = []
    latency = report.get("latency_ms", {})
    p95 = _as_float(latency.get("p95"))
    if p95 > max_p95_ms:
        violations.append(f"latency.p95 {p95:.2f}ms exceeds budget {max_p95_ms:.2f}ms")
    return violations


def check_track_coverage(
    report: Dict[str, Any],
    *,
    required_track_cases: Dict[str, int],
) -> List[str]:
    """Validate that required tracks exist and meet minimum case counts."""
    violations: List[str] = []
    if not required_track_cases:
        return violations

    tracks = report.get("tracks", {})
    if not isinstance(tracks, dict):
        return ["Missing tracks section in report"]

    for track_name, min_cases in required_track_cases.items():
        track_entry = tracks.get(track_name)
        if not isinstance(track_entry, dict):
            violations.append(f"Missing required track '{track_name}'")
            continue
        observed = int(_as_float(track_entry.get("cases"), 0.0))
        required = max(1, int(min_cases))
        if observed < required:
            violations.append(
                f"Track '{track_name}' has {observed} cases, below required minimum {required}"
            )
    return violations
