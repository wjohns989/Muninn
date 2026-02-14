"""Retrieval evaluation metrics for Muninn roadmap benchmarks."""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Mapping, Any


def recall_at_k(relevant_ids: set[str], ranked_ids: Sequence[str], k: int) -> float:
    """Compute Recall@k with binary relevance."""
    if k <= 0 or not relevant_ids:
        return 0.0
    top_k_unique = set(ranked_ids[:k])
    hits = len(top_k_unique.intersection(relevant_ids))
    return hits / len(relevant_ids)


def mrr_at_k(relevant_ids: set[str], ranked_ids: Sequence[str], k: int) -> float:
    """Compute MRR@k (single-query reciprocal rank)."""
    if k <= 0 or not relevant_ids:
        return 0.0
    for idx, item in enumerate(ranked_ids[:k], start=1):
        if item in relevant_ids:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(relevant_ids: set[str], ranked_ids: Sequence[str], k: int) -> float:
    """Compute nDCG@k with binary relevance."""
    if k <= 0 or not relevant_ids:
        return 0.0

    def _discount(position_1_based: int) -> float:
        # log2(pos + 1) in denominator
        return 1.0 / math.log2(position_1_based + 1)

    dcg = 0.0
    seen_relevant: set[str] = set()
    for idx, item in enumerate(ranked_ids[:k], start=1):
        if item in relevant_ids and item not in seen_relevant:
            dcg += _discount(idx)
            seen_relevant.add(item)

    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(_discount(i) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_case(relevant_ids: Iterable[str], ranked_ids: Sequence[str], k: int) -> dict[str, float]:
    """Evaluate a single query case at cutoff `k`."""
    relevant = set(relevant_ids)
    return {
        "recall": recall_at_k(relevant, ranked_ids, k),
        "mrr": mrr_at_k(relevant, ranked_ids, k),
        "ndcg": ndcg_at_k(relevant, ranked_ids, k),
    }


def summarize_latency_ms(latencies_ms: Sequence[float]) -> dict[str, float]:
    """Compute average/p50/p95 latency summary in milliseconds."""
    values = sorted(float(x) for x in latencies_ms if x is not None and x >= 0.0)
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "count": 0.0}

    def _pct(percent: float) -> float:
        if len(values) == 1:
            return values[0]
        idx = int(round((percent / 100.0) * (len(values) - 1)))
        idx = max(0, min(len(values) - 1, idx))
        return values[idx]

    return {
        "avg": sum(values) / len(values),
        "p50": _pct(50.0),
        "p95": _pct(95.0),
        "count": float(len(values)),
    }


def evaluate_batch(
    cases: Sequence[Mapping[str, Any]],
    ks: Sequence[int] = (5, 10),
) -> dict[str, Any]:
    """
    Evaluate a list of retrieval cases.

    Each case must provide:
    - `relevant_ids`: iterable of relevant memory ids
    - `ranked_ids`: ranked retrieval output
    """
    valid_ks = sorted({k for k in ks if k > 0})
    if not valid_ks:
        raise ValueError("At least one positive k is required")

    def _init_totals() -> dict[int, dict[str, float]]:
        return {k: {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0} for k in valid_ks}

    totals: dict[int, dict[str, float]] = _init_totals()
    case_count = 0
    latencies_ms: list[float] = []
    track_counts: dict[str, int] = {}
    track_totals: dict[str, dict[int, dict[str, float]]] = {}
    track_latencies: dict[str, list[float]] = {}

    for case in cases:
        ranked_ids = list(case.get("ranked_ids", []))
        relevant_ids = case.get("relevant_ids", [])
        latency_ms = case.get("latency_ms")
        per_k_metrics: dict[int, dict[str, float]] = {}
        if isinstance(latency_ms, (int, float)):
            latencies_ms.append(float(latency_ms))
        for k in valid_ks:
            metrics = evaluate_case(relevant_ids, ranked_ids, k)
            per_k_metrics[k] = metrics
            for metric_name, value in metrics.items():
                totals[k][metric_name] += value
        track = case.get("track")
        if isinstance(track, str) and track.strip():
            track_name = track.strip()
            if track_name not in track_totals:
                track_totals[track_name] = _init_totals()
                track_latencies[track_name] = []
                track_counts[track_name] = 0
            track_counts[track_name] += 1
            for k in valid_ks:
                metrics = per_k_metrics[k]
                for metric_name, value in metrics.items():
                    track_totals[track_name][k][metric_name] += value
            if isinstance(latency_ms, (int, float)):
                track_latencies[track_name].append(float(latency_ms))
        case_count += 1

    if case_count == 0:
        return {
            "cases": 0,
            "cutoffs": {
                f"@{k}": {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
                for k in valid_ks
            },
        }

    report = {
        "cases": case_count,
        "cutoffs": {
            f"@{k}": {
                metric_name: metric_sum / case_count
                for metric_name, metric_sum in totals[k].items()
            }
            for k in valid_ks
        },
        "latency_ms": summarize_latency_ms(latencies_ms),
    }
    if track_totals:
        report["tracks"] = {}
        for track_name, totals_by_k in track_totals.items():
            track_case_count = track_counts[track_name]
            report["tracks"][track_name] = {
                "cases": track_case_count,
                "cutoffs": {
                    f"@{k}": {
                        metric_name: metric_sum / track_case_count
                        for metric_name, metric_sum in totals_by_k[k].items()
                    }
                    for k in valid_ks
                },
                "latency_ms": summarize_latency_ms(track_latencies.get(track_name, [])),
            }
    return report
