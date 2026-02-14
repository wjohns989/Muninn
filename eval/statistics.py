"""Statistical utilities for paired retrieval-evaluation comparisons."""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence, Dict, Any, List, Literal


CorrectionMethod = Literal["none", "bonferroni", "holm", "bh"]


def mean(values: Sequence[float]) -> float:
    """Compute arithmetic mean with empty-sequence safety."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def sample_std(values: Sequence[float]) -> float:
    """Sample standard deviation (n-1 denominator)."""
    n = len(values)
    if n < 2:
        return 0.0
    mu = mean(values)
    var = sum((x - mu) * (x - mu) for x in values) / (n - 1)
    if var <= 0.0:
        return 0.0
    return math.sqrt(var)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    alpha: float = 0.05,
    samples: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Percentile bootstrap confidence interval for mean(values).
    """
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        v = float(values[0])
        return (v, v)
    draws = max(200, int(samples))
    rng = random.Random(seed)
    n = len(values)
    boot_means = []
    for _ in range(draws):
        sample = [values[rng.randrange(0, n)] for _ in range(n)]
        boot_means.append(mean(sample))
    boot_means.sort()
    lower_idx = max(0, min(len(boot_means) - 1, int((alpha / 2.0) * len(boot_means))))
    upper_idx = max(0, min(len(boot_means) - 1, int((1.0 - alpha / 2.0) * len(boot_means)) - 1))
    return (boot_means[lower_idx], boot_means[upper_idx])


def permutation_p_value(
    deltas: Sequence[float],
    *,
    rounds: int = 4000,
    seed: int = 42,
) -> float:
    """
    Two-sided paired randomization p-value via sign-flip permutations.
    """
    if not deltas:
        return 1.0
    observed = abs(mean(deltas))
    if observed == 0.0:
        return 1.0

    runs = max(500, int(rounds))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(runs):
        flipped = [x if rng.random() < 0.5 else -x for x in deltas]
        if abs(mean(flipped)) >= observed:
            extreme += 1
    # +1 correction prevents zero p-values at finite rounds.
    return (extreme + 1) / (runs + 1)


def summarize_paired_deltas(
    deltas: Sequence[float],
    *,
    alpha: float = 0.05,
    bootstrap_samples: int = 2000,
    permutation_rounds: int = 4000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Summarize paired metric deltas with CI, permutation p-value, and effect size.

    Deltas should be `current_metric - baseline_metric` per query.
    """
    if not deltas:
        return {
            "n": 0,
            "mean_delta": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_value": 1.0,
            "p_value_adjusted": 1.0,
            "cohens_d": 0.0,
            "significant_raw": False,
            "significant_regression_raw": False,
            "significant": False,
            "significant_regression": False,
        }

    mu = mean(deltas)
    ci_low, ci_high = bootstrap_mean_ci(
        deltas,
        alpha=alpha,
        samples=bootstrap_samples,
        seed=seed,
    )
    p_value = permutation_p_value(
        deltas,
        rounds=permutation_rounds,
        seed=seed + 1,
    )
    sigma = sample_std(deltas)
    cohens_d = (mu / sigma) if sigma > 0.0 else 0.0
    significant_raw = p_value < alpha
    significant_regression_raw = significant_raw and mu < 0.0 and ci_high < 0.0
    return {
        "n": len(deltas),
        "mean_delta": mu,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "p_value_adjusted": p_value,
        "cohens_d": cohens_d,
        "significant_raw": significant_raw,
        "significant_regression_raw": significant_regression_raw,
        "significant": significant_raw,
        "significant_regression": significant_regression_raw,
    }


def adjust_p_values(
    p_values: Sequence[float],
    *,
    method: CorrectionMethod = "none",
) -> List[float]:
    """
    Apply multiple-comparison correction to a sequence of p-values.

    Supported methods:
    - none: no correction
    - bonferroni: family-wise error control via m * p
    - holm: Holm-Bonferroni step-down FWER control
    - bh: Benjamini-Hochberg FDR control
    """
    m = len(p_values)
    if m == 0:
        return []

    clipped = [max(0.0, min(1.0, float(p))) for p in p_values]
    if method == "none":
        return clipped

    if method == "bonferroni":
        return [min(1.0, p * m) for p in clipped]

    indexed = sorted(enumerate(clipped), key=lambda x: x[1])

    if method == "holm":
        # Step-down: adjusted p_i = max_{j<=i} ((m-j+1) * p_(j))
        adjusted_sorted = [0.0] * m
        running_max = 0.0
        for rank, (_, p) in enumerate(indexed, start=1):
            candidate = (m - rank + 1) * p
            running_max = max(running_max, candidate)
            adjusted_sorted[rank - 1] = min(1.0, running_max)
        adjusted = [0.0] * m
        for rank, (original_idx, _) in enumerate(indexed, start=1):
            adjusted[original_idx] = adjusted_sorted[rank - 1]
        return adjusted

    if method == "bh":
        # Step-up: adjusted p_i = min_{j>=i} (m/j * p_(j))
        adjusted_sorted = [0.0] * m
        running_min = 1.0
        for rank in range(m, 0, -1):
            _, p = indexed[rank - 1]
            candidate = (m / rank) * p
            running_min = min(running_min, candidate)
            adjusted_sorted[rank - 1] = min(1.0, running_min)
        adjusted = [0.0] * m
        for rank, (original_idx, _) in enumerate(indexed, start=1):
            adjusted[original_idx] = adjusted_sorted[rank - 1]
        return adjusted

    raise ValueError(f"Unsupported correction method: {method}")


def apply_multiple_testing_correction(
    summaries: Sequence[Dict[str, Any]],
    *,
    alpha: float,
    method: CorrectionMethod = "none",
) -> None:
    """
    Mutate significance summaries to include adjusted p-values and decisions.
    """
    if not summaries:
        return

    p_values = [float(summary.get("p_value", 1.0)) for summary in summaries]
    adjusted = adjust_p_values(p_values, method=method)
    for summary, p_adj in zip(summaries, adjusted):
        summary["p_value_adjusted"] = float(p_adj)
        significant = bool(p_adj < alpha)
        summary["significant"] = significant
        mean_delta = float(summary.get("mean_delta", 0.0))
        ci_high = float(summary.get("ci_high", 0.0))
        summary["significant_regression"] = bool(significant and mean_delta < 0.0 and ci_high < 0.0)


def group_deltas(
    rows: Iterable[Dict[str, Any]],
    *,
    key_fields: Sequence[str],
    value_field: str = "delta",
) -> Dict[str, list[float]]:
    """Group numeric delta rows by a stable string key over selected fields."""
    grouped: Dict[str, list[float]] = {}
    for row in rows:
        key = "|".join(str(row.get(field, "")) for field in key_fields)
        value = row.get(value_field)
        if not isinstance(value, (int, float)):
            continue
        grouped.setdefault(key, []).append(float(value))
    return grouped
