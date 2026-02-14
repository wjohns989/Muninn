import pytest

from eval.statistics import (
    adjust_p_values,
    apply_multiple_testing_correction,
    bootstrap_mean_ci,
    mean,
    permutation_p_value,
    summarize_paired_deltas,
)


def test_mean_and_bootstrap_ci():
    values = [1.0, 2.0, 3.0, 4.0]
    assert mean(values) == 2.5
    low, high = bootstrap_mean_ci(values, alpha=0.05, samples=500, seed=11)
    assert low <= 2.5 <= high


def test_permutation_p_value_detects_large_shift():
    # Strong positive deltas should be highly significant.
    deltas = [0.4] * 30
    p_value = permutation_p_value(deltas, rounds=1000, seed=7)
    assert p_value < 0.01


def test_summarize_paired_deltas_significant_regression():
    deltas = [-0.2] * 40
    summary = summarize_paired_deltas(
        deltas,
        alpha=0.05,
        bootstrap_samples=800,
        permutation_rounds=1200,
        seed=3,
    )
    assert summary["significant"] is True
    assert summary["significant_regression"] is True
    assert summary["significant_raw"] is True
    assert summary["significant_regression_raw"] is True
    assert summary["mean_delta"] < 0.0
    assert summary["ci_high"] < 0.0


def test_adjust_p_values_holm_and_bh():
    p_values = [0.01, 0.03, 0.04]
    holm = adjust_p_values(p_values, method="holm")
    bh = adjust_p_values(p_values, method="bh")
    assert holm == pytest.approx([0.03, 0.06, 0.06])
    assert bh == pytest.approx([0.03, 0.04, 0.04])


def test_apply_multiple_testing_correction_updates_significance():
    summaries = [
        {"p_value": 0.01, "mean_delta": -0.2, "ci_high": -0.01},
        {"p_value": 0.04, "mean_delta": -0.2, "ci_high": -0.01},
    ]
    apply_multiple_testing_correction(summaries, alpha=0.05, method="bonferroni")
    assert summaries[0]["p_value_adjusted"] == pytest.approx(0.02)
    assert summaries[0]["significant"] is True
    assert summaries[0]["significant_regression"] is True
    assert summaries[1]["p_value_adjusted"] == pytest.approx(0.08)
    assert summaries[1]["significant"] is False
    assert summaries[1]["significant_regression"] is False
