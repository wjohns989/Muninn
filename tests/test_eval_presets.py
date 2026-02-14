import pytest

from eval.presets import get_preset


def test_vibecoder_preset_significance_defaults():
    preset = get_preset("vibecoder_memoryagentbench_v1")
    assert preset.gate_significant_regressions is True
    assert preset.significance_alpha == 0.05
    assert preset.significance_correction == "holm"
    assert preset.significance_correction_family == "by_track"
    assert preset.bootstrap_samples >= 1000
    assert preset.permutation_rounds >= 2000
    assert preset.required_track_cases["accurate_retrieval"] == 22
    assert preset.required_track_cases["long_range_understanding"] == 110


def test_stress_preset_defaults():
    preset = get_preset("vibecoder_memoryagentbench_stress_v1")
    assert preset.significance_correction == "holm"
    assert preset.significance_correction_family == "by_track"
    assert preset.max_p95_latency_ms == 170.0
    assert preset.required_track_cases["accurate_retrieval"] == 16
    assert preset.required_track_cases["long_range_understanding"] == 30


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        get_preset("does-not-exist")
