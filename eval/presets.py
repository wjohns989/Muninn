"""Evaluation preset profiles for reproducible release gating."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from eval.statistics import CorrectionMethod


@dataclass(frozen=True)
class EvalPreset:
    """Named profile for eval defaults and release-gate policy."""

    name: str
    description: str
    ks: str = "5,10"
    dataset_path: Optional[str] = None
    predictions_path: Optional[str] = None
    baseline_report_path: Optional[str] = None
    max_metric_regression: float = 0.01
    max_track_metric_regression: float = 0.01
    max_p95_latency_ms: Optional[float] = None
    required_track_cases: Dict[str, int] = field(default_factory=dict)
    significance_alpha: float = 0.05
    bootstrap_samples: int = 2000
    permutation_rounds: int = 4000
    gate_significant_regressions: bool = False
    significance_correction: CorrectionMethod = "none"
    significance_correction_family: str = "all"


PRESETS: Dict[str, EvalPreset] = {
    "vibecoder_memoryagentbench_v1": EvalPreset(
        name="vibecoder_memoryagentbench_v1",
        description=(
            "Memory-agent competency gate profile aligned to MemoryAgentBench-style tracks."
        ),
        ks="5,10",
        dataset_path="eval/artifacts/vibecoder_memoryagentbench_v1/dataset.jsonl",
        predictions_path="eval/artifacts/vibecoder_memoryagentbench_v1/baseline_predictions.jsonl",
        baseline_report_path="eval/artifacts/vibecoder_memoryagentbench_v1/baseline_report.json",
        max_metric_regression=0.01,
        max_track_metric_regression=0.015,
        max_p95_latency_ms=120.0,
        required_track_cases={
            "accurate_retrieval": 22,
            "test_time_learning": 6,
            "long_range_understanding": 110,
            "conflict_resolution": 8,
        },
        significance_alpha=0.05,
        bootstrap_samples=2000,
        permutation_rounds=4000,
        gate_significant_regressions=True,
        significance_correction="holm",
        significance_correction_family="by_track",
    ),
    "vibecoder_memoryagentbench_stress_v1": EvalPreset(
        name="vibecoder_memoryagentbench_stress_v1",
        description=(
            "Robustness stress slice with harder negatives and elevated latency profile."
        ),
        ks="5,10",
        dataset_path="eval/artifacts/vibecoder_memoryagentbench_stress_v1/dataset.jsonl",
        predictions_path="eval/artifacts/vibecoder_memoryagentbench_stress_v1/baseline_predictions.jsonl",
        baseline_report_path="eval/artifacts/vibecoder_memoryagentbench_stress_v1/baseline_report.json",
        max_metric_regression=0.015,
        max_track_metric_regression=0.02,
        max_p95_latency_ms=170.0,
        required_track_cases={
            "accurate_retrieval": 16,
            "test_time_learning": 6,
            "long_range_understanding": 30,
            "conflict_resolution": 8,
        },
        significance_alpha=0.05,
        bootstrap_samples=2000,
        permutation_rounds=4000,
        gate_significant_regressions=True,
        significance_correction="holm",
        significance_correction_family="by_track",
    )
}


def get_preset(name: str) -> EvalPreset:
    """Resolve a named preset or raise ValueError."""
    preset = PRESETS.get(name)
    if preset is None:
        available = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown eval preset '{name}'. Available presets: {available}")
    return preset


def resolve_optional_path(path_value: Optional[str]) -> Optional[Path]:
    """Convert optional path string to Path."""
    if not path_value:
        return None
    return Path(path_value)
