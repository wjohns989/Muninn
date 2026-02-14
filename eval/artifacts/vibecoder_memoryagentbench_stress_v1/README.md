# vibecoder_memoryagentbench_stress_v1 Artifacts

Canonical robustness stress-slice bundle for Muninn's preset-driven eval gate.

## Purpose

This bundle extends coverage beyond the primary baseline by using a deterministic,
cross-track stress slice with:
- harder negative competition in ranked outputs,
- occasional controlled misses on challenging tracks,
- elevated latency profile for budget pressure testing.

## Files

- `dataset.jsonl`: stratified 60-case subset with `query_id`, `track`, `relevant_ids`.
- `baseline_predictions.jsonl`: deterministic stress rankings with `latency_ms`.
- `baseline_report.json`: canonical report from `python -m eval.run --preset vibecoder_memoryagentbench_stress_v1 --skip-baseline-compare`.
- `manifest.json`: SHA-256 checksums + dataset contract metadata.

## Integrity + Reproducibility Verification

```bash
python -m eval.artifacts verify --preset vibecoder_memoryagentbench_stress_v1
```

