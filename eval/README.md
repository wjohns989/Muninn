# Muninn Eval Harness (Phase 1.1B Starter)

This directory contains the first production-grade baseline for retrieval evaluation.

## Inputs

- `dataset.jsonl`: one object per line with:
  - `query_id` (string)
  - `relevant_ids` (array of memory IDs)
  - `track` (optional string competency label, e.g. `accurate_retrieval`)
- `predictions.jsonl`: one object per line with:
  - `query_id` (string)
  - `ranked_ids` (ordered array of retrieved memory IDs)
  - `latency_ms` (optional numeric query latency for budget checks)

## Run

```bash
python -m eval.run --dataset dataset.jsonl --predictions predictions.jsonl --ks 5,10

# Regression/budget gate mode
python -m eval.run \
  --dataset dataset.jsonl \
  --predictions predictions.jsonl \
  --baseline-report reports/baseline.json \
  --max-metric-regression 0.01 \
  --max-p95-latency-ms 120

# Preset-based policy profile (MemoryAgentBench-style coverage gates)
python -m eval.run \
  --preset vibecoder_memoryagentbench_v1 \
  --dataset current_dataset_override.jsonl \
  --predictions current_predictions_override.jsonl \
  --required-track accurate_retrieval:22 \
  --required-track test_time_learning:6 \
  --required-track long_range_understanding:110 \
  --required-track conflict_resolution:8

# Run canonical baseline report directly from preset artifact paths
python -m eval.run --preset vibecoder_memoryagentbench_v1
python -m eval.run --preset vibecoder_memoryagentbench_stress_v1

# Refresh baseline output intentionally without comparing to existing baseline report
python -m eval.run --preset vibecoder_memoryagentbench_v1 --skip-baseline-compare

# Paired significance/effect-size analysis against baseline predictions
python -m eval.run \
  --dataset dataset.jsonl \
  --predictions current_predictions.jsonl \
  --baseline-predictions baseline_predictions.jsonl \
  --gate-significant-regressions \
  --significance-correction holm \
  --significance-correction-family by_track \
  --significance-alpha 0.05 \
  --bootstrap-samples 2000 \
  --permutation-rounds 4000
```

## Output

The report includes:
- `Recall@k`
- `MRR@k`
- `nDCG@k`
- optional per-track breakdowns (`tracks`) when dataset rows include `track`
- optional paired significance/effect-size report (`significance`) when `--baseline-predictions` is supplied
  - includes raw and adjusted significance fields (`p_value`, `p_value_adjusted`, `significant_raw`, `significant`)
- latency summary (`avg`, `p50`, `p95`)
- case and match counts
- gate verdict (`passed`, `violations`) when baseline/budget args are provided
- explicit gate policy echo (`gate_config`) for auditability

This is intentionally lightweight and dependency-free so it can run in CI before larger benchmark integrations are added.

## Preset Gate Policy

- `vibecoder_memoryagentbench_v1`
  - Targeted to memory-agent competency slices inspired by MemoryAgentBench.
  - Default artifact paths:
    - `dataset`: `eval/artifacts/vibecoder_memoryagentbench_v1/dataset.jsonl`
    - `predictions`: `eval/artifacts/vibecoder_memoryagentbench_v1/baseline_predictions.jsonl`
    - `baseline report`: `eval/artifacts/vibecoder_memoryagentbench_v1/baseline_report.json`
  - Defaults:
    - `ks=5,10`
    - `max_metric_regression=0.01` (global cutoffs)
    - `max_track_metric_regression=0.015` (per-track cutoffs)
    - `max_p95_latency_ms=120`
    - `significance_correction=holm`
    - `significance_correction_family=by_track`
    - required track case minimums:
      - `accurate_retrieval`: 22
      - `test_time_learning`: 6
      - `long_range_understanding`: 110
      - `conflict_resolution`: 8

- `vibecoder_memoryagentbench_stress_v1`
  - Cross-track robustness slice with hard negatives and elevated latency pressure.
  - Default artifact paths:
    - `dataset`: `eval/artifacts/vibecoder_memoryagentbench_stress_v1/dataset.jsonl`
    - `predictions`: `eval/artifacts/vibecoder_memoryagentbench_stress_v1/baseline_predictions.jsonl`
    - `baseline report`: `eval/artifacts/vibecoder_memoryagentbench_stress_v1/baseline_report.json`
  - Defaults:
    - `ks=5,10`
    - `max_metric_regression=0.015` (global cutoffs)
    - `max_track_metric_regression=0.02` (per-track cutoffs)
    - `max_p95_latency_ms=170`
    - `significance_correction=holm`
    - `significance_correction_family=by_track`
    - required track case minimums:
      - `accurate_retrieval`: 16
      - `test_time_learning`: 6
      - `long_range_understanding`: 30
      - `conflict_resolution`: 8

## Canonical Artifact Verification

```bash
python -m eval.artifacts verify --preset vibecoder_memoryagentbench_v1
python -m eval.artifacts verify --preset vibecoder_memoryagentbench_stress_v1
python -m eval.artifacts verify --all
```

This command validates checksums, dataset contract invariants, and baseline report reproducibility against current eval code.

## Local Ollama Model Matrix + Benchmark

Use this for local model-cache sync and side-by-side throughput/latency plus ability-vs-resource benchmarking.

```bash
# Show matrix install status
python -m eval.ollama_local_benchmark list

# Pull default benchmark set (xlam, qwen3:8b, deepseek-r1:8b, qwen2.5-coder:7b, llama3.1:8b)
python -m eval.ollama_local_benchmark sync

# Pull all matrix models including optional entries like qwen3:14b
python -m eval.ollama_local_benchmark sync --include-optional

# Benchmark installed default models against the curated prompt pack
python -m eval.ollama_local_benchmark benchmark --repeats 2 --num-predict 192

# Benchmark explicit model subset
python -m eval.ollama_local_benchmark benchmark --models xlam:latest,qwen3:8b

# Build legacy-ingestion benchmark cases from old project roots and score model ability/resource efficiency
python -m eval.ollama_local_benchmark legacy-benchmark \
  --legacy-roots "C:/projects/old_repo_1,C:/projects/old_repo_2" \
  --repeats 1 \
  --dump-cases eval/reports/ollama/legacy_cases.jsonl

# Evaluate profile-promotion gates using live + legacy reports
python -m eval.ollama_local_benchmark profile-gate \
  --live-report eval/reports/ollama/report_<live>.json \
  --legacy-report eval/reports/ollama/legacy_report_<legacy>.json

# Run full development-cycle benchmark flow (live + legacy + profile gate)
python -m eval.ollama_local_benchmark dev-cycle \
  --legacy-roots "C:/path/to/old_project_1,C:/path/to/old_project_2" \
  --repeats 1

# Run dev-cycle and apply profile defaults to a running Muninn server
# (writes checkpoint artifact for rollback before applying)
python -m eval.ollama_local_benchmark dev-cycle \
  --legacy-roots "C:/path/to/old_project_1,C:/path/to/old_project_2" \
  --repeats 1 \
  --apply-policy \
  --muninn-url http://127.0.0.1:42069

# Roll back to previous profile policy using checkpoint
python -m eval.ollama_local_benchmark rollback-policy \
  --checkpoint eval/reports/ollama/profile_policy_checkpoint_<run_id>.json

# Record explicit approval/rejection decision for a checkpoint
python -m eval.ollama_local_benchmark approval-manifest \
  --checkpoint eval/reports/ollama/profile_policy_checkpoint_<run_id>.json \
  --decision approved \
  --approved-by "operator@example" \
  --pr-number 28 \
  --pr-url "https://github.com/<owner>/<repo>/pull/28" \
  --commit-sha "abc1234" \
  --branch-name "feat/phase4o-approval-provenance-context" \
  --notes "Gate evidence + reviewer approval"

# Apply a checkpoint only when approved by manifest
python -m eval.ollama_local_benchmark apply-checkpoint \
  --checkpoint eval/reports/ollama/profile_policy_checkpoint_<run_id>.json \
  --approval-manifest eval/reports/ollama/policy_approval_<run_id>.json \
  --require-change-context \
  --require-pr-number \
  --require-commit-sha \
  --require-branch-name \
  --muninn-url http://127.0.0.1:42069
```

Versioned inputs:
- Model matrix: `eval/ollama_model_matrix.json`
- Prompt pack: `eval/ollama_benchmark_prompts.jsonl`
- Profile promotion policy: `eval/ollama_profile_promotion_policy.json`

Generated reports are written to `eval/reports/ollama/` (gitignored).

`benchmark` report summaries now include:
- `avg_ability_score`
- `ability_pass_rate`
- `resource_efficiency.ability_per_second`
- `resource_efficiency.ability_per_vram_gb`

`legacy-benchmark` generates deterministic ingestion-like extraction cases from local project files and reports the same ability/resource metrics per model.

`profile-gate` consumes benchmark reports and emits per-profile pass/fail + recommendation decisions for `low_latency`, `balanced`, and `high_reasoning` promotion policies.

`dev-cycle` runs `benchmark`, `legacy-benchmark`, and `profile-gate` sequentially in one operator-triggered command and emits a summary that maps recommended models to profile usage roles.

`dev-cycle --apply-policy` additionally:
- validates gate/recommendation evidence for target profile defaults,
- fetches current `/profiles/model` policy from server,
- writes checkpoint artifact with previous policy + apply payload,
- applies new profile defaults (unless `--apply-dry-run` is used).

`rollback-policy` restores profile defaults from a checkpoint artifact and writes a rollback report.

`approval-manifest` writes an explicit approval/rejection artifact tied to checkpoint path + SHA-256 digest + reviewer identity.

`approval-manifest` also supports optional change-context provenance (`pr_number`, `pr_url`, `commit_sha`, `branch_name`), with git auto-detection for commit/branch when not explicitly provided.

`apply-checkpoint` enforces approval-manifest controls before applying:
- manifest decision must be `approved`,
- manifest checkpoint SHA-256 must match the supplied checkpoint file,
- optional manifest checkpoint path must match the supplied checkpoint path,
- optional manifest `change_context` must be a JSON object when present,
- optional enforcement flags can require provenance fields before apply:
  - `--require-change-context`
  - `--require-pr-number`
  - `--require-commit-sha`
  - `--require-branch-name`
- successful apply writes a deterministic apply report artifact.

## Phase Hygiene Gate

Use this utility at each phase boundary (and before merge) to enforce one-open-PR policy and catch test-quality drift.

```bash
# Full gate (PR status + review/check signals + pytest summary budgets)
python -m eval.phase_hygiene \
  --max-open-prs 1 \
  --require-open-pr \
  --pytest-command "python -m pytest -q"

# PR-only check (skip test command)
python -m eval.phase_hygiene \
  --max-open-prs 1 \
  --pytest-command ""
```

The report is written to `eval/reports/hygiene/phase_hygiene_<timestamp>.json` and includes:
- open PR inventory,
- selected PR review/check summary,
- parsed pytest summary (passed/failed/skipped/warnings) using JUnit XML when pytest is detected,
- deterministic pass/fail + violations list.

Implementation note:
- test command execution is tokenized (`shell=False`) to avoid shell-injection surface.
