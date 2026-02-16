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

# Run deferred dev-cycle during active enhancement phases (reuse existing benchmark reports)
python -m eval.ollama_local_benchmark dev-cycle \
  --legacy-roots "C:/path/to/old_project_1,C:/path/to/old_project_2" \
  --defer-benchmarks \
  --existing-live-report eval/reports/ollama/cycle_live_<run_id>.json \
  --existing-legacy-report eval/reports/ollama/cycle_legacy_<run_id>.json \
  --max-reused-report-age-hours 72

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
  --require-commit-reachable-from main \
  --muninn-url http://127.0.0.1:42069

# Unified SOTA+ go/no-go verdict (quality + reliability + stats + reproducibility)
python -m eval.ollama_local_benchmark sota-verdict \
  --candidate-eval-report eval/reports/current_eval.json \
  --baseline-eval-report eval/artifacts/vibecoder_memoryagentbench_v1/baseline_report.json \
  --profile-gate-report eval/reports/ollama/cycle_gate_<run_id>.json \
  --transport-report eval/reports/mcp_transport/mcp_transport_soak_<run_id_1>.json \
  --transport-report eval/reports/mcp_transport/mcp_transport_soak_<run_id_2>.json
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

`dev-cycle` supports two execution modes:
- full mode (default): runs `benchmark`, `legacy-benchmark`, and `profile-gate` sequentially.
- deferred mode (`--defer-benchmarks`): skips live/legacy generation, reuses existing reports, and re-runs `profile-gate` for fresh policy/governance evaluation.

Deferred mode is designed for active enhancement phases where running full benchmark suites on every tranche is low ROI. Use `--max-reused-report-age-hours` to enforce freshness bounds for reused evidence.

`dev-cycle --apply-policy` additionally:
- validates gate/recommendation evidence for target profile defaults,
- fetches current `/profiles/model` policy from server,
- writes checkpoint artifact with previous policy + apply payload,
- applies new profile defaults (unless `--apply-dry-run` is used).

`sota-verdict` normalizes retrieval-eval and transport reports into one deterministic decision artifact:
- quality gate (primary improvement + track non-negative ratio),
- reliability gate (p95 regression + transport timeout/closure incidence),
- statistical validity gate (significance payload checks),
- reproducibility/integrity gate (artifact verification),
- profile-policy gate (profile-gate pass + governance block state).

## MCP Transport Soak

Use this to stress MCP transport behavior under controlled backend outage conditions.

```bash
python -m eval.mcp_transport_soak \
  --iterations 6 \
  --warmup-requests 1 \
  --timeout-sec 12 \
  --transport framed \
  --server-url http://127.0.0.1:1 \
  --failure-threshold 1 \
  --cooldown-sec 30 \
  --max-p95-ms 2500 \
  --task-result-mode auto \
  --task-result-auto-retry-clients "claude desktop,claude code,cursor,windsurf,continue" \
  --probe-nonterminal-task-result \
  --task-worker-start-delay-ms 350 \
  --inject-malformed-frame
```

Reports are written to `eval/reports/mcp_transport/`.

### Transport Blocker Closure Campaign

Use this to run deterministic multi-run closure checks against the transport intermittency criteria.

```bash
python -m eval.mcp_transport_closure \
  --streak-target 30 \
  --max-campaign-runs 60 \
  --transports framed,line \
  --min-p95-compliance-ratio 0.95 \
  --soak-iterations 25 \
  --soak-warmup-requests 2 \
  --soak-timeout-sec 15 \
  --soak-max-p95-ms 5000 \
  --soak-task-result-mode auto \
  --soak-task-result-auto-retry-clients "claude desktop,claude code,cursor,windsurf,continue" \
  --soak-probe-nonterminal-task-result \
  --soak-task-worker-start-delay-ms 350 \
  --soak-server-url http://127.0.0.1:1
```

This emits `eval/reports/mcp_transport/mcp_transport_closure_<run_id>.json` with:
- closure-ready verdict,
- consecutive-pass streak state,
- p95 compliance ratio in observation window,
- explicit criteria flags (including unresolved regression/defect inputs and `nonterminal_task_result_probe_met`),
- telemetry rollups for error-code totals, task-result compatibility mode/profile distributions, and non-terminal probe success ratios.

### Wrapper Response-Compaction Knobs

For oversized tool payloads in host runtimes, wrapper-side response shaping can be tuned via:
- `MUNINN_MCP_TOOL_RESPONSE_PREVIEW_MAX_ITEMS` (default `200`)
- `MUNINN_MCP_TOOL_RESPONSE_PREVIEW_MAX_DEPTH` (default `6`)
- `MUNINN_MCP_TOOL_RESPONSE_PREVIEW_MAX_STRING_CHARS` (default `2000`)
- `MUNINN_MCP_TOOL_RESPONSE_MAX_CHARS` (default `12000`, final emitted text cap)

### Transport Diagnostics Bundle

Use this to generate deterministic incident triage artifacts from wrapper logs plus recent soak/closure reports:

```bash
python -m eval.mcp_transport_diagnostics \
  --lookback-hours 24 \
  --recent-soak-limit 5 \
  --recent-closure-limit 3
```

This emits `eval/reports/mcp_transport/mcp_transport_diagnostics_<run_id>.json` with:
- wrapper incident counters (`transport_closed`, deadline exhaustion),
- per-tool wall-time/response-size summaries,
- near-timeout event extraction,
- recent soak/closure artifact rollups,
- blocker-signal heuristic summary for wrapper-vs-host attribution.

Optional enforcement mode (for CI/release gates):

```bash
python -m eval.mcp_transport_diagnostics \
  --lookback-hours 24 \
  --max-transport-closed-count 0 \
  --max-deadline-exhaustion-count 0 \
  --max-near-timeout-count 0 \
  --enforce-gate
```

### Phase Hygiene + Transport Diagnostics

You can wire transport diagnostics directly into `eval.phase_hygiene`:

```bash
python -m eval.phase_hygiene \
  --require-open-pr \
  --pytest-command "python -m pytest -q tests/test_mcp_transport_diagnostics.py tests/test_phase_hygiene.py" \
  --transport-diagnostics-command "python -m eval.mcp_transport_diagnostics --lookback-hours 24 --max-transport-closed-count 0 --max-deadline-exhaustion-count 0 --max-near-timeout-count 0 --enforce-gate" \
  --fail-on-transport-diagnostics \
  --max-transport-closed-incidents 0 \
  --max-transport-deadline-exhaustion-incidents 0 \
  --max-transport-near-timeout-incidents 0
```

### Transport Incident Replay Automation

Use this to trigger diagnostics capture only when transport incident signatures are detected in the wrapper log window:

```bash
python -m eval.mcp_transport_incident_replay \
  --lookback-hours 24 \
  --signature-pattern "MCP stdio transport closed while sending JSON-RPC message" \
  --require-log-path-exists \
  --diagnostics-command "python -m eval.mcp_transport_diagnostics --lookback-hours 24 --max-transport-closed-count 0 --max-deadline-exhaustion-count 0 --max-near-timeout-count 0 --enforce-gate"
```

This emits `eval/reports/mcp_transport/mcp_transport_incident_replay_<run_id>.json` with:
- detected incident signature counts and sampled matching lines,
- trigger decision (`results.triggered`),
- diagnostics command execution details + exit code,
- resolved diagnostics artifact path when available.

PR/release workflow wiring is available in:
- `.github/workflows/transport-incident-replay-gate.yml`
- Workflow profile options:
  - `pr_safe` (default): non-strict mode, no required host log.
  - `release_host_captured`: strict mode (`--require-log-path-exists` + `--include-log-sha256`) for host-captured release environments.

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
- optional git ancestry enforcement:
  - `--require-commit-reachable-from <ref>` validates manifest commit SHA reachability from the specified git ref/branch before apply.
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
