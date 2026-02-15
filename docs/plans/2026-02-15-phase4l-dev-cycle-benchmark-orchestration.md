# Phase 4L Plan: Development-Cycle Benchmark Orchestration

Date: 2026-02-15
Owner: Codex
Status: Implemented baseline in branch `feat/phase4k-roadmap-closure`

## Objective

Make model benchmarking an explicit part of the development cycle (not nightly automation) so profile/model choices are evidence-based for each workload role.

## Implemented

1. Added `dev-cycle` command to benchmark utility:
   - `python -m eval.ollama_local_benchmark dev-cycle ...`
2. Command orchestration:
   - runs `benchmark` (live prompts),
   - runs `legacy-benchmark` (old-project ingestion-like prompts),
   - runs `profile-gate` (policy-driven promotion decisions).
3. Added role-oriented recommendation summary:
   - emits `dev_cycle_summary_<timestamp>.json`,
   - maps recommended model per profile and usage role:
     - `low_latency`: runtime helper during active coding/tool-heavy IDE use.
     - `balanced`: default implementation assistant.
     - `high_reasoning`: planning/architecture/deep analysis sessions.
4. Added focused test coverage:
   - `test_cmd_dev_cycle_writes_role_recommendations` in `tests/test_ollama_local_benchmark.py`.

## Why this improves ROI

1. Converts model selection from preference-driven to evidence-driven decisions.
2. Preserves your no-nightly requirement while still enforcing disciplined benchmarking per development phase.
3. Produces durable artifacts that explain not only which model won, but what role it should serve.

## Command Example

```bash
python -m eval.ollama_local_benchmark dev-cycle \
  --legacy-roots "C:/path/to/old_project_1,C:/path/to/old_project_2" \
  --repeats 1 \
  --models "xlam:latest,qwen3:8b,qwen3:14b"
```

## Next Follow-ups

1. Add browser UI trigger for `dev-cycle` plus recommendation visualization.
2. Add controlled profile-policy apply/rollback action from dev-cycle summary output.
3. Add policy note section that records why each profile recommendation was accepted/rejected in the active branch.
