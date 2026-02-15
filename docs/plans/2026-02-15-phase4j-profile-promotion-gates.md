# Phase 4J Plan: Profile Promotion Gates

Date: 2026-02-15
Owner: Codex
Status: Implemented in branch `feat/phase4j-profile-promotion-gates`

## Objective

Convert benchmark outputs into deterministic promotion decisions for model profiles (`low_latency`, `balanced`, `high_reasoning`) so defaults are evidence-driven instead of manual/ad-hoc.

## Implemented

1. Added promotion policy file:
   - `eval/ollama_profile_promotion_policy.json`
2. Added gate evaluator command:
   - `python -m eval.ollama_local_benchmark profile-gate ...`
3. Gate evaluator behavior:
   - consumes live benchmark report (`benchmark`),
   - optionally consumes legacy benchmark report (`legacy-benchmark`),
   - evaluates per-profile candidate models against threshold policy,
   - emits pass/fail + recommended model with composite score.
4. Added test coverage:
   - `tests/test_ollama_local_benchmark.py`
5. Updated docs:
   - `eval/README.md`
   - roadmap/plan docs (`docs/PLAN_GAP_EVALUATION.md`, `docs/MUNINN_COMPREHENSIVE_ROADMAP.md`, `SOTA_PLUS_PLAN.md`).

## Command examples

```bash
python -m eval.ollama_local_benchmark profile-gate \
  --live-report eval/reports/ollama/report_<timestamp>.json \
  --legacy-report eval/reports/ollama/legacy_report_<timestamp>.json
```

```bash
python -m eval.ollama_local_benchmark profile-gate \
  --live-report eval/reports/ollama/report_<timestamp>.json \
  --allow-failures
```

## Why this matters (ROI)

1. Prevents silent quality regressions when switching default models.
2. Makes 16GB helper-first tradeoffs explicit and reviewable.
3. Provides machine-readable gate artifacts suitable for CI/nightly promotion pipelines.

## Open follow-ups

1. Add CI/nightly automation that runs `benchmark`, `legacy-benchmark`, and `profile-gate`.
2. Bind successful gate outputs to controlled profile-policy updates (with audit-event trail).
3. Add browser UI panel for gate report visualization and one-click recommended profile staging.

## Workflow rule update

For PR hygiene, preserve reviewer soak window:
1. Create PR with validation evidence.
2. Do not merge in the same execution response.
3. On subsequent interaction, re-check comments/reviews/checks before merge.
