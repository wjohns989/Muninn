# Phase 4AF: Unified SOTA+ Verdict Command + Benchmark Normalization Hooks

Date: 2026-02-15  
Branch: `feat/phase4v-task-metadata-cursor-compliance`

## Problem

The roadmap defined a quantitative SOTA+ release gate, but there was no single command that could aggregate evidence from retrieval eval, transport soak, profile-gate output, and artifact-integrity checks into one deterministic go/no-go artifact.

## Objectives

1. Add one CLI command to produce a machine-auditable SOTA+ verdict.
2. Normalize heterogeneous benchmark/report payloads into stable schema blocks suitable for downstream automation.
3. Keep gate criteria configurable while defaulting to strict evidence requirements.

## Implemented

### `eval/ollama_local_benchmark.py`

- Added `sota-verdict` subcommand.
- Implemented normalization helpers for:
  - retrieval eval reports (`_normalize_eval_report_for_sota`),
  - auxiliary benchmark reports (`_normalize_aux_benchmark_report`),
  - transport soak reports (`_normalize_transport_report` + aggregate),
  - profile-gate reports (`_normalize_profile_gate_report`).
- Implemented deterministic gate evaluation:
  - quality (primary improvement + per-track non-negative ratio),
  - reliability (p95 regression + timeout/transport-closed incidence + transport p95),
  - statistical validity (significance payload checks with positive significant-delta requirement),
  - reproducibility/integrity (artifact verification integration),
  - profile-policy pass/governance block checks.
- Output artifact now includes:
  - full normalized inputs,
  - per-gate pass/fail + violations,
  - threshold/config echo,
  - final `sota_plus_verdict` status.

### `tests/test_ollama_local_benchmark.py`

- Added passing-path test:
  - `test_cmd_sota_verdict_passes_with_complete_evidence`
- Added failing-path test:
  - `test_cmd_sota_verdict_fails_when_required_evidence_missing`

### `eval/README.md`

- Added `sota-verdict` usage example and gate semantics summary.

## Verification

- `python -m py_compile eval/ollama_local_benchmark.py tests/test_ollama_local_benchmark.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_ollama_local_benchmark.py`
  - `32 passed`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_phase_hygiene.py tests/test_ollama_local_benchmark.py`
  - `39 passed`

## ROI

- Converts SOTA+ release claims from narrative assessment to deterministic gate artifacts.
- Creates a reusable normalization contract for future benchmark integrations (LongMemEval/StructMemEval/Mem2Act adapters).
- Reduces promotion ambiguity by binding one final verdict to explicit thresholds and evidence inputs.

## Next Follow-up

1. Add external benchmark adapters producing normalized `task/track/metric` records directly from LongMemEval/StructMemEval/Mem2Act outputs.
2. Wire `sota-verdict` into scheduled CI replay with drift alerts.
3. Add signed promotion manifest generation tied to verdict artifact SHA + commit SHA.
