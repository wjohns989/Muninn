# Phase 4K Plan: Hygiene Gate + Roadmap Refresh

Date: 2026-02-15
Owner: Codex
Status: Implemented in branch `feat/phase4k-roadmap-closure`

## Objective

Codify phase-boundary discipline so PR hygiene and test-quality signals are checked deterministically, then refresh the roadmap with the remaining highest-ROI work in execution order.

## Implemented

1. Added phase hygiene utility:
   - `eval/phase_hygiene.py`
2. Utility capabilities:
   - enforces one-open-PR policy (`--max-open-prs`, default `1`),
   - inspects a target PR (explicit `--pr-number` or auto-selected when exactly one open PR exists),
   - surfaces review/check health (`reviewDecision`, status-check rollup summary),
   - runs configurable test command with shell-safe tokenized execution and parses deterministic JUnit counts (`passed`, `failed`, `skipped`) plus warning budgets,
   - applies deterministic thresholds for skipped tests and warnings.
3. Added focused tests:
   - `tests/test_phase_hygiene.py`
4. Updated eval docs:
   - `eval/README.md` includes command usage and report contract.

## Validation

```bash
python -m py_compile eval/phase_hygiene.py tests/test_phase_hygiene.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_phase_hygiene.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_ollama_local_benchmark.py tests/test_phase_hygiene.py
```

Observed in-session:
- `5 passed` (`tests/test_phase_hygiene.py`)
- `13 passed` (`tests/test_ollama_local_benchmark.py`, `tests/test_phase_hygiene.py`)

## Brainstormed Missing Features (ROI-ordered)

1. On-demand benchmark orchestration:
   - run `benchmark`, `legacy-benchmark`, and `profile-gate` as explicit operator-controlled workflows,
   - archive reports/artifacts for trend analysis and promotion auditability.
2. Controlled profile-promotion automation:
   - apply gate recommendations to runtime profile policy only when guardrails pass,
   - auto-create rollback checkpoint + audit event link.
3. Browser operator center expansion:
   - add policy panel for runtime/ingestion/legacy profile knobs,
   - add benchmark launch/summary views,
   - add preference presets (Battery Saver, Balanced Dev, Deep Planning).
4. Ingestion hardening (security):
   - sandbox optional binary parsers (`pdf/docx`) out-of-process,
   - bound parser runtime and memory to reduce blast radius.
5. Benchmark corpus breadth expansion:
   - add adversarial/noise and cross-domain datasets beyond current two canonical bundles,
   - increase statistical power for profile-promotion decisions.
6. Observability/alerts:
   - alert on profile-policy churn, benchmark regressions, and repeated gate failures.

## Model-Matrix Reality Check (2026-02-15)

To keep 16GB workflows practical while preserving SOTA+ optionality:

1. Keep helper/runtime defaults in small-to-mid models:
   - `llama3.2:3b`/`qwen3:8b` class for active coding sessions.
2. Keep high-reasoning profiles optional and session-scoped:
   - `qwen3:14b` viable on 16GB-class systems with tighter concurrency,
   - larger 30B+ tiers should remain explicit opt-in.
3. Keep xLAM as an optional structured-extraction candidate:
   - good function-calling/tool orientation, but should not force global default.

Reference checkpoints used for policy sanity:
- Ollama library model pages for `qwen3`, `deepseek-r1`, `llama3.2`
- Ollama library community model page for `xlam`
- xLAM and Qwen3 technical reports

## Full Remaining Roadmap (post-Phase 4L)

Phase 4L is now implemented in `docs/plans/2026-02-15-phase4l-dev-cycle-benchmark-orchestration.md`.

### Phase 4M - Safe Auto-Promotion + Rollback
1. Apply `profile-gate` recommendation to active profile policy via API/MCP path.
2. Record mutation with audit-event correlation to benchmark report IDs.
3. Add rollback command to restore prior policy snapshot.

Exit criteria:
- Promotion can run unattended with deterministic rollback path.
- Every promotion has provenance to benchmark + audit event.

### Phase 4N - Browser UI Operator Controls
1. Add profile-policy controls (runtime/ingestion/legacy) in `dashboard.html`.
2. Add benchmark result viewer and gate status panel.
3. Add user-facing presets and one-click apply.

Exit criteria:
- Non-CLI operator can inspect and apply profile policy safely.
- UI changes emit audited policy events.

### Phase 5A - Parser Sandbox Hardening
1. Move optional binary parsers to isolated subprocess/worker profile.
2. Add execution timeout and memory budget controls.
3. Preserve fail-open semantics with explicit per-file error reporting.

Exit criteria:
- Parser crashes/timeouts cannot destabilize main process.
- Security posture documented and verified by tests.

### Phase 5B - Corpus + Statistical Expansion
1. Add additional domain bundles and adversarial/noise slices.
2. Extend significance/correction reporting across expanded suites.
3. Refresh canonical manifests and reproducibility checks.

Exit criteria:
- Promotion decisions are supported by broader, representative evidence.
- Artifact verification remains deterministic.

### Phase 6 - Observability and Ops Alerting
1. Add dashboards and alert rules for:
   - profile-policy mutation anomaly,
   - benchmark quality/latency regressions,
   - repeated hygiene-gate failures.
2. Add runbook updates for incident response.

Exit criteria:
- Ops can detect and respond to quality drift without manual log inspection.
