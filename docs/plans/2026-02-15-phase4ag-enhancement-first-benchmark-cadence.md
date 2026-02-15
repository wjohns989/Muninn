# Phase 4AG: Enhancement-First Benchmark Cadence

Date: 2026-02-15  
Branch: `feat/phase4v-task-metadata-cursor-compliance`

## Decision

Yes, it is better to continue remaining improvement/enhancement phases first and defer full benchmark matrix replay to release-readiness boundaries, with one constraint:

- development tranches must still run fast deterministic quality gates, and
- deferred benchmark evidence must be freshness-bounded before reuse.

This avoids repeatedly spending expensive benchmark runtime on intermediate states that are expected to change before release.

## Why This Is Higher ROI

1. Full benchmark suites are high-cost and produce short-lived evidence during active implementation churn.
2. Fast tranche gates (unit/protocol/hygiene + transport soak spot checks) catch regressions earlier per unit time.
3. Final SOTA+ claims remain defensible because full replay is still mandatory before promotion and is now bound to a deterministic verdict artifact.

## Implemented

### `eval/ollama_local_benchmark.py`

- Added deferred dev-cycle mode:
  - `--defer-benchmarks`
  - `--existing-live-report`
  - `--existing-legacy-report`
  - `--max-reused-report-age-hours`
- `dev-cycle` now supports:
  - `full` mode (default): run live + legacy generation + profile gate,
  - `deferred` mode: skip generation, reuse existing reports, re-run profile gate for fresh governance/policy decisions.
- Added report freshness validation for deferred mode to block stale evidence.
- Summary artifact now records:
  - execution mode,
  - deferred flag,
  - reused report paths.

### `tests/test_ollama_local_benchmark.py`

- Added deferred-mode reuse test to ensure benchmark generators are not invoked.
- Added stale-report rejection test using bounded report age.

### `eval/README.md`

- Added deferred dev-cycle usage example and guidance.
- Documented two-mode cadence model (full vs deferred).

## Verification

- `python -m py_compile eval/ollama_local_benchmark.py tests/test_ollama_local_benchmark.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_phase_hygiene.py tests/test_ollama_local_benchmark.py`

## Research Signals Used

1. Pytest marker-based selective execution supports splitting fast vs expensive test classes:
   - https://docs.pytest.org/en/stable/example/markers.html
2. GitHub Actions supports scheduled workflows for periodic/heavier runs:
   - https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions#onschedule
3. MCP transport spec complexity (stdio + Streamable HTTP) reinforces keeping transport checks explicit and staged:
   - https://modelcontextprotocol.io/specification/draft/basic/transports

## Follow-up

1. Add CI workflow split:
   - fast tranche checks on PR/push,
   - full benchmark matrix + `sota-verdict` on schedule and release-candidate trigger.
2. Add signed promotion manifest emission from final SOTA+ verdict artifact.
3. Keep transport intermittency blocker open until rolling soak closure criteria are met.
