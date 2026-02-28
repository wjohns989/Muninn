# Phase 4Y Plan: Profile-Governance Alerts + Apply Guardrails

Date: 2026-02-15  
Owner: Codex  
Status: Implemented in branch `feat/phase4v-task-metadata-cursor-compliance`

## Objective

Close the remaining profile-policy governance gap by adding deterministic alert telemetry and policy-apply guardrails:
1. expose recommendation-confidence alerts in `profile-gate` outputs,
2. allow gate/cycle commands to fail when governance policy marks alerts as blocking, and
3. block policy apply when governance state is not clean.

## Implemented

1. Governance alert policy support in `eval/ollama_local_benchmark.py`:
   - new policy surface under `governance.alerts`:
     - `min_composite_score`
     - `min_score_margin`
     - `blocking_severities`
   - deterministic alert generation per profile:
     - `no_passing_candidate` (`critical`)
     - `low_composite_score` (`warning`)
     - `narrow_recommendation_margin` (`warning`)

2. Profile-gate enforcement controls:
   - `profile-gate` report now includes top-level `governance` block:
     - policy snapshot,
     - alert list,
     - severity counts,
     - blocking alert count,
     - blocked boolean.
   - new CLI flag:
     - `profile-gate --enforce-governance`
     - returns non-zero when governance reports blocking alerts.

3. Dev-cycle governance controls:
   - `dev-cycle` now forwards governance enforcement to profile-gate:
     - `--enforce-governance`
   - policy-apply path now supports:
     - `--require-governance-clean`
     - refuses apply when gate report includes blocking governance alerts.
   - dev-cycle summary now includes governance snapshot from gate report.

4. Policy baseline update:
   - `eval/ollama_profile_promotion_policy.json` now includes governance alert thresholds with blocking severity defaults.

5. Test coverage:
   - added governance enforcement coverage in `tests/test_ollama_local_benchmark.py`.
   - updated existing command-namespace fixtures for new flags.

## Validation

1. `python -m py_compile eval/ollama_local_benchmark.py tests/test_ollama_local_benchmark.py`
2. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_ollama_local_benchmark.py`
3. Result: `30 passed`

## ROI / Impact

1. Policy governance is now machine-enforced instead of report-only; low-confidence recommendations can be blocked in CI/operator loops.
2. Apply path safety is improved by preventing policy mutation when governance alert state is explicitly blocking.
3. Benchmark evidence now carries alert telemetry useful for trend monitoring and incident review.

## Remaining Open Work

1. Fully unattended auto-promotion scheduling/roll-forward policy remains out of scope for this tranche.
2. Benchmark corpus breadth expansion (noise/adversarial/domain slices) is still required to increase governance signal confidence.
