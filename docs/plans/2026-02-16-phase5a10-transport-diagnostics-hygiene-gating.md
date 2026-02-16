# Phase 5A.10: Transport Diagnostics Gate + Phase-Hygiene Integration

Date: 2026-02-16  
Status: Implemented

## Objective

Attach deterministic transport diagnostics evidence directly to PR/release checks and fail policy gates when transport incident thresholds are exceeded.

## Implemented

1. `eval.mcp_transport_diagnostics` now supports gate thresholds:
   - `--max-transport-closed-count`
   - `--max-deadline-exhaustion-count`
   - `--max-near-timeout-count`
   - `--enforce-gate` (non-zero exit on violations)
2. Diagnostics output now includes explicit gate verdict:
   - `results.gate.passed`
   - `results.gate.violations`
3. `eval.phase_hygiene` now supports transport diagnostics policy wiring:
   - `--transport-diagnostics-command`
   - `--fail-on-transport-diagnostics`
   - `--max-transport-closed-incidents`
   - `--max-transport-deadline-exhaustion-incidents`
   - `--max-transport-near-timeout-incidents`
4. Hygiene report now captures diagnostics command, return code, and parsed diagnostics summary for auditable gating.

## Verification

1. Compile checks:
   - `python -m py_compile eval/mcp_transport_diagnostics.py eval/phase_hygiene.py tests/test_mcp_transport_diagnostics.py tests/test_phase_hygiene.py`
2. Targeted test suite:
   - `python -m pytest -q tests/test_mcp_transport_diagnostics.py tests/test_phase_hygiene.py tests/test_mcp_wrapper_protocol.py tests/test_mcp_transport_soak.py tests/test_mcp_transport_closure.py`
   - Result: `119 passed`.
3. Live diagnostics gate run:
   - `python -m eval.mcp_transport_diagnostics --lookback-hours 24 --recent-soak-limit 3 --recent-closure-limit 2 --enforce-gate --max-transport-closed-count 0 --max-deadline-exhaustion-count 0 --max-near-timeout-count 0`
   - Artifact: `eval/reports/mcp_transport/mcp_transport_diagnostics_20260216_005047.json`
   - Result: `gate.passed=true`.

## ROI / Blocker Impact

1. Converts transport diagnostics from observational-only into enforceable gate policy.
2. Enables attaching deterministic transport health evidence to PR/release checks.
3. Reduces risk of promoting phases while transport incident signatures are regressing.
