# Phase 5B.1: Transport Blocker Decision Utility

Date: 2026-02-16  
Status: Implemented

## Objective

Replace manual interpretation of replay/closure artifacts with a deterministic blocker closure decision utility that can be run in CI/release workflows.

## Implemented

1. Added utility: `python -m eval.mcp_transport_blocker_decision`.
2. Utility now evaluates replay + closure artifacts over configurable lookback windows with criteria for:
   - replay run-count minimum,
   - replay signature budget,
   - replay diagnostics return-code health,
   - replay provenance requirements,
   - closure artifact count,
   - latest closure-ready criterion,
   - latest non-terminal probe criterion.
3. Utility emits deterministic JSON report:
   - `eval/reports/mcp_transport/mcp_transport_blocker_decision_<run_id>.json`
   - includes criteria booleans, violations, and analyzed artifact summaries.
4. Gate-enforcement mode added:
   - `--enforce-gate` returns non-zero when closure criteria fail.

## Verification

1. Compile checks:
   - `python -m py_compile eval/mcp_transport_blocker_decision.py tests/test_mcp_transport_blocker_decision.py`
2. Unit tests:
   - `python -m pytest -q tests/test_mcp_transport_blocker_decision.py`
   - Result: `3 passed`.
3. Expanded targeted suite:
   - `python -m py_compile eval/mcp_transport_diagnostics.py eval/phase_hygiene.py eval/mcp_transport_incident_replay.py eval/mcp_transport_blocker_decision.py tests/test_mcp_transport_diagnostics.py tests/test_phase_hygiene.py tests/test_mcp_transport_incident_replay.py tests/test_mcp_transport_blocker_decision.py`
   - `python -m pytest -q tests/test_mcp_transport_diagnostics.py tests/test_phase_hygiene.py tests/test_mcp_transport_incident_replay.py tests/test_mcp_transport_blocker_decision.py tests/test_mcp_wrapper_protocol.py tests/test_mcp_transport_soak.py tests/test_mcp_transport_closure.py`
   - Result: `127 passed`.
4. Live decision run:
   - `python -m eval.mcp_transport_blocker_decision --lookback-hours 48 --min-replay-runs 3 --max-replay-signature-count 0 --require-replay-provenance --min-closure-runs 1 --require-latest-closure-ready --require-latest-probe-criterion`
   - Artifact: `eval/reports/mcp_transport/mcp_transport_blocker_decision_20260216_013548.json`
   - Result: `blocker_closure_ready=false` with explicit violations:
     - `replay_run_count_meets_min`
     - `replay_provenance_met`

## ROI / Blocker Impact

1. Converts blocker closure calls from narrative judgment to machine-checkable criteria.
2. Surfaces exactly what evidence is still missing to close blocker (count/provenance gaps).
3. Enables future CI/release enforcement with deterministic fail conditions.
