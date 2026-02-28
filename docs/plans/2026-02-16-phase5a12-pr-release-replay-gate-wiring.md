# Phase 5A.12: PR/Release Replay Gate Wiring

Date: 2026-02-16  
Status: Implemented

## Objective

Wire transport incident replay automation into PR/release checks so diagnostics artifacts are emitted and attached automatically in check outputs.

## Implemented

1. Added GitHub Actions workflow:
   - `.github/workflows/transport-incident-replay-gate.yml`
2. Workflow triggers:
   - `pull_request` on `main`,
   - `push` on `main`,
   - `workflow_dispatch` with runtime inputs (`log_path`, `lookback_hours`, `require_log_path_exists`, `always_run_diagnostics`).
3. Workflow execution:
   - runs `python -m eval.mcp_transport_incident_replay` with deterministic diagnostics command wiring,
   - emits CI-scoped replay artifact path (`mcp_transport_incident_replay_ci_<run_id>_<attempt>.json`),
   - uploads replay + diagnostics artifacts,
   - writes a compact run summary to `GITHUB_STEP_SUMMARY`.
4. Replay utility hardening:
   - added `--require-log-path-exists` guardrail for strict host-runtime environments,
   - returns deterministic non-zero (`4`) when strict mode is enabled and log file is absent.

## Verification

1. Compile checks:
   - `python -m py_compile eval/mcp_transport_incident_replay.py tests/test_mcp_transport_incident_replay.py`
2. Replay test suite:
   - `python -m pytest -q tests/test_mcp_transport_incident_replay.py`
   - Result: `4 passed`.
3. Expanded transport targeted suite:
   - `python -m py_compile eval/mcp_transport_diagnostics.py eval/phase_hygiene.py eval/mcp_transport_incident_replay.py tests/test_mcp_transport_diagnostics.py tests/test_phase_hygiene.py tests/test_mcp_transport_incident_replay.py`
   - `python -m pytest -q tests/test_mcp_transport_diagnostics.py tests/test_phase_hygiene.py tests/test_mcp_transport_incident_replay.py tests/test_mcp_wrapper_protocol.py tests/test_mcp_transport_soak.py tests/test_mcp_transport_closure.py`
   - Result: `123 passed`.

## ROI / Blocker Impact

1. Closes the planned wiring gap between replay utility and PR/release checks.
2. Produces deterministic transport evidence artifacts in check runs without manual intervention.
3. Adds strict-mode misconfiguration detection for host-runtime pipelines where log capture is mandatory.
