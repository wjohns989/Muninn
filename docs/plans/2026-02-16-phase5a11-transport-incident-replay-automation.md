# Phase 5A.11: Transport Incident Replay Automation

Date: 2026-02-16  
Status: Implemented

## Objective

Automate diagnostics capture when host/runtime transport-closure signatures appear, so PR/release checks can attach deterministic incident evidence without manual triage steps.

## Implemented

1. Added deterministic replay utility: `python -m eval.mcp_transport_incident_replay`.
2. Replay utility scans wrapper logs in a bounded lookback window for configured regex signatures (default transport-closure signature).
3. Replay utility triggers diagnostics command execution only when:
   - signature count reaches configured threshold, or
   - `--always-run-diagnostics` is explicitly enabled.
4. Replay report now includes:
   - signature totals + per-pattern counts,
   - sampled matching lines with timestamps,
   - trigger decision,
   - diagnostics command return code and parsed payload,
   - resolved diagnostics artifact path when available.
5. Replay utility supports policy-style failure behavior (`--fail-on-diagnostics-error`) so CI/release checks can fail deterministically when diagnostics execution fails.

## Verification

1. Compile checks:
   - `python -m py_compile eval/mcp_transport_incident_replay.py tests/test_mcp_transport_incident_replay.py`
2. Targeted replay test suite:
   - `python -m pytest -q tests/test_mcp_transport_incident_replay.py`
   - Result: `3 passed`.
3. Live replay run:
   - `python -m eval.mcp_transport_incident_replay --lookback-hours 24 --signature-pattern "MCP stdio transport closed while sending JSON-RPC message" --diagnostics-command "python -m eval.mcp_transport_diagnostics --lookback-hours 24 --max-transport-closed-count 0 --max-deadline-exhaustion-count 0 --max-near-timeout-count 0 --enforce-gate"`
   - Artifact: `eval/reports/mcp_transport/mcp_transport_incident_replay_20260216_010414.json`
   - Result: `results.triggered=false` (no matching signatures in lookback window).

## ROI / Blocker Impact

1. Reduces MTTR by auto-linking incident signatures to diagnostics artifacts rather than relying on manual capture.
2. Tightens blocker governance by making host-runtime incident evidence deterministic and machine-readable.
3. Closes the automation gap between transport signature detection and diagnostics artifact production for PR/release workflows.
