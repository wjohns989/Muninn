# Phase 5B.4: Release-Boundary Blocker Decision Gate Wiring

Date: 2026-02-16  
Status: Implemented

## Objective

Enforce transport blocker-closure criteria automatically at release boundaries so closure readiness does not depend on manual command execution.

## Implemented

1. Updated workflow: `.github/workflows/transport-incident-replay-gate.yml`.
2. Added release-aware profile defaulting:
   - when `gate_profile` input is not provided, `release` events now default to `release_host_captured` (PR/push remain `pr_safe`).
3. Added release-profile blocker decision step:
   - runs only when profile resolves to `release_host_captured`,
   - executes:
     - `python -m eval.mcp_transport_blocker_decision --enforce-gate --replay-provenance-policy latest_min`
   - emits CI decision artifact:
     - `eval/reports/mcp_transport/mcp_transport_blocker_decision_ci_<run_id>_<attempt>.json`
4. Extended artifact upload and step summary:
   - decision artifact is uploaded with replay/diagnostics artifacts,
   - summary now includes gate profile + blocker decision verdict/violations for release-profile runs.

## Verification

1. Workflow file updated without changing PR-safe execution path semantics.
2. In-session blocker decision command already validated with same policy/enforcement flags:
   - `eval/reports/mcp_transport/mcp_transport_blocker_decision_20260216_015409.json`
   - `blocker_closure_ready=true`, `violations=[]`.

## ROI / Blocker Impact

1. Converts release-time blocker closure from manual procedure to deterministic CI policy enforcement.
2. Keeps replay diagnostics and closure-decision evidence co-located in one workflow artifact set.
3. Reduces risk of releasing while closure criteria silently regress.
