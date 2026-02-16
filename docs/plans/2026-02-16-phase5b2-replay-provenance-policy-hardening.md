# Phase 5B.2: Replay Provenance Policy Hardening

Date: 2026-02-16  
Status: Implemented

## Objective

Eliminate false-negative blocker decisions caused by legacy non-strict replay artifacts while preserving strict provenance requirements for current closure evidence.

## Implemented

1. Extended `python -m eval.mcp_transport_blocker_decision` with:
   - `--replay-provenance-policy all|latest_min` (default: `all`).
2. Added deterministic provenance-scope behavior:
   - `all`: provenance must pass for all replay reports in lookback window.
   - `latest_min`: provenance must pass for latest required replay evidence set (`max(min_replay_runs, 1)`).
3. Added explicit replay-provenance audit payload in decision reports:
   - `policy`,
   - `required_count`,
   - `evaluated_count`,
   - `passing_count`,
   - `selected_paths`.
4. Added test coverage for mixed historical evidence:
   - legacy missing-provenance replay plus latest strict-provenance replays now passes in `latest_min` mode when required latest evidence is satisfied.

## Verification

1. Compile checks:
   - `python -m py_compile eval/mcp_transport_blocker_decision.py tests/test_mcp_transport_blocker_decision.py`
2. Decision utility tests:
   - `python -m pytest -q tests/test_mcp_transport_blocker_decision.py`
   - Result: `4 passed`.
3. Expanded targeted suite:
   - `python -m py_compile eval/mcp_transport_diagnostics.py eval/phase_hygiene.py eval/mcp_transport_incident_replay.py eval/mcp_transport_blocker_decision.py tests/test_mcp_transport_diagnostics.py tests/test_phase_hygiene.py tests/test_mcp_transport_incident_replay.py tests/test_mcp_transport_blocker_decision.py`
   - `python -m pytest -q tests/test_mcp_transport_diagnostics.py tests/test_phase_hygiene.py tests/test_mcp_transport_incident_replay.py tests/test_mcp_transport_blocker_decision.py tests/test_mcp_wrapper_protocol.py tests/test_mcp_transport_soak.py tests/test_mcp_transport_closure.py`
   - Result: `128 passed`.
4. Live decision artifact:
   - `python -m eval.mcp_transport_blocker_decision --lookback-hours 48 --min-replay-runs 3 --max-replay-signature-count 0 --require-replay-provenance --replay-provenance-policy latest_min --min-closure-runs 1 --require-latest-closure-ready --require-latest-probe-criterion`
   - Artifact: `eval/reports/mcp_transport/mcp_transport_blocker_decision_20260216_014909.json`
   - Result: blocker remains open, now with explicit evidence shortage:
     - `replay_run_count_meets_min=false`,
     - `replay_provenance_met=false` (`required_count=3`, `evaluated_count=2`, `passing_count=1`).

## ROI / Blocker Impact

1. Prevents historical non-strict artifacts from obscuring current strict-evidence readiness in closure decisions.
2. Makes blocker-closure gaps operationally actionable via explicit required/evaluated/passing evidence counts.
3. Preserves deterministic strictness: blocker still cannot close until required strict replay evidence count is actually met.

## Follow-on

This policy hardening was validated in Phase 5B.3 (`docs/plans/2026-02-16-phase5b3-strict-replay-evidence-closure-readiness.md`), where strict replay evidence count/provenance criteria were satisfied and the enforced blocker decision gate passed.
