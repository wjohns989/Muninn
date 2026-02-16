# Phase 5A.13: Release-Profile Replay Strictness + Log Provenance

Date: 2026-02-16  
Status: Implemented

## Objective

Harden transport replay checks for release boundaries by adding strict host-log requirements and explicit provenance evidence in check summaries.

## Implemented

1. Workflow profile controls added in replay gate workflow:
   - `gate_profile` (`pr_safe` | `release_host_captured`)
   - release profile auto-enables strict replay posture.
2. Replay gate workflow now supports release event execution:
   - `release` trigger (`published`) added in addition to PR/push/manual dispatch.
3. Strict release profile behavior:
   - forces `--require-log-path-exists` and `--include-log-sha256` for host-captured contexts,
   - preserves non-strict default behavior for PR-safe environments.
4. Replay utility provenance enrichment:
   - report now includes log metadata (`size_bytes`, `modified_at`) and optional SHA-256 digest.
5. Check summary enrichment:
   - summary now includes log existence, size, modification timestamp, and digest (when present).

## Verification

1. Compile checks:
   - `python -m py_compile eval/mcp_transport_incident_replay.py tests/test_mcp_transport_incident_replay.py`
2. Replay suite:
   - `python -m pytest -q tests/test_mcp_transport_incident_replay.py`
   - Result: `5 passed`.
3. Expanded transport targeted suite:
   - `python -m py_compile eval/mcp_transport_diagnostics.py eval/phase_hygiene.py eval/mcp_transport_incident_replay.py tests/test_mcp_transport_diagnostics.py tests/test_phase_hygiene.py tests/test_mcp_transport_incident_replay.py`
   - `python -m pytest -q tests/test_mcp_transport_diagnostics.py tests/test_phase_hygiene.py tests/test_mcp_transport_incident_replay.py tests/test_mcp_wrapper_protocol.py tests/test_mcp_transport_soak.py tests/test_mcp_transport_closure.py`
   - Result: `124 passed`.
4. Live strict-profile replay run:
   - `python -m eval.mcp_transport_incident_replay --lookback-hours 24 --signature-pattern "MCP stdio transport closed while sending JSON-RPC message" --require-log-path-exists --include-log-sha256 --diagnostics-command "python -m eval.mcp_transport_diagnostics --lookback-hours 24 --max-transport-closed-count 0 --max-deadline-exhaustion-count 0 --max-near-timeout-count 0 --enforce-gate"`
   - Artifact: `eval/reports/mcp_transport/mcp_transport_incident_replay_20260216_012731.json`
   - Result: `results.triggered=false`, provenance fields populated (`size_bytes`, `modified_at`, `sha256`).

## ROI / Blocker Impact

1. Strengthens release-boundary evidence integrity by requiring log presence and digest provenance in strict mode.
2. Reduces false confidence from missing or ambiguous host logs in release checks.
3. Moves blocker closure criteria closer by making release check outputs sufficient for transport evidence audit.
