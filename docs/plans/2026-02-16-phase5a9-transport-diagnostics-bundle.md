# Phase 5A.9: Transport Diagnostics Bundle Utility

Date: 2026-02-16  
Status: Implemented

## Objective

Reduce MTTR for intermittent host-side transport issues by generating a deterministic diagnostic bundle from wrapper logs plus recent soak/closure artifacts.

## Implemented

1. Added new utility: `python -m eval.mcp_transport_diagnostics`
2. Bundle includes:
   - log-window incident counts (`transport_closed`, deadline exhaustion, startup-budget skips),
   - per-tool wall-time and response-size telemetry summaries (`p50/p95/max`, outcome counts),
   - near-timeout event extraction,
   - recent soak and closure artifact summaries,
   - blocker-signal heuristics for quick triage.
3. Added tests:
   - telemetry line parser behavior,
   - end-to-end diagnostics bundle generation from fixture log/report inputs.

## Verification

1. Compile checks:
   - `python -m py_compile eval/mcp_transport_diagnostics.py tests/test_mcp_transport_diagnostics.py`
2. Targeted tests:
   - `python -m pytest -q tests/test_mcp_transport_diagnostics.py tests/test_mcp_wrapper_protocol.py tests/test_mcp_transport_soak.py tests/test_mcp_transport_closure.py`
   - Result: `110 passed`.
3. Live diagnostics artifact generated:
   - `python -m eval.mcp_transport_diagnostics --lookback-hours 24 --recent-soak-limit 3 --recent-closure-limit 2`
   - Artifact: `eval/reports/mcp_transport/mcp_transport_diagnostics_20260216_001515.json`
   - Current blocker-signal result: no wrapper-level failure signals in this window.

## ROI / Blocker Impact

1. Converts ad-hoc log inspection into deterministic machine-readable evidence.
2. Improves attribution of wrapper vs host-runtime issues.
3. Enables future CI/release gating on transport incident trends.
