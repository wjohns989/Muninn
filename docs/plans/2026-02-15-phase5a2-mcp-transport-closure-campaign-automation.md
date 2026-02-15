# Phase 5A.2: MCP Transport Blocker-Closure Campaign Automation

Date: 2026-02-15  
Status: Implemented (automation baseline complete + 30-run closure evidence captured)

## Objective

Convert transport intermittency closure criteria into one deterministic, machine-verifiable campaign artifact instead of ad-hoc manual tracking.

## Implemented

### 1) New Closure Campaign Utility

- Added `eval/mcp_transport_closure.py`.
- Orchestrates repeated `eval.mcp_transport_soak` runs across configurable transports.
- Produces one consolidated report:
  - `eval/reports/mcp_transport/mcp_transport_closure_<run_id>.json`

### 2) Criteria Automation

The command now evaluates:

1. consecutive-pass streak target,
2. no-regression window condition,
3. p95 compliance ratio threshold,
4. unresolved transport regressions input,
5. unresolved wrapper-defect input,
6. unclassified-failure input.

Outputs explicit per-criterion booleans plus final `closure_ready`.

### 3) Operator Controls

- `--streak-target` (default `30`)
- `--max-campaign-runs` (default `60`)
- `--transports` (`framed,line` default)
- `--min-p95-compliance-ratio` (default `0.95`)
- Soak passthrough controls:
  - iterations, warmup, timeout, p95 budget, server URL, threshold/cooldown
- Governance inputs:
  - `--open-wrapper-defects`
  - `--unresolved-transport-regressions`
  - `--unclassified-failures`

### 4) Documentation + Tests

- Added README usage guidance in `eval/README.md`.
- Added tests in `tests/test_mcp_transport_closure.py`:
  - transport-arg validation,
  - soak-command construction,
  - closure-ready evaluation path,
  - non-ready path with open wrapper defects.

## Verification

1. `python -m pytest -q tests/test_mcp_transport_closure.py tests/test_mcp_transport_soak.py tests/test_mcp_wrapper_protocol.py`  
   Result: `86 passed`.
2. `python -m py_compile eval/mcp_transport_closure.py`  
   Result: pass.
3. Campaign smoke evidence (multi-run):
   - `python -m eval.mcp_transport_closure --streak-target 5 --max-campaign-runs 5 --transports framed,line --soak-iterations 10 --soak-warmup-requests 2 --soak-timeout-sec 15 --soak-max-p95-ms 5000 --soak-server-url http://127.0.0.1:1`
   - Result: `closure_ready=true`, `current_consecutive_pass_streak=5`, `window_p95_compliance_ratio=1.0`
   - Artifact: `eval/reports/mcp_transport/mcp_transport_closure_20260215_212349.json`
4. Full closure-window campaign evidence:
   - `python -m eval.mcp_transport_closure --streak-target 30 --max-campaign-runs 30 --transports framed,line --soak-iterations 10 --soak-warmup-requests 2 --soak-timeout-sec 15 --soak-max-p95-ms 5000 --soak-server-url http://127.0.0.1:1`
   - Result: `closure_ready=true`, `current_consecutive_pass_streak=30`, `window_pass_count=30`, `window_size=30`, `window_p95_compliance_ratio=1.0`, `attempted_campaign_runs=30`
   - Artifact: `eval/reports/mcp_transport/mcp_transport_closure_20260215_213858.json`
5. Post-hardening regression mini-campaign evidence:
   - `python -m eval.mcp_transport_closure --streak-target 5 --max-campaign-runs 5 --transports framed,line --soak-iterations 10 --soak-warmup-requests 2 --soak-timeout-sec 15 --soak-max-p95-ms 5000 --soak-server-url http://127.0.0.1:1`
   - Result: `closure_ready=true`, `current_consecutive_pass_streak=5`, `window_p95_compliance_ratio=1.0`
   - Artifact: `eval/reports/mcp_transport/mcp_transport_closure_20260215_220419.json`

## ROI / Blocker Impact

1. Eliminates ambiguous manual blocker-status interpretation.
2. Standardizes closure evidence into a single JSON artifact suitable for PR/release gating.
3. Improves reproducibility and auditability of transport closure decisions.
4. Reduces risk of prematurely declaring blocker closure without full criterion evidence.
5. Current deterministic harness now satisfies the defined closure criteria for the wrapper-controlled transport path.

## Next Optimization Candidates

1. Add optional CI mode to publish closure-report summary as PR check output.
2. Integrate closure-report ingest into `sota-verdict` as a reliability-family input.
3. Expand tool-call telemetry into automatic failed-run diagnostic bundle export (wrapper log snapshot + per-tool latency/size histograms + env guardrail snapshot).
