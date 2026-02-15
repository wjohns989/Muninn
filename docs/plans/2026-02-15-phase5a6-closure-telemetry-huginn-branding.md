# Phase 5A.6: Closure Telemetry + Huginn Browser Branding

Date: 2026-02-15  
Status: Implemented

## Objective

1. Improve transport blocker diagnostics by embedding compatibility and error-code telemetry directly in closure artifacts.
2. Improve standalone usability by clearly presenting browser-first mode as **Huginn** while retaining **Muninn** naming for MCP-attached assistant mode.

## Implemented

### 1) Transport Closure Telemetry Enrichment

- `eval.mcp_transport_soak` now supports compatibility telemetry controls:
  - `--task-result-mode auto|blocking|immediate_retry`
  - `--task-result-auto-retry-clients "<csv tokens>"`
- Soak reports now include:
  - `config.task_result_mode`
  - `config.task_result_auto_retry_clients`
- `eval.mcp_transport_closure` now forwards compatibility telemetry controls:
  - `--soak-task-result-mode`
  - `--soak-task-result-auto-retry-clients`
- Closure report transport entries now include:
  - per-transport `error_codes`
  - `task_result_mode`
  - `task_result_auto_retry_clients`
- Closure report now emits campaign-level telemetry rollups:
  - `error_code_totals`
  - `task_result_mode_distribution`
  - `task_result_auto_retry_profile_distribution`
  - `retryable_task_result_error_count`
  - `retryable_task_result_error_ratio`

### 2) Huginn Browser-Mode Branding + Practical UI Wiring

- Browser control center branding updated:
  - title and hero now identify browser mode as **Huginn**.
  - explicit mode chip added (`Huginn (browser)`) for quick operator context.
  - profile-context section copy now reflects Huginn/Muninn shared engine behavior.
- Standalone launcher and packaging UX updated:
  - `muninn_standalone.py` help/description now names **Huginn** as browser-first mode.
  - `scripts/build_standalone.py` default executable name updated to `HuginnControlCenter`.
  - README standalone instructions updated to distinguish:
    - Huginn mode = browser-first standalone
    - Muninn mode = MCP wrapper for active assistant sessions.
- Server dashboard load fallback heading now reports `Huginn UI unavailable`.

## Verification

1. `python -m py_compile eval/mcp_transport_soak.py eval/mcp_transport_closure.py mcp_wrapper.py muninn_standalone.py scripts/build_standalone.py server.py`  
   Result: pass.
2. `python -m pytest -q tests/test_mcp_transport_closure.py tests/test_mcp_transport_soak.py tests/test_mcp_wrapper_protocol.py tests/test_standalone_entrypoint.py tests/test_build_standalone.py tests/test_memory_user_profile.py tests/test_ingestion_discovery.py`  
   Result: `111 passed`.
3. Soak regression evidence:
   - `eval/reports/mcp_transport/mcp_transport_soak_20260215_224206.json` (pass)
4. Closure regression evidence with telemetry:
   - `eval/reports/mcp_transport/mcp_transport_closure_20260215_224225.json`
   - `closure_ready=true`, streak `5`, p95 ratio `1.0`
   - telemetry includes mode/profile distributions and error-code totals.

## ROI / Blocker Impact

1. Gives release gates machine-readable visibility into compatibility policy distribution and error-code composition, reducing ambiguous blocker interpretation.
2. Improves root-cause velocity when transport incidents recur by distinguishing policy-driven retries from backend/transport faults.
3. Reduces user/operator confusion by making standalone/browser workflow naming explicit (Huginn) while preserving Muninn MCP identity.

## Next Optimization Candidates

1. Add optional `tasks/result` non-terminal scenario to soak harness to intentionally exercise `-32002` paths and validate compatibility-mode behavior under load.
2. Add CI comparator that fails when closure telemetry mode distribution deviates from expected policy (for deterministic release posture).
3. Add UI mode toggle/help link that explains Huginn vs Muninn execution surfaces directly in-dashboard.
