# Phase 5A.3: MCP Tool-Call Telemetry Hardening

Date: 2026-02-15  
Status: Implemented

## Objective

Improve transport-timeout root-cause visibility by emitting deterministic per-tool-call telemetry from the wrapper for both direct and task-backed execution paths.

## Implemented

1. Added per-tool-call telemetry context in `mcp_wrapper.py`:
   - `elapsed_ms`
   - response message count
   - response byte totals and max message size on stdio
   - initial/remaining deadline-budget snapshots
2. Added near-timeout warning threshold:
   - `MUNINN_MCP_TOOL_CALL_WARN_MS` (default `90000`)
   - call telemetry logs at warning level once elapsed time crosses threshold.
3. Added deterministic metric capture during `send_json_rpc` emission for matching request IDs so task-backed and direct call paths are both covered.
4. Added protocol tests in `tests/test_mcp_wrapper_protocol.py`:
   - response-metric capture behavior (`_record_tool_call_response_metrics`)
   - warn-threshold env parsing fallback behavior (`_get_tool_call_warn_ms`)

## Verification

1. `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`  
   Result: pass.
2. `python -m pytest -q tests/test_mcp_wrapper_protocol.py tests/test_mcp_transport_soak.py tests/test_mcp_transport_closure.py`  
   Result: `88 passed`.
3. `python -m pytest -q tests/test_memory_user_profile.py tests/test_ingestion_discovery.py`  
   Result: `5 passed`.

## ROI / Blocker Impact

1. Cuts diagnosis time for intermittent transport-close reports by exposing per-call wall-time and response-size evidence.
2. Makes timeout-adjacent behavior auditable against configured deadline budgets and warning thresholds.
3. Provides concrete signal to separate wrapper-path regressions from external host/runtime transport issues.

## Next Optimization Candidates

1. Emit structured JSON telemetry lines for easier ingestion by CI/observability pipelines.
2. Add optional automatic diagnostic bundle capture on first near-timeout/timeout event per session.
3. Feed telemetry rollups into `eval.mcp_transport_closure` output as campaign-level summary metadata.
