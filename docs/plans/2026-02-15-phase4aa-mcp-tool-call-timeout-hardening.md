# Phase 4AA Plan: MCP `tools/call` Timeout Hardening

Date: 2026-02-15
Status: Implemented (branch-local, unmerged)
Owner: Codex

## Problem
External host-side MCP sessions intermittently closed transport around 120s during long `tools/call` operations. Wrapper retries could exceed host channel limits because per-attempt timeouts were not bounded by an overall response budget.

## Objectives
1. Ensure wrapper returns deterministically before host-side 120s timeout windows.
2. Preserve retry behavior while constraining total wall-clock budget.
3. Eliminate known `delete_memory` path-segment encoding bug in wrapper transport.

## Implemented
1. Added bounded tool-call deadline budget in wrapper:
   - Env: `MUNINN_MCP_TOOL_CALL_DEADLINE_SEC`
   - Default: `110` seconds (set `<=0` to disable)
2. Extended retry engine to support absolute deadlines:
   - `make_request_with_retry(..., deadline_epoch=...)`
   - Per-attempt `timeout` is clamped to remaining budget.
   - Retry backoff respects remaining budget and aborts when exhausted.
   - Deterministic timeout error is raised on deadline exhaustion.
3. Routed all `handle_call_tool` backend requests through a deadline-aware request helper.
4. Fixed wrapper `delete_memory` endpoint path encoding with URL quoting.

## Regression Coverage
1. Added protocol tests for timeout hardening:
   - timeout clamping to remaining deadline budget,
   - fast failure when deadline is already exhausted,
   - delete-memory URL encoding + deadline propagation.

## Verification
1. `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
2. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_mcp_wrapper_protocol.py`
   - Result: `62 passed`

## Operational Guidance
1. Keep default `MUNINN_MCP_TOOL_CALL_DEADLINE_SEC=110` for environments with 120s host transport ceilings.
2. If host timeouts differ, tune budget to maintain safety margin below host channel timeout.
3. For deeply long-running operations, prefer task-based `tools/call` flows and polling APIs over single blocking calls.

## Residual Risk
1. Host/runtime variability outside wrapper process control may still affect transport stability.
2. If backend operations routinely exceed deadline budget, requests will fail fast by design; this should be handled via task mode or server-side optimization.
