# Phase 5A.4: MCP `tasks/result` Host-Safe Wait Budget

Date: 2026-02-15  
Status: Implemented

## Objective

Eliminate one remaining transport-closure vector: indefinite `tasks/result` blocking that can exceed host-side request timeout windows.

## Root Cause

- `tasks/result` was designed to block while a task remained non-terminal.
- In external host runtimes with hard request ceilings (commonly ~120s), long waits can trigger host transport teardown before task completion.
- This can manifest as intermittent transport closure despite otherwise healthy wrapper behavior.

## Implemented

1. Added configurable max blocking budget for `tasks/result`:
   - `MUNINN_MCP_TASK_RESULT_MAX_WAIT_SEC`
2. Default behavior now uses host-safe derived budget:
   - `MUNINN_MCP_HOST_TOOLS_CALL_TIMEOUT_SEC - MUNINN_MCP_TOOL_CALL_DEADLINE_MARGIN_SEC`
3. On budget exhaustion with non-terminal task:
   - wrapper returns deterministic retryable JSON-RPC error:
     - code: `-32002`
     - message: continue polling `tasks/get` and retry `tasks/result`
4. Added diagnostic log event for exhausted waits (elapsed seconds + request id).

## Verification

1. `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`  
   Result: pass.
2. `python -m pytest -q tests/test_mcp_wrapper_protocol.py tests/test_mcp_transport_soak.py tests/test_mcp_transport_closure.py`  
   Result: `92 passed`.
3. `python -m pytest -q tests/test_memory_user_profile.py tests/test_ingestion_discovery.py`  
   Result: `5 passed`.
4. Transport soak regression check:
   - `python -m eval.mcp_transport_soak --iterations 10 --warmup-requests 2 --timeout-sec 15 --transport framed --max-p95-ms 5000 --server-url http://127.0.0.1:1`
   - Artifact: `eval/reports/mcp_transport/mcp_transport_soak_20260215_220359.json`
5. Closure mini-campaign post-change:
   - `python -m eval.mcp_transport_closure --streak-target 5 --max-campaign-runs 5 --transports framed,line --soak-iterations 10 --soak-warmup-requests 2 --soak-timeout-sec 15 --soak-max-p95-ms 5000 --soak-server-url http://127.0.0.1:1`
   - Artifact: `eval/reports/mcp_transport/mcp_transport_closure_20260215_220419.json`
   - Result: `closure_ready=true`, streak `5`, p95 ratio `1.0`.

## ROI / Blocker Impact

1. Prevents `tasks/result` wait path from silently overrunning host timeout limits.
2. Converts transport-teardown class failures into deterministic, recoverable retry flow.
3. Further narrows remaining intermittency risk to external host/runtime transport behavior outside wrapper control.

## Newly Discovered Optimization

MCP task semantics references appear to differ across documents (blocking-oriented text vs immediate-error guidance for non-completed states).  
To reduce cross-host ambiguity risk, next candidate is an explicit compatibility switch (e.g., strict-blocking vs immediate-retry mode) with default selected by negotiated protocol/client profile.

## Sources

- Model Context Protocol 2025-11-05 specification root: https://modelcontextprotocol.io/specification/2025-11-05  
- MCP tasks guidance page: https://modelcontextprotocol.io/specification/2025-11-05/client/tasks  
- MCP streamable-http/tasks SEP draft excerpt (non-completed `tasks/result` immediate error language): https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/draft/basic/transports.mdx
