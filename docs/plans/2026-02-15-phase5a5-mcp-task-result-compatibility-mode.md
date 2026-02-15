# Phase 5A.5: MCP `tasks/result` Compatibility Mode

Date: 2026-02-15  
Status: Implemented

## Objective

Handle cross-host/spec semantic drift for `tasks/result` without sacrificing transport stability.

## Problem

Current MCP references show mixed guidance for non-terminal task handling:

1. Utility-client tasks guidance indicates `tasks/result` can block until terminal state.
2. SEP-draft guidance indicates non-completed tasks should return an immediate error.

Without explicit compatibility handling, heterogeneous clients can exhibit different expectations and trigger avoidable timeout/transport failures.

## Implemented

1. Added explicit runtime policy:
   - `MUNINN_MCP_TASK_RESULT_MODE` in:
     - `auto` (default)
     - `blocking`
     - `immediate_retry`
2. Added client-profile driven auto-mode:
   - `MUNINN_MCP_TASK_RESULT_AUTO_RETRY_CLIENTS` (default includes `claude desktop`, `claude code`, `cursor`, `windsurf`, `continue`).
   - In `auto`, matching client profiles use immediate-retry behavior for non-terminal tasks.
3. Added per-request override:
   - `tasks/result` now accepts optional `params.wait` boolean:
     - `true` forces blocking behavior (still bounded by Phase 5A.4 host-safe wait budget),
     - `false` forces immediate-retry behavior.
4. Added deterministic validation:
   - non-boolean `params.wait` returns `-32602`.
5. Non-terminal immediate-retry path returns deterministic retryable error:
   - code `-32002`
   - message instructing poll via `tasks/get` and retry `tasks/result`.

## Verification

1. `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`  
   Result: pass.
2. `python -m pytest -q tests/test_mcp_wrapper_protocol.py tests/test_mcp_transport_soak.py tests/test_mcp_transport_closure.py`  
   Result: `98 passed`.
3. `python -m pytest -q tests/test_memory_user_profile.py tests/test_ingestion_discovery.py`  
   Result: `5 passed`.
4. Post-change soak regression evidence:
   - `eval/reports/mcp_transport/mcp_transport_soak_20260215_221650.json` (pass)
5. Post-change closure mini-campaign evidence:
   - `eval/reports/mcp_transport/mcp_transport_closure_20260215_221709.json`
   - `closure_ready=true`, streak `5`, p95 ratio `1.0`.

## ROI / Blocker Impact

1. Reduces client-integration ambiguity by making behavior explicit and configurable.
2. Preserves compatibility with both blocking-oriented and immediate-retry-oriented clients.
3. Further reduces host-side timeout transport-closure risk in client profiles known to have strict request ceilings.

## Sources

- MCP utility-client tasks guidance: https://modelcontextprotocol.io/specification/2025-11-05/client/tasks#tasksresult  
- MCP SEP draft (`tasks/result` non-completed immediate error guidance): https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/draft/sep-1686.md
