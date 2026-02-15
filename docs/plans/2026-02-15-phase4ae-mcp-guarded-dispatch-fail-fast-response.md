# Phase 4AE: MCP Guarded-Dispatch Fail-Fast Response

Date: 2026-02-15  
Branch: `feat/phase4v-task-metadata-cursor-compliance`

## Problem

`_dispatch_rpc_message_guarded` previously only logged unexpected dispatcher exceptions. For request messages (especially background-dispatched `tasks/result` and optional background `tools/call`), this could leave the request ID without a response, forcing host clients to wait until their tool-call timeout window and increasing transport-closed risk.

## Objectives

1. Ensure request messages receive deterministic JSON-RPC error replies when guarded dispatch fails unexpectedly.
2. Preserve existing behavior for notification-only messages (no `id`).
3. Keep behavior transport-safe by avoiding writes once transport is already marked closed.

## Implemented

### `mcp_wrapper.py`

- Updated `_dispatch_rpc_message_guarded` to capture `msg_id` and emit:
  - `code: -32603`
  - `message: "Internal error during request dispatch."`
- Emission occurs only when:
  - inbound message has an `id`, and
  - `_TRANSPORT_CLOSED` is not already set.

### `tests/test_mcp_wrapper_protocol.py`

- Added `test_dispatch_guard_returns_internal_error_for_request_id`:
  - forces dispatcher exception,
  - verifies deterministic `-32603` response for request ID.

## Verification

- `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_mcp_wrapper_protocol.py`
  - `71 passed`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_phase_hygiene.py tests/test_mcp_wrapper_protocol.py`
  - `78 passed`
- `python -m eval.mcp_transport_soak --iterations 10 --warmup-requests 2 --timeout-sec 15 --transport framed --max-p95-ms 5000`
  - `PASS` (`run_id=20260215_170548`, p95 ~= `80.95ms`)

## ROI

- Converts silent background dispatch hangs into immediate protocol-visible failures.
- Reduces host-side timeout accumulation that can cascade into `transport closed` session teardowns.
- Improves observability and recoverability for intermittent runtime exceptions in dispatch paths.

## Residual Risk

- Host-side runtime variability and external process restarts can still impact transport stability.
- Continue monitoring timeout-related envs and soak trends before marking blocker fully closed.