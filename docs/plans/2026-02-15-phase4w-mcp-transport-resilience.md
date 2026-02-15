# Phase 4W Plan: MCP Transport Resilience Hardening

Date: 2026-02-15  
Owner: Codex  
Status: Implemented in branch `feat/phase4v-task-metadata-cursor-compliance`

## Objective

Reduce MCP session fragility when backends or host transports are unstable by hardening:
1. stdio message parsing resilience,
2. backend outage handling behavior, and
3. request dispatch pressure control.

## Implemented

1. Framed parser recovery:
   - `_read_rpc_message` now treats malformed framed payloads as recoverable noise.
   - invalid `Content-Length` and invalid framed JSON no longer terminate the read loop.
   - truncated framed payloads are logged and treated as stream-end only when bytes are missing.

2. Backend circuit-breaker fast-fail:
   - `make_request_with_retry` now short-circuits repeated outage windows with a bounded cooldown.
   - consecutive connection/timeout failures open a circuit (`MUNINN_MCP_BACKEND_FAILURE_THRESHOLD`, `MUNINN_MCP_BACKEND_COOLDOWN_SEC`).
   - requests during cooldown fail fast instead of consuming full retry windows.

3. Dispatch backpressure:
   - background-dispatch queue now uses a bounded semaphore (`MUNINN_MCP_DISPATCH_QUEUE_LIMIT`).
   - saturation now returns deterministic JSON-RPC error `-32001` for request messages.
   - queued slot is always released via done-callback, preventing leakage under normal completion.

4. Transport-close containment:
   - `send_json_rpc` now guards broken-pipe write failures and marks transport closed.
   - subsequent writes are dropped without repeated exceptions.

## Validation

1. `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
2. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_mcp_wrapper_protocol.py`
3. Result: `56 passed`
4. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_phase_hygiene.py tests/test_ollama_local_benchmark.py tests/test_mcp_wrapper_protocol.py`
5. Result: `92 passed`
6. `python -m eval.phase_hygiene --max-open-prs 1 --pytest-command ""`
7. Result: `PASS` (`eval/reports/hygiene/phase_hygiene_20260215_064545.json`)

## ROI / Impact

1. Faster failure signaling under backend outage reduces blocked tool-call windows and improves operator recovery speed.
2. Queue backpressure prevents unbounded dispatch pileup, protecting overall session responsiveness.
3. Parser recovery reduces accidental wrapper exits caused by malformed framed input.

## Blockers / Notes

1. External Muninn MCP tool call from this session (`muninn/search_memory`) still timed out at 120s post-reboot; wrapper code hardening is complete, but live host process restart + soak verification is still required to confirm end-to-end recovery.

## Follow-up Opportunities

1. Add optional periodic backend health-check telemetry with rolling failure counters.
2. Add dynamic queue limit policy (auto-scale based on request latency/timeout trend).
3. Add live soak harness for repeated `tools/call` under induced backend faults.
