# Phase 4X Plan: MCP Transport Soak + Dispatch Policy Hardening

Date: 2026-02-15  
Owner: Codex  
Status: Implemented in branch `feat/phase4v-task-metadata-cursor-compliance`

## Objective

Convert transport hardening into repeatable evidence and close a newly discovered latency/amplification issue:
1. add deterministic soak verification for MCP transport behavior,
2. remove default behavior that can suppress/delay `tools/call` response determinism under faulted backends, and
3. avoid expensive preflight start probes when outage signals are already known.

## Root-Cause Findings

1. `tools/call` background dispatch under outage conditions could produce long/unstable response timing in live framed sessions.
2. `handle_call_tool` still preflighted `ensure_server_running()` on every call, which amplified outage latency even when autostart was disabled or a backend circuit cooldown was active.

## Implemented

1. Soak harness:
   - new utility: `eval/mcp_transport_soak.py`
   - supports `framed` and `line` transport modes, malformed-frame injection, warmup controls, and p95 latency budgets.
   - emits deterministic JSON reports to `eval/reports/mcp_transport/`.

2. Dispatch policy hardening:
   - `tasks/result` remains background-dispatched.
   - `tools/call` is now **foreground by default** for deterministic request-response behavior.
   - background `tools/call` remains available via opt-in env flag:
     - `MUNINN_MCP_BACKGROUND_TOOLS_CALL=1`

3. Preflight outage amplification fix:
   - `handle_call_tool` now skips preflight server-start probes when:
     - `MUNINN_MCP_AUTOSTART_SERVER=0`, or
     - backend circuit is already open.

4. Test coverage:
   - new `tests/test_mcp_transport_soak.py` for soak-helper math/path logic.
   - protocol tests expanded for dispatch-policy and preflight gating behavior.

## Validation

1. `python -m py_compile eval/mcp_transport_soak.py tests/test_mcp_transport_soak.py mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
2. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_mcp_transport_soak.py tests/test_mcp_wrapper_protocol.py tests/test_phase_hygiene.py tests/test_ollama_local_benchmark.py`
3. Result: `98 passed`
4. Soak command:
   - `python -m eval.mcp_transport_soak --iterations 6 --warmup-requests 1 --timeout-sec 12 --transport framed --server-url http://127.0.0.1:1 --failure-threshold 1 --cooldown-sec 30 --max-p95-ms 2500 --inject-malformed-frame`
5. Result: `PASS` (`eval/reports/mcp_transport/mcp_transport_soak_20260215_074136.json`)
6. Hygiene gate:
   - `python -m eval.phase_hygiene --max-open-prs 1 --pytest-command ""`
7. Result: `PASS` (`eval/reports/hygiene/phase_hygiene_20260215_074404.json`)

## ROI / Impact

1. Transport resiliency is now measurable and repeatable in CI/operator loops, not only anecdotal.
2. Default `tools/call` semantics are now deterministic under faulted backends, reducing client-perceived hangs.
3. Outage-mode request handling avoids repeated preflight startup penalties when policy or circuit state already indicates backend unavailability.

## Blockers / Notes

1. External MCP Muninn tool calls from this assistant session still hit host-side 120-second deadlines (`muninn/search_memory`), so host-process-level diagnostics/restart remain necessary despite local wrapper soak pass.

## Follow-up Opportunities

1. Add an optional `--method` switch to soak harness for `ping`/`tasks/list`/`tools/call` profiles.
2. Add a CI smoke profile that runs 3-5 soak iterations against a closed-port backend URL.
3. Add host-side diagnostics capture for external MCP timeout cases (wrapper PID, stderr, and health probe snapshots).
