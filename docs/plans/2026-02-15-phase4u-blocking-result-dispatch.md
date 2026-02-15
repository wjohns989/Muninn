# Phase 4U Plan: Blocking `tasks/result` Compliance + Responsive Dispatch

Date: 2026-02-15  
Owner: Codex  
Status: Implemented in branch `feat/phase4u-task-dispatch-blocking-compliance`

## Objective

Resolve protocol-vs-runtime tension discovered during PR review by keeping MCP lifecycle-compliant blocking behavior for `tasks/result` while preserving wrapper responsiveness under concurrent requests.

## Research Basis

- MCP lifecycle guidance (result retrieval semantics):
  - https://modelcontextprotocol.io/specification/2025-11-05/basic/lifecycle
- MCP 2025-11-25 schema reference:
  - https://raw.githubusercontent.com/modelcontextprotocol/specification/main/schema/2025-11-25/schema.json

## Implemented

1. Restored blocking result semantics:
   - `tasks/result` now waits until task status becomes terminal or `input_required`.

2. Wrapper responsiveness hardening:
   - main dispatch loop now routes potentially blocking methods (`tasks/result`, `tools/call`) to background worker threads.
   - dispatch guard added so background thread exceptions are logged without crashing stdio loop.

3. Output-channel integrity:
   - `send_json_rpc` now uses a process-wide stdout write lock to avoid interleaved JSON payload corruption under concurrent writes.

4. Review-thread hardening retained:
   - terminal-cancel error messages remain non-reflective (no raw user `taskId` reflection).
   - async worker internal errors remain generic for client output (details remain log-side only).

## Validation

1. `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
2. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_mcp_wrapper_protocol.py`
3. Result: `50 passed`
4. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_phase_hygiene.py tests/test_ollama_local_benchmark.py tests/test_mcp_wrapper_protocol.py`
5. Result: `86 passed`
6. `python -m eval.phase_hygiene --max-open-prs 1 --pytest-command ""`
7. Result: `PASS` (`eval/reports/hygiene/phase_hygiene_20260215_055320.json`)

## ROI / Impact

1. Preserves MCP lifecycle correctness (`tasks/result` wait semantics) without serial-channel starvation.
2. Improves multi-request robustness for assistants that pipeline polling, health probes, and tool calls concurrently.
3. Reduces risk of malformed outbound JSON under concurrent task/status emissions.

## Follow-up opportunities

1. Add bounded-wait controls (`maxWaitMs`) for environments that require strict latency ceilings.
2. Add `input_required` continuation flow (`task/continue` style wrapper contract) for interactive destructive tools.
3. Add optional persisted task registry backing for restart continuity.
