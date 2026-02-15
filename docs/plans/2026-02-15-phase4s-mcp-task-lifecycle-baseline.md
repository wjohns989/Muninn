# Phase 4S Plan: MCP Task Lifecycle Baseline

Date: 2026-02-15  
Owner: Codex  
Status: Implemented baseline in branch `feat/phase4s-mcp-task-lifecycle-baseline`

## Objective

Close the highest-ROI MCP follow-up gap after Phase 4R by implementing baseline task lifecycle request handling (`tasks/get`, `tasks/result`, `tasks/cancel`) and aligning capability signaling with the 2025-11-25 schema.

## Research Basis

Implementation semantics were checked against the official MCP schema source:

- `https://raw.githubusercontent.com/modelcontextprotocol/specification/main/schema/2025-11-25/schema.json`
  - `GetTaskRequest` (`tasks/get`) with required `params.taskId`
  - `GetTaskPayloadRequest` (`tasks/result`) with required `params.taskId`
  - `CancelTaskRequest` (`tasks/cancel`) with required `params.taskId`
  - `ServerCapabilities.tasks` support flags (`list`, `cancel`, and request augmentation map)

## Implemented

1. Session state lifecycle support:
   - wrapper session state now includes a deterministic in-memory task registry.
2. Capability alignment:
   - initialize response now advertises both `capabilities.tasks.list` and `capabilities.tasks.cancel`.
3. New request handlers:
   - `tasks/get`: returns task state when found, validates lifecycle and params.
   - `tasks/result`: returns stored task result payload only for terminal task states.
   - `tasks/cancel`: transitions non-terminal tasks to `cancelled` with timestamped status update.
4. Deterministic validation/error behavior:
   - non-object params and missing/invalid `taskId` return explicit `-32602` errors.
   - unknown `taskId` returns explicit `-32602` errors.
   - terminal-state violations for cancel/non-terminal-result access return deterministic server errors.
5. Extended protocol tests:
   - task capability assertions,
   - task-get/result/cancel request-path validation and state transitions.

## Validation

1. `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
2. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_mcp_wrapper_protocol.py`
3. Result: `43 passed`
4. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_ollama_local_benchmark.py tests/test_phase_hygiene.py tests/test_mcp_wrapper_protocol.py`
5. Result: `77 passed`
6. `python -m eval.phase_hygiene --max-open-prs 1 --pytest-command ""`
7. Result: `PASS` (`eval/reports/hygiene/phase_hygiene_20260215_050011.json`)

## ROI / Impact

1. Removes a concrete protocol gap for task lifecycle polling/cancellation requests.
2. Improves client interoperability by matching schema-level method and parameter contracts.
3. Establishes a safe foundation for future long-running `tools/call` task augmentation without protocol churn.

## Follow-up opportunities

1. Add server-side creation of task records for task-augmented `tools/call`.
2. Emit `notifications/tasks/status` transitions for active task observers.
3. Introduce task retention/eviction policy and paginated cursor continuation semantics for high task volume.
