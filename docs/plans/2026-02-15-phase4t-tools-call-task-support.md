# Phase 4T Plan: Task-Augmented tools/call + Task Governance

Date: 2026-02-15  
Owner: Codex  
Status: Implemented baseline in branch `feat/phase4t-tools-call-task-support`

## Objective

Close the remaining MCP tasking gap after Phase 4S by implementing:
1. task-augmented `tools/call` execution (`params.task`),
2. `notifications/tasks/status` emission on state transitions, and
3. deterministic task registry retention + cursor pagination governance.

## Research Basis

Implementation semantics were checked against official MCP lifecycle guidance and schema references:

- MCP lifecycle docs (`tasks/result`, `tasks/cancel`, task notifications):
  - https://modelcontextprotocol.io/specification/2025-11-05/basic/lifecycle
- MCP schema source (`ServerCapabilities.tasks`, task request/notification contracts):
  - https://raw.githubusercontent.com/modelcontextprotocol/specification/main/schema/2025-11-25/schema.json

## Implemented

1. Capabilities and tool metadata alignment:
   - initialize response now advertises:
     - `capabilities.tasks.requests["tools/call"]`
     - `capabilities.tasks.notifications.status`
   - tools now default `execution.taskSupport` to `optional`.

2. Task-augmented `tools/call` baseline:
   - `tools/call` now validates optional `params.task` object.
   - when task is provided, wrapper returns immediate `CreateTaskResult` style payload with generated `taskId` and model immediate-response metadata.
   - tool execution runs asynchronously in background worker and writes terminal task outcome (`completed`/`failed`).

3. Task status transition notifications:
   - wrapper emits `notifications/tasks/status` for task state transitions (`working`, `completed`, `cancelled`).

4. Deterministic task registry governance:
   - task TTL is validated/clamped (`MUNINN_MCP_TASK_TTL_MS`, `MUNINN_MCP_TASK_MAX_TTL_MS`).
   - expired tasks are purged on task operations.
   - retained-task cap enforced (`MUNINN_MCP_TASKS_MAX_RETAINED`) with deterministic eviction ordering.
   - `tasks/list` now supports deterministic cursor pagination (`cursor` decimal offset + optional `limit`, bounded by `MUNINN_MCP_TASKS_LIST_PAGE_SIZE`).

5. Lifecycle behavior corrections:
   - `tasks/result` now blocks until terminal/input-required state per MCP lifecycle semantics and returns related-task metadata (`io.modelcontextprotocol/related-task`) once terminal.
   - `tasks/result` now returns stored JSON-RPC error payload for failed/cancelled executions.
   - `tasks/cancel` now returns `-32602` for already-terminal tasks (schema-aligned) and stamps cancelled task error payload (`-32800` request-cancelled style).
   - terminal-cancel and internal-task-worker error messaging no longer reflect raw user-originated values in client-visible error strings.
   - wrapper dispatch now routes blocking methods (`tasks/result`, `tools/call`) through background worker threads with stdout-write locking, so lifecycle waits do not starve the JSON-RPC channel.

## Validation

1. `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
2. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_mcp_wrapper_protocol.py`
3. Result: `50 passed`
4. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_phase_hygiene.py tests/test_ollama_local_benchmark.py tests/test_mcp_wrapper_protocol.py`
5. Result: `86 passed`
6. `python -m eval.phase_hygiene --max-open-prs 1 --pytest-command ""`
7. Result: `PASS` (`eval/reports/hygiene/phase_hygiene_20260215_055320.json`)

## ROI / Impact

1. Eliminates remaining ambiguity for clients that prefer asynchronous `tools/call` orchestration over direct blocking responses.
2. Adds deterministic status observability (`notifications/tasks/status`) and reduces polling inefficiency.
3. Prevents unbounded in-memory task growth through TTL + retention governance.
4. Aligns cancel/result behavior with lifecycle semantics for safer multi-client interoperability.

## Blockers / Notes

1. During this tranche, direct Muninn MCP tool calls from this assistant session still intermittently returned `Transport closed`; code work proceeded via local repo/CLI validation and existing recovery runbook (`docs/plans/2026-02-15-mcp-transport-closed-recovery.md`).

## Follow-up opportunities

1. Add explicit `input_required` task path support for elicitation-driven tool workflows.
2. Add persistent task registry backing (optional) for cross-wrapper-restart task introspection.
3. Add task result payload size budget + truncation metadata policy for very large tool outputs.
