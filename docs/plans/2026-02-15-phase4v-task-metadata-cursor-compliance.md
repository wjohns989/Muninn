# Phase 4V Plan: Task Metadata + Cursor Compliance Hardening

Date: 2026-02-15  
Owner: Codex  
Status: Implemented in branch `feat/phase4v-task-metadata-cursor-compliance`

## Objective

Close remaining high-ROI MCP task-contract mismatches by aligning:
1. related-task metadata keying (`taskId`),
2. `tasks/list` cursor semantics (opaque token contract), and
3. task polling guidance (`pollInterval`) for created tasks.

## Research Basis

- MCP lifecycle guidance:
  - https://modelcontextprotocol.io/specification/2025-11-05/basic/lifecycle
- MCP schema source (2025-11-25):
  - https://raw.githubusercontent.com/modelcontextprotocol/specification/main/schema/2025-11-25/schema.json

## Implemented

1. Related-task metadata correction:
   - `_meta.io.modelcontextprotocol/related-task` now uses `{"taskId": ...}` instead of non-spec `{"id": ...}`.

2. Opaque cursor pagination:
   - `tasks/list` now emits opaque `nextCursor` tokens via base64url-encoded internal cursor payload (`tasks:v1:<offset>`).
   - parser accepts opaque token contract and preserves legacy numeric cursors for backward compatibility.
   - invalid cursor errors now explicitly reference opaque-token expectation.

3. Poll guidance for clients:
   - task objects created through task-augmented `tools/call` now include `pollInterval` (env-configurable via `MUNINN_MCP_TASK_POLL_INTERVAL_MS`).

4. Test coverage updates:
   - protocol tests now assert `taskId` key in related-task metadata,
   - pagination tests assert opaque cursor contract and decode path,
   - task-create tests assert poll interval presence.

## Validation

1. `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
2. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_mcp_wrapper_protocol.py`
3. Result: `52 passed`
4. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_phase_hygiene.py tests/test_ollama_local_benchmark.py tests/test_mcp_wrapper_protocol.py`
5. Result: `88 passed`
6. `python -m eval.phase_hygiene --max-open-prs 1 --pytest-command ""`
7. Result: `PASS` (`eval/reports/hygiene/phase_hygiene_20260215_061319.json`)

## ROI / Impact

1. Removes a concrete schema-level mismatch in task-result correlation metadata (`taskId`), reducing client integration ambiguity.
2. Aligns task pagination contract with opaque-cursor semantics, enabling future cursor evolution without client breakage.
3. Improves client polling behavior consistency through explicit per-task poll interval hints.

## Blockers / Notes

1. Muninn MCP tool server from this assistant session remains operationally unstable (`Transport closed` earlier, 120-second tool-call deadlines after reboot); planning/memory retrieval used local docs + GitHub + MCP spec sources in this tranche.

## Follow-up opportunities

1. Add optional signed cursor payloads to prevent cursor tampering in stricter deployment modes.
2. Add explicit cursor version migration tests for future cursor payload evolution.
3. Implement advanced `input_required` continuation path (`task/continue` style contract) for interactive tool workflows.
