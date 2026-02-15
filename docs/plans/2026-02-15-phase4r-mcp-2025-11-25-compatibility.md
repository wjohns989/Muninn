# Phase 4R Plan: MCP 2025-11-25 Compatibility Baseline

Date: 2026-02-15  
Owner: Codex  
Status: Implemented baseline in branch `feat/phase4r-mcp-2025-11-25-compat`

## Objective

Advance wrapper interoperability with the MCP 2025-11-25 protocol by adding tasks capability/list behavior, honoring elicitation capability defaults, and tightening tool metadata declarations for task-aware clients.

## Implemented

1. Initialize capability expansion:
   - wrapper now advertises `capabilities.tasks.list` support.
2. Client elicitation capability parsing:
   - wrapper now parses client `capabilities.elicitation`,
   - empty elicitation capability object (`{}`) is interpreted as form-only support for backwards compatibility.
3. `tasks/list` server method support:
   - `tasks/list` now validates init lifecycle and params shape,
   - cursor type is validated when present,
   - deterministic empty result shape returned (`{"tasks": []}`) while async task lifecycle is not yet enabled.
4. Tool metadata compatibility hardening:
   - each tool now explicitly returns `execution.taskSupport` (default `forbidden`),
   - annotations expanded with `readOnlyHint`, `destructiveHint`, `idempotentHint`, and `openWorldHint`.
5. Protocol test coverage additions:
   - initialize capability payload checks for `tasks.list`,
   - elicitation default + mode parsing checks,
   - `tasks/list` lifecycle/params/result checks,
   - tool metadata contract checks for richer annotations + `execution.taskSupport`.

## Validation

1. `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
2. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_mcp_wrapper_protocol.py`
3. Result: `36 passed`
4. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_ollama_local_benchmark.py tests/test_phase_hygiene.py tests/test_mcp_wrapper_protocol.py`
5. Result: `70 passed`
6. `python -m eval.phase_hygiene --max-open-prs 1 --pytest-command ""`
7. Result: `PASS` (`eval/reports/hygiene/phase_hygiene_20260215_044031.json`)

## ROI / Impact

1. Reduces client integration ambiguity by exposing explicit `tasks/list` support and deterministic behavior.
2. Improves compatibility with newer MCP clients expecting elicitation default semantics and tool-level task metadata.
3. Clarifies task-augmentation posture (`forbidden` by default) to prevent accidental task-mode misuse for current synchronous tools.

## Follow-up opportunities

1. Implement full task lifecycle (`tasks/get`, `tasks/result`, `tasks/cancel`) for long-running tool operations.
2. Add optional async/task mode for selected tools where `execution.taskSupport` can be promoted to `optional`.
3. Wire server-initiated `elicitation/create` flows for sensitive/interactive operations with mode-aware guardrails.
