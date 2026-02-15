# Phase 4AD Plan: MCP Explicit-Deadline Overrun Guardrail

Date: 2026-02-15
Status: Implemented (branch-local, unmerged)
Owner: Codex

## Problem
Even with host-timeout-derived defaults, an explicit `MUNINN_MCP_TOOL_CALL_DEADLINE_SEC` value could be set above host-safe budgets and reintroduce timeout-window risk.

## Objectives
1. Prevent unsafe explicit deadline misconfiguration by default.
2. Preserve explicit expert override path when truly needed.
3. Keep guardrail behavior deterministic and test-covered.

## Implemented
1. Added explicit-deadline overrun guardrail in `_get_tool_call_deadline_seconds`:
   - explicit deadline now clamps to host-safe budget by default when above safe threshold,
   - host-safe budget remains `MUNINN_MCP_HOST_TOOLS_CALL_TIMEOUT_SEC - MUNINN_MCP_TOOL_CALL_DEADLINE_MARGIN_SEC`.
2. Added opt-out for expert sessions:
   - `MUNINN_MCP_TOOL_CALL_DEADLINE_ALLOW_OVERRUN=1` preserves explicit over-budget values.
3. Preserved existing semantics:
   - `MUNINN_MCP_TOOL_CALL_DEADLINE_SEC <= 0` still disables deadline budgeting.

## Regression Coverage
1. Added protocol tests for:
   - explicit over-budget value clamped by default,
   - explicit over-budget value allowed when opt-out flag enabled.

## Verification
1. `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
2. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_mcp_wrapper_protocol.py`
   - Result: `70 passed`
3. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_phase_hygiene.py tests/test_mcp_wrapper_protocol.py`
   - Result: `77 passed`
4. Live post-restart MCP sanity checks:
   - `get_model_profiles` succeeded,
   - `add_memory` succeeded.

## Operational Guidance
1. Keep default guardrail on (`MUNINN_MCP_TOOL_CALL_DEADLINE_ALLOW_OVERRUN` unset/false).
2. Enable overrun only in controlled diagnostics with known host timeout behavior.
3. Prefer host-timeout-derived defaults over static explicit deadlines for broad client compatibility.

## Residual Risk
1. External host/runtime variability still requires ongoing observation before fully closing the blocker.
2. Overrun opt-out can intentionally reintroduce risk if used without matching host timeout policy.
