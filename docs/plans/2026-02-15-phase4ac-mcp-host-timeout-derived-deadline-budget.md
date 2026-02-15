# Phase 4AC Plan: MCP Host-Timeout-Derived Deadline Budget

Date: 2026-02-15
Status: Implemented (branch-local, unmerged)
Owner: Codex

## Problem
Wrapper-level deadline budgeting existed, but static defaults required manual retuning across host clients with different `tools/call` timeout ceilings. This increased operational drift risk.

## Objectives
1. Adapt default deadline budget to host timeout policy automatically.
2. Preserve explicit operator override/disable controls.
3. Keep deterministic guardrails against invalid derived budgets.

## Implemented
1. Added host-timeout-derived budgeting path in `_get_tool_call_deadline_seconds`:
   - `MUNINN_MCP_HOST_TOOLS_CALL_TIMEOUT_SEC` (default `120`)
   - `MUNINN_MCP_TOOL_CALL_DEADLINE_MARGIN_SEC` (default `10`)
   - Derived default: `host_timeout - margin`
2. Preserved explicit override/disable behavior:
   - `MUNINN_MCP_TOOL_CALL_DEADLINE_SEC` has precedence
   - `MUNINN_MCP_TOOL_CALL_DEADLINE_SEC <= 0` disables deadline budget
3. Added safety clamp:
   - derived non-positive values now clamp to `1s`
   - invalid/non-finite values fall back to safe defaults with logging

## Regression Coverage
1. Added protocol tests for:
   - explicit override behavior,
   - explicit disable behavior,
   - derived host-timeout-minus-margin behavior,
   - minimum clamp behavior when margin exceeds host timeout.

## Verification
1. `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
2. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_mcp_wrapper_protocol.py`
   - Result: `68 passed`
3. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_phase_hygiene.py tests/test_mcp_wrapper_protocol.py`
   - Result: `75 passed`

## Operational Guidance
1. Keep explicit `MUNINN_MCP_TOOL_CALL_DEADLINE_SEC` unset to use adaptive host-derived policy.
2. Set `MUNINN_MCP_HOST_TOOLS_CALL_TIMEOUT_SEC` to the real client timeout ceiling and keep a positive margin.
3. Use explicit override only for controlled environments where host timeout policy is known stable.

## Residual Risk
1. Host/runtime rollout remains required for external clients to consume this hardening.
2. Misconfigured host timeout/margin values can still create suboptimal behavior; logs now surface invalid inputs.
