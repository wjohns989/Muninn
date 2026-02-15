# Phase 4AB Plan: MCP Startup-Recovery Budget Gating

Date: 2026-02-15
Status: Implemented (branch-local, unmerged)
Owner: Codex

## Problem
Even with tool-call deadline budgeting, retry paths could still invoke startup recovery (`ensure_server_running`) late in the lifecycle when little budget remained. That recovery path can consume significant wall time and risk crossing host-side timeout windows.

## Objectives
1. Prevent low-remaining-budget startup recovery from extending tool-call wall time.
2. Keep recovery behavior for healthy budget windows.
3. Preserve deterministic behavior under intermittent backend outages.

## Implemented
1. Added startup-recovery budget threshold:
   - Env: `MUNINN_MCP_STARTUP_RECOVERY_MIN_BUDGET_SEC`
   - Default: `28` seconds
2. Added shared budget gate helper for startup recovery decisions.
3. Applied budget gating to preflight in `handle_call_tool`:
   - skips `ensure_server_running()` when remaining budget is below threshold.
4. Applied budget gating to retry path in `make_request_with_retry`:
   - skips late startup recovery when remaining budget is below threshold.
5. Added diagnostic logging for skipped-recovery events to aid production tuning.

## Regression Coverage
1. Added protocol tests to verify:
   - retry startup recovery is skipped under low remaining budget,
   - preflight startup recovery is skipped when configured tool-call budget is below threshold.

## Verification
1. `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
2. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_mcp_wrapper_protocol.py`
   - Result: `64 passed`
3. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_phase_hygiene.py tests/test_mcp_wrapper_protocol.py`
   - Result: `71 passed`

## Operational Guidance
1. Keep `MUNINN_MCP_TOOL_CALL_DEADLINE_SEC` below host timeout ceilings (default `110`).
2. Keep `MUNINN_MCP_STARTUP_RECOVERY_MIN_BUDGET_SEC` near worst-case startup-recovery duration (default `28`).
3. Tune both values together per host/client timeout behavior.

## Residual Risk
1. Host-specific transport behavior remains outside wrapper control.
2. Aggressive threshold values may reduce startup-recovery attempts during outages; monitor failure patterns and tune accordingly.
3. External host runtime still needs restart/rollout validation; an in-session `muninn/add_memory` call continued to hit host-side `tools/call` 120s timeout on 2026-02-15.
