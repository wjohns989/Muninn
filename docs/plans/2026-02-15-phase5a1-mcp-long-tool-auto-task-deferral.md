# Phase 5A.1: MCP Long-Tool Auto-Task Deferral for Host Timeout Stability

Date: 2026-02-15  
Status: Implemented (continuation tranche)

## Objective

Reduce external host-side `tools/call` timeout risk (120s window) by ensuring long-running MCP tools can return immediate task handles instead of waiting synchronously for full completion.

## Problem Statement

Even after deadline-budget and response-size hardening, long-running ingestion calls can still approach host timeout windows when invoked as synchronous `tools/call` operations.

## Implemented

### 1) Auto-Deferral Policy for Long Tools

- Added long-tool auto-task policy in `mcp_wrapper.py`:
  - `_get_auto_task_tool_names()`
  - `_client_declared_tasks_capability()`
  - `_should_auto_task_tool_call(...)`
- `tools/call` dispatch now auto-converts configured long tools into task mode when no `params.task` is provided.
- Default configured long-tool set:
  - `ingest_sources`
  - `ingest_legacy_sources`
  - `discover_legacy_sources`

### 2) Runtime Controls

- `MUNINN_MCP_AUTO_TASK_FOR_LONG_TOOLS` (default `1`)
  - enables/disables automatic deferral.
- `MUNINN_MCP_AUTO_TASK_TOOL_NAMES`
  - comma-separated tool-name allowlist for auto-deferral.
- `MUNINN_MCP_AUTO_TASK_REQUIRE_CLIENT_CAP` (default `0`)
  - when `1`, only auto-defers if client declared `tasks` capability.

### 3) Protocol Safety

- Auto-deferral only applies when:
  - request method is `tools/call`,
  - tool name is in configured long-tool set,
  - caller did not already provide `params.task`.
- Existing explicit task requests keep prior behavior unchanged.
- Non-long tools keep synchronous behavior unchanged.

## Verification

1. `python -m pytest -q tests/test_mcp_wrapper_protocol.py tests/test_mcp_transport_soak.py`  
   Result: `82 passed`.
2. `python -m pytest -q tests/test_memory_user_profile.py tests/test_ingestion_discovery.py`  
   Result: `5 passed`.

Added protocol tests:

- auto-defers configured long tool into task mode,
- can be disabled by env,
- can require client capability declaration.

## ROI / Blocker Impact

1. Reduces probability of host-side `120s` synchronous timeout on long ingestion tool calls by returning immediate task handles.
2. Preserves deterministic retry/poll semantics already implemented in MCP task lifecycle.
3. Keeps rollback surface small via env flags, allowing quick policy adjustment per host/client behavior.

Current blocker status:

- Reduced risk for long-running MCP tool paths.
- Blocker remains open pending rolling soak evidence in external host runtime.

## Newly Identified Follow-Up Optimizations

1. Add optional server-side progress notifications for long task workers to improve operator visibility during ingest-heavy sessions.
2. Add per-tool p95 wall-time counters in wrapper diagnostics to detect regression before host timeout windows are hit.
3. Add `task mode recommended` hint in `tools/list` metadata for long tools.

## Sources

- MCP transport specification: https://modelcontextprotocol.io/specification/draft/basic/transports
- MCP architecture and lifecycle references: https://modelcontextprotocol.io/docs/learn/architecture
- Requests timeout behavior reference: https://requests.readthedocs.io/en/latest/user/quickstart/#timeouts
