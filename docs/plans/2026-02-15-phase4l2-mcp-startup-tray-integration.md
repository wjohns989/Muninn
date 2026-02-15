# Phase 4L.2 - MCP Startup Ollama Trigger + Windows Tray MCP Controls

Date: 2026-02-15  
Branch: `feat/phase4k-roadmap-closure`

## Objective

Strengthen day-0 operator experience after reboot/session start by:
1. triggering Ollama startup at MCP wrapper process launch when user autostart is enabled, and
2. using the Windows tray wrapper as the primary operational bridge into browser UI and MCP diagnostics.

## Implemented

1. `mcp_wrapper.py` launch-time bootstrap:
   - added `_bootstrap_dependencies_on_launch()` with explicit flag controls:
     - `MUNINN_MCP_AUTOSTART_ON_LAUNCH` (default `true`)
     - `MUNINN_MCP_AUTOSTART_SERVER` (existing)
     - `MUNINN_MCP_AUTOSTART_OLLAMA` (existing)
   - `main()` now starts bootstrap in a background thread so startup is non-blocking while still triggering dependency readiness quickly.

2. `tray_app.py` Windows tray operational UX:
   - `Open Browser UI` action now works as an entrypoint even when server is offline (it auto-starts server thread first).
   - added Ollama startup path from tray (`Start Ollama`) with `MUNINN_TRAY_AUTOSTART_OLLAMA` control (default `true`).
   - added `MCP Health Check` tray action that performs:
     - server health check,
     - Ollama health check,
     - wrapper initialize + tools/list probe.
   - added `View MCP Wrapper Log` tray action for transport diagnostics.
   - improved probe execution to prefer `python.exe` when tray is running under `pythonw.exe` to preserve stdout piping for wrapper validation.

## Validation

1. `python -m py_compile mcp_wrapper.py tray_app.py tests/test_mcp_wrapper_protocol.py`
2. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_mcp_wrapper_protocol.py`
3. `echo '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\"}}' | python mcp_wrapper.py`
4. Result:
   - compile passed,
   - protocol tests `30 passed`,
   - wrapper initialize smoke output returned expected `result`.

## ROI and Optimization Notes

1. Startup latency reduction for first real interaction:
   - dependency warmup starts at wrapper launch, not first tool call.
2. Support burden reduction:
   - tray now has direct MCP health probe + wrapper log shortcut, reducing manual CLI triage.
3. User habit alignment:
   - browser UI can be opened directly from tray in one action, with server auto-start fallback.

## Follow-up Opportunities

1. Add a dedicated `/health/mcp` REST probe endpoint to expose wrapper lifecycle telemetry in browser UI.
2. Add optional tray toast notifications for degraded state transitions (`Ollama down`, `Server recovered`) to shorten mean-time-to-awareness.
