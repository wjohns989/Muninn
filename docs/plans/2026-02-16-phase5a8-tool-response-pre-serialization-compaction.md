# Phase 5A.8: Tool-Response Pre-Serialization Compaction

Date: 2026-02-16  
Status: Implemented

## Objective

Reduce intermittent host transport timeout risk from oversized MCP tool payload formatting by eliminating full-payload JSON serialization before truncation.

## Root-Cause Risk Addressed

Prior behavior truncated only after `_safe_json_dumps(result)`, which meant very large nested payloads could still incur large CPU/memory costs and wall-time spikes before response emission.

## Implemented

1. Added bounded response-preview controls in `mcp_wrapper.py`:
   - `MUNINN_MCP_TOOL_RESPONSE_PREVIEW_MAX_ITEMS` (default `200`)
   - `MUNINN_MCP_TOOL_RESPONSE_PREVIEW_MAX_DEPTH` (default `6`)
   - `MUNINN_MCP_TOOL_RESPONSE_PREVIEW_MAX_STRING_CHARS` (default `2000`)
2. Added deterministic payload compaction helper:
   - `_compact_tool_response_payload(...)`
   - applies depth/item/string limits before serialization,
   - handles circular references safely.
3. Updated `_format_tool_result_text(...)`:
   - compacts first,
   - then serializes and applies final text-size truncation.

## Verification

1. Compile checks:
   - `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
2. Targeted protocol/transport tests:
   - `python -m pytest -q tests/test_mcp_wrapper_protocol.py tests/test_mcp_transport_soak.py tests/test_mcp_transport_closure.py`
   - Result: `108 passed`.
3. New/expanded tests include:
   - collection compaction behavior,
   - depth compaction behavior,
   - long-string preview truncation behavior.

## ROI / Blocker Impact

1. Lowers worst-case wrapper formatting overhead for oversized tool payloads.
2. Reduces chance that response formatting itself contributes to host-side timeout windows.
3. Improves determinism and observability of payload shaping with explicit operator-tunable controls.
