# Phase 4L.1 - MCP Transport Closed Recovery + Framing Compatibility

Date: 2026-02-15  
Branch: `feat/phase4k-roadmap-closure`

## Objective

Resolve intermittent `tools/call failed: Transport closed` failures for the Muninn MCP server by:
1. hardening stdio transport parsing for cross-client framing differences, and
2. documenting deterministic recovery steps when a session already holds a dead transport.

## Implemented

1. `mcp_wrapper.py` now accepts both inbound transport styles:
   - newline-delimited JSON-RPC (`{"jsonrpc":"2.0", ...}\n`)
   - `Content-Length` framed JSON-RPC payloads
2. Added targeted protocol tests in `tests/test_mcp_wrapper_protocol.py`:
   - JSON-line parsing
   - `Content-Length` parsing
   - invalid header handling

## Validation

1. `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
2. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_mcp_wrapper_protocol.py`
3. Result: `28 passed`

## Operational Recovery Runbook (for already broken sessions)

If a current assistant/IDE session still returns `Transport closed`, apply:

1. Verify MCP registration:
   - `codex mcp get muninn`
2. Verify wrapper process launch path manually:
   - `python mcp_wrapper.py` (stdin smoke test, or use existing test command)
3. Verify backend services:
   - `curl http://localhost:42069/health`
   - `curl http://localhost:11434/api/version`
4. Restart the assistant/IDE MCP session (required if the transport object is already closed).
5. Re-test with a read-only tool call (`search_memory`/`get_project_goal`).

Note: Parser hardening fixes future session initialization compatibility; it does not resurrect an already closed transport object in a running client process.

## Continuation Update (Phase 4L.2)

Follow-on hardening now implemented in `docs/plans/2026-02-15-phase4l2-mcp-startup-tray-integration.md`:

1. Launch-time dependency bootstrap:
   - wrapper startup now triggers best-effort server/Ollama readiness when autostart flags are enabled.
2. Windows tray operational controls:
   - direct `Open Browser UI`,
   - `MCP Health Check` (server + Ollama + wrapper probe),
   - explicit `Start Ollama`,
   - `View MCP Wrapper Log`.
3. Verification increment:
   - protocol/startup tests now at `30 passed` (`tests/test_mcp_wrapper_protocol.py`).

## Continuation Update (Phase 4N)

Operational governance follow-on is now implemented in `docs/plans/2026-02-15-phase4n-policy-approval-manifest.md`:

1. Checkpoint approval manifests:
   - explicit approve/reject artifacts now bind checkpoint path + SHA-256.
2. Controlled checkpoint apply:
   - apply from checkpoint now requires approved manifest and integrity match checks.
3. Validation increment:
   - benchmark/policy command tests now include approval/apply flows (`16 passed`, `tests/test_ollama_local_benchmark.py`).

## Continuation Update (Phase 4O)

Governance-provenance follow-on is now implemented in `docs/plans/2026-02-15-phase4o-approval-provenance-context.md`:

1. Approval manifest provenance:
   - approval artifacts now support PR/commit/branch provenance context with validation and git fallback discovery.
2. Apply-report propagation:
   - checkpoint apply reports now include manifest provenance context for audit linkage.
3. Validation increment:
   - benchmark/policy command tests now at `18 passed` (`tests/test_ollama_local_benchmark.py`).

## Continuation Update (Phase 4P)

Governance-enforcement follow-on is now implemented in `docs/plans/2026-02-15-phase4p-apply-checkpoint-provenance-enforcement.md`:

1. Apply-time provenance enforcement:
   - checkpoint apply now supports enforcement flags for required provenance context and fields.
2. Deterministic provenance validation:
   - `change_context` shape and required field checks now gate apply before mutation.
3. Validation increment:
   - benchmark/policy command tests now at `23 passed` (`tests/test_ollama_local_benchmark.py`).

## Continuation Update (Phase 4Q)

Governance-lineage follow-on is now implemented in `docs/plans/2026-02-15-phase4q-git-ancestry-enforcement.md`:

1. Git ancestry enforcement:
   - checkpoint apply now supports `--require-commit-reachable-from <ref>` lineage verification.
2. Deterministic git validation:
   - apply path now validates ref resolvability and commit ancestry before mutation.
3. Security hardening follow-up:
   - git ancestry helper now uses `rev-parse --verify -- <ref>` and evaluates ancestry against resolved ref SHA to prevent option-injection behavior from dash-prefixed refs.
4. Validation increment:
   - benchmark/policy command tests now at `29 passed` (`tests/test_ollama_local_benchmark.py`).

## Continuation Update (Phase 4R)

MCP-compatibility follow-on is now implemented in `docs/plans/2026-02-15-phase4r-mcp-2025-11-25-compatibility.md`:

1. MCP 2025-11-25 capability alignment:
   - wrapper initialize now advertises `tasks.list` support and tracks client elicitation capabilities.
2. Elicitation default semantics:
   - empty client `capabilities.elicitation` now defaults to form-mode support for compatibility.
3. Deterministic tasks/list handling:
   - lifecycle + params validation added for `tasks/list`, with deterministic empty task result until async lifecycle is enabled.
4. Tool metadata hardening:
   - each tool now returns `execution.taskSupport` plus richer annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`).
5. Review follow-up correctness fix:
   - `idempotentHint` classification now reflects true idempotent tool semantics instead of read-only-only proxying.
6. Validation increment:
   - protocol tests now at `36 passed` (`tests/test_mcp_wrapper_protocol.py`).

## Continuation Update (Phase 4S)

Task-lifecycle follow-on is now implemented in `docs/plans/2026-02-15-phase4s-mcp-task-lifecycle-baseline.md`:

1. MCP task lifecycle request baseline:
   - wrapper now handles `tasks/get`, `tasks/result`, and `tasks/cancel` with schema-aligned `taskId` validation.
2. Capability signaling alignment:
   - initialize now advertises `tasks.cancel` alongside `tasks.list`.
3. Deterministic lifecycle behavior:
   - unknown/invalid task IDs and non-terminal/terminal-state misuse now return explicit deterministic errors.
4. Validation increment:
   - protocol tests now at `43 passed` (`tests/test_mcp_wrapper_protocol.py`).
