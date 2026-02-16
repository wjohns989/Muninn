# Phase 5A: User Profile + Standalone Foundation

Date: 2026-02-15  
Status: Implemented (baseline complete + continuation hardening/diagnostics/task-result compatibility + Huginn browser branding + pre-serialization compaction + diagnostics gate/hygiene integration + incident replay automation + PR/release replay gate wiring + release-profile replay strictness/provenance applied)

## Objective

Deliver the first production slice of Phase 5 improvements focused on:

1. Editable user profile/global context as a first-class system primitive.
2. Stronger chronology/hierarchy contextualization for legacy ingestion.
3. Standalone executable path so Muninn can run browser-first without assistant mediation.

## Implemented

### 1) Editable User Profile/Global Context

- Added persistent `user_profiles` table in SQLite metadata store.
- Added memory-core APIs:
  - `set_user_profile(profile, user_id, merge, source)`
  - `get_user_profile(user_id)`
- Added REST APIs:
  - `POST /profile/user/set`
  - `GET /profile/user/get`
- Added MCP tools:
  - `set_user_profile`
  - `get_user_profile`
- Added SDK parity (sync + async):
  - `set_user_profile(...)`
  - `get_user_profile(...)`
- Added browser UI profile editor:
  - load profile,
  - save/merge profile patch,
  - replace profile.

### 2) Legacy Chronology + Hierarchy Context Enrichment

- Legacy discovery now emits:
  - `parent_path`
  - `path_depth`
  - `relative_path_hint`
  - `modified_at_epoch`
  - `modified_at_iso`
- Legacy ingestion now propagates these fields into chunk metadata:
  - `legacy_source_parent_path`
  - `legacy_source_path_depth`
  - `legacy_source_relative_path`
  - `legacy_source_modified_at_epoch`
  - `legacy_source_modified_at_iso`
  - `legacy_contextualization_mode=chronological_hierarchy`
- Browser legacy table now displays modified timestamp and relative path.
- Added one-click operator flow: auto discover + import parser-supported legacy sources.

### 3) Standalone Executable Foundation

- Added standalone launcher entrypoint: `muninn_standalone.py`
  - starts server/UI directly,
  - optional browser auto-open,
  - host/port/log-level controls.
- Added packaging helper: `scripts/build_standalone.py`
  - deterministic PyInstaller command generation,
  - dashboard asset bundling via `--add-data`,
  - optional `--onefile` and `--windowed`.

### 4) Transport + UI Surface Hardening Continuation

- `mcp_wrapper.py` now enforces bounded MCP tool-response payload size via:
  - `MUNINN_MCP_TOOL_RESPONSE_MAX_CHARS` (default `12000`, floor `256`),
  - deterministic truncation trailer metadata in text payloads.
- Search text payloads now use the same bounded response limiter path.
- Public MCP error responses are now sanitized by error class:
  - retain actionable validation errors (`ValueError`),
  - redact connection/timeout/internal details to stable operator-safe messages.
- User-profile REST endpoints now return generic 500-detail strings while logging full exception traces (`exc_info=True`) for operator diagnostics.
- `dashboard.html` XSS surface reduced:
  - replaced dynamic `innerHTML` JSON rendering with DOM-safe `textContent` rendering,
  - replaced legacy discovery table string-HTML rendering with explicit DOM node creation.
- Browser API error handling now summarizes/redacts potentially sensitive backend detail strings before display.

### 5) Long-Tool Timeout Mitigation Continuation (Phase 5A.1)

- MCP `tools/call` dispatch now supports automatic task-mode deferral for configured long-running tools when callers omit `params.task`.
- Default long-tool auto-deferral set:
  - `ingest_sources`
  - `ingest_legacy_sources`
  - `discover_legacy_sources`
- Runtime controls added:
  - `MUNINN_MCP_AUTO_TASK_FOR_LONG_TOOLS` (default `1`)
  - `MUNINN_MCP_AUTO_TASK_TOOL_NAMES`
  - `MUNINN_MCP_AUTO_TASK_REQUIRE_CLIENT_CAP` (default `0`)
- Detailed tranche note: `docs/plans/2026-02-15-phase5a1-mcp-long-tool-auto-task-deferral.md`.

### 6) Transport Closure Campaign Automation (Phase 5A.2)

- Added deterministic closure utility: `python -m eval.mcp_transport_closure`.
- Utility now orchestrates repeated soak runs and emits one closure verdict artifact with per-criterion booleans.
- Criteria mapped into machine-evaluable outputs:
  - streak target,
  - no-regression observation window,
  - p95 compliance ratio,
  - unresolved regression/defect/failure classification inputs.
- Detailed tranche note: `docs/plans/2026-02-15-phase5a2-mcp-transport-closure-campaign-automation.md`.

### 7) Tool-Call Telemetry Hardening (Phase 5A.3)

- `mcp_wrapper.py` now tracks per-tool-call transport telemetry:
  - elapsed wall time (`elapsed_ms`),
  - response message count,
  - total/max response bytes sent over stdio,
  - initial and remaining deadline budget snapshots.
- Added configurable near-timeout warning threshold:
  - `MUNINN_MCP_TOOL_CALL_WARN_MS` (default `90000`).
- Telemetry is emitted for both direct and task-backed tool execution paths, improving diagnosis for intermittent host transport closures.
- Detailed tranche note: `docs/plans/2026-02-15-phase5a3-mcp-tool-call-telemetry-hardening.md`.

### 8) Host-Safe `tasks/result` Wait Budget Hardening (Phase 5A.4)

- `tasks/result` wait path now enforces a configurable maximum blocking budget:
  - `MUNINN_MCP_TASK_RESULT_MAX_WAIT_SEC`
  - defaults to host-safe budget (`MUNINN_MCP_HOST_TOOLS_CALL_TIMEOUT_SEC - MUNINN_MCP_TOOL_CALL_DEADLINE_MARGIN_SEC`).
- When non-terminal task wait exceeds budget, wrapper now returns deterministic retryable JSON-RPC error:
  - code `-32002`
  - message instructs caller to continue polling `tasks/get` and retry `tasks/result`.
- This prevents indefinite blocking inside `tasks/result` from overrunning host-side timeout windows and triggering transport teardown.
- Detailed tranche note: `docs/plans/2026-02-15-phase5a4-mcp-task-result-host-safe-wait-budget.md`.

### 9) `tasks/result` Compatibility Mode (Phase 5A.5)

- Added explicit compatibility policy control:
  - `MUNINN_MCP_TASK_RESULT_MODE` = `auto|blocking|immediate_retry` (default `auto`).
- Added auto-mode client profile control:
  - `MUNINN_MCP_TASK_RESULT_AUTO_RETRY_CLIENTS` (comma-delimited match list).
  - In `auto`, matching client profiles use immediate-retry semantics for non-terminal task states.
- Added per-request override in `tasks/result`:
  - `params.wait=true` forces blocking (bounded by Phase 5A.4 wait budget),
  - `params.wait=false` forces immediate-retry.
- Added strict params validation:
  - non-boolean `params.wait` returns `-32602`.
- Detailed tranche note: `docs/plans/2026-02-15-phase5a5-mcp-task-result-compatibility-mode.md`.

### 10) Closure Telemetry + Huginn Browser Branding (Phase 5A.6)

- Closure telemetry enrichment shipped in eval pipeline:
  - soak now captures compatibility policy config (`task_result_mode`, `task_result_auto_retry_clients`),
  - closure now emits error-code totals and compatibility mode/profile distributions.
- Standalone/browser UX branding updated for practicality:
  - browser control center presented as **Huginn** (standalone mode),
  - explicit mode chip added in dashboard header,
  - README/launcher/build helper wording now distinguishes:
    - Huginn = browser-first standalone surface,
    - Muninn = MCP-attached assistant surface.
- Detailed tranche note: `docs/plans/2026-02-15-phase5a6-closure-telemetry-huginn-branding.md`.

### 11) Deterministic Non-Terminal `tasks/result` Probe Hardening (Phase 5A.7)

- Soak runner now supports explicit non-terminal probe path:
  - `--probe-nonterminal-task-result`
  - `--task-worker-start-delay-ms`
- Wrapper task-worker start-delay control added for deterministic probe timing:
  - `MUNINN_MCP_TASK_WORKER_START_DELAY_MS`
- Closure campaign now forwards probe controls and enforces probe criterion:
  - `nonterminal_task_result_probe_met`
- Closure telemetry now tracks non-terminal probe coverage/quality:
  - enabled count, success count, failure count, success ratio.
- Detailed tranche note: `docs/plans/2026-02-15-phase5a7-task-result-nonterminal-probe.md`.

### 12) Tool-Response Pre-Serialization Compaction (Phase 5A.8)

- Wrapper now compacts tool result payloads before JSON serialization to prevent large pre-truncation formatting spikes.
- New operator controls:
  - `MUNINN_MCP_TOOL_RESPONSE_PREVIEW_MAX_ITEMS`
  - `MUNINN_MCP_TOOL_RESPONSE_PREVIEW_MAX_DEPTH`
  - `MUNINN_MCP_TOOL_RESPONSE_PREVIEW_MAX_STRING_CHARS`
- Existing post-serialization output cap (`MUNINN_MCP_TOOL_RESPONSE_MAX_CHARS`) remains in place as final transport guardrail.
- Detailed tranche note: `docs/plans/2026-02-16-phase5a8-tool-response-pre-serialization-compaction.md`.

### 13) Transport Diagnostics Bundle Utility (Phase 5A.9)

- Added deterministic diagnostics utility:
  - `python -m eval.mcp_transport_diagnostics`
- Utility now composes:
  - wrapper incident counts (`transport_closed`, deadline exhaustion, startup-budget skips),
  - per-tool latency/size telemetry summaries,
  - near-timeout events,
  - recent soak/closure report summaries,
  - blocker-signal heuristics for wrapper-vs-host attribution.
- Detailed tranche note: `docs/plans/2026-02-16-phase5a9-transport-diagnostics-bundle.md`.

### 14) Transport Diagnostics Gate + Hygiene Integration (Phase 5A.10)

- `eval.mcp_transport_diagnostics` now supports threshold-based gate enforcement:
  - `--max-transport-closed-count`
  - `--max-deadline-exhaustion-count`
  - `--max-near-timeout-count`
  - `--enforce-gate`
- Diagnostics report output now includes explicit gate verdict fields:
  - `results.gate.passed`
  - `results.gate.violations`
- `eval.phase_hygiene` now supports transport diagnostics policy wiring:
  - `--transport-diagnostics-command`
  - `--fail-on-transport-diagnostics`
  - `--max-transport-closed-incidents`
  - `--max-transport-deadline-exhaustion-incidents`
  - `--max-transport-near-timeout-incidents`
- Hygiene artifacts now include diagnostics command/return-code/summary fields for auditable release gating.
- Detailed tranche note: `docs/plans/2026-02-16-phase5a10-transport-diagnostics-hygiene-gating.md`.

### 15) Transport Incident Replay Automation (Phase 5A.11)

- Added deterministic replay utility:
  - `python -m eval.mcp_transport_incident_replay`
- Replay utility now scans wrapper logs for configured transport-incident signatures and triggers diagnostics capture only when incident thresholds are met (or always-run mode is enabled).
- Replay artifacts now include:
  - incident signature totals and per-pattern counts,
  - sampled matched lines with timestamps,
  - trigger decision,
  - diagnostics command execution metadata and parsed payload,
  - resolved diagnostics artifact path when available.
- Replay utility now supports deterministic CI/release failure behavior on diagnostics command errors:
  - `--fail-on-diagnostics-error` (enabled by default).
- Detailed tranche note: `docs/plans/2026-02-16-phase5a11-transport-incident-replay-automation.md`.

### 16) PR/Release Replay Gate Wiring (Phase 5A.12)

- Added CI workflow wiring:
  - `.github/workflows/transport-incident-replay-gate.yml`
- Workflow now executes replay on:
  - pull requests targeting `main`,
  - pushes to `main`,
  - manual dispatch (`workflow_dispatch`) with host-log controls.
- Workflow now uploads replay/diagnostics artifacts and writes compact replay summaries in check output.
- Replay utility strict-mode guardrail added:
  - `--require-log-path-exists` for environments where missing wrapper logs must fail checks.
- Detailed tranche note: `docs/plans/2026-02-16-phase5a12-pr-release-replay-gate-wiring.md`.

### 17) Release-Profile Replay Strictness + Log Provenance (Phase 5A.13)

- Replay gate workflow now supports profile modes:
  - `pr_safe`
  - `release_host_captured`
- Workflow now runs on release publication boundaries:
  - `release` trigger (`published`).
- Release-host profile now enforces strict replay posture:
  - `--require-log-path-exists`
  - `--include-log-sha256`
- Replay reports and check summaries now include log provenance fields:
  - log existence,
  - log size,
  - log modified timestamp,
  - optional SHA-256 digest.
- Detailed tranche note: `docs/plans/2026-02-16-phase5a13-release-profile-replay-strictness-provenance.md`.

## Verification

- Targeted + integration suite for this tranche:
  - `116 passed` in:
    - `tests/test_ingestion_discovery.py`
    - `tests/test_memory_ingestion.py`
    - `tests/test_memory_user_profile.py`
    - `tests/test_sqlite_goal_handoff.py`
    - `tests/test_sdk_client.py`
    - `tests/test_mcp_wrapper_protocol.py`
    - `tests/test_standalone_entrypoint.py`
    - `tests/test_build_standalone.py`
- Continuation hardening verification:
  - `79 passed` in:
    - `tests/test_mcp_wrapper_protocol.py`
    - `tests/test_mcp_transport_soak.py`
  - `5 passed` in:
    - `tests/test_memory_user_profile.py`
    - `tests/test_ingestion_discovery.py`
  - `104 passed` in:
    - `tests/test_memory_user_profile.py`
    - `tests/test_sdk_client.py`
    - `tests/test_mcp_wrapper_protocol.py`
    - `tests/test_mcp_transport_soak.py`
  - `82 passed` in:
    - `tests/test_mcp_wrapper_protocol.py`
    - `tests/test_mcp_transport_soak.py`
  - transport-closure automation verification:
    - `python -m py_compile eval/mcp_transport_closure.py`
    - `86 passed` (`tests/test_mcp_transport_closure.py`, `tests/test_mcp_transport_soak.py`, `tests/test_mcp_wrapper_protocol.py`)
    - `5 passed` (`tests/test_memory_user_profile.py`, `tests/test_ingestion_discovery.py`)
    - full closure campaign evidence:
      - `closure_ready=true`, streak `30`, window `30/30`, p95 ratio `1.0`
      - artifact: `eval/reports/mcp_transport/mcp_transport_closure_20260215_213858.json`
  - tool-call telemetry hardening verification:
    - `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
    - `88 passed` (`tests/test_mcp_wrapper_protocol.py`, `tests/test_mcp_transport_soak.py`, `tests/test_mcp_transport_closure.py`)
  - host-safe `tasks/result` wait-budget hardening verification:
    - `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
    - `92 passed` (`tests/test_mcp_wrapper_protocol.py`, `tests/test_mcp_transport_soak.py`, `tests/test_mcp_transport_closure.py`)
    - `5 passed` (`tests/test_memory_user_profile.py`, `tests/test_ingestion_discovery.py`)
    - soak regression check pass: `eval/reports/mcp_transport/mcp_transport_soak_20260215_220359.json`
    - closure mini-campaign pass: `eval/reports/mcp_transport/mcp_transport_closure_20260215_220419.json` (`closure_ready=true`, streak `5`, p95 ratio `1.0`)
  - `tasks/result` compatibility-mode hardening verification:
    - `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
    - `98 passed` (`tests/test_mcp_wrapper_protocol.py`, `tests/test_mcp_transport_soak.py`, `tests/test_mcp_transport_closure.py`)
    - `5 passed` (`tests/test_memory_user_profile.py`, `tests/test_ingestion_discovery.py`)
    - soak regression check pass: `eval/reports/mcp_transport/mcp_transport_soak_20260215_221650.json`
    - closure mini-campaign pass: `eval/reports/mcp_transport/mcp_transport_closure_20260215_221709.json` (`closure_ready=true`, streak `5`, p95 ratio `1.0`)
  - closure telemetry + Huginn branding verification:
    - `python -m py_compile eval/mcp_transport_soak.py eval/mcp_transport_closure.py mcp_wrapper.py muninn_standalone.py scripts/build_standalone.py server.py`
    - `111 passed` (`tests/test_mcp_transport_closure.py`, `tests/test_mcp_transport_soak.py`, `tests/test_mcp_wrapper_protocol.py`, `tests/test_standalone_entrypoint.py`, `tests/test_build_standalone.py`, `tests/test_memory_user_profile.py`, `tests/test_ingestion_discovery.py`)
    - soak regression check pass: `eval/reports/mcp_transport/mcp_transport_soak_20260215_224206.json`
    - closure mini-campaign pass with telemetry: `eval/reports/mcp_transport/mcp_transport_closure_20260215_224225.json` (`closure_ready=true`, streak `5`, p95 ratio `1.0`)
  - non-terminal `tasks/result` probe hardening verification:
    - `python -m py_compile eval/mcp_transport_soak.py eval/mcp_transport_closure.py mcp_wrapper.py tests/test_mcp_transport_soak.py tests/test_mcp_transport_closure.py tests/test_mcp_wrapper_protocol.py`
    - `105 passed` (`tests/test_mcp_transport_soak.py`, `tests/test_mcp_transport_closure.py`, `tests/test_mcp_wrapper_protocol.py`)
    - soak probe pass: `eval/reports/mcp_transport/mcp_transport_soak_20260215_235614.json` (observed `-32002`)
    - closure probe-criteria pass: `eval/reports/mcp_transport/mcp_transport_closure_20260215_235635.json` (`closure_ready=true`, `nonterminal_task_result_probe_met=true`)
  - tool-response pre-serialization compaction verification:
    - `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
    - `108 passed` (`tests/test_mcp_wrapper_protocol.py`, `tests/test_mcp_transport_soak.py`, `tests/test_mcp_transport_closure.py`)
  - transport diagnostics bundle verification:
    - `python -m py_compile eval/mcp_transport_diagnostics.py tests/test_mcp_transport_diagnostics.py`
    - `110 passed` (`tests/test_mcp_transport_diagnostics.py`, `tests/test_mcp_wrapper_protocol.py`, `tests/test_mcp_transport_soak.py`, `tests/test_mcp_transport_closure.py`)
    - live diagnostics artifact: `eval/reports/mcp_transport/mcp_transport_diagnostics_20260216_001515.json`
  - transport diagnostics gate + hygiene integration verification:
    - `python -m py_compile eval/mcp_transport_diagnostics.py eval/phase_hygiene.py tests/test_mcp_transport_diagnostics.py tests/test_phase_hygiene.py`
    - `119 passed` (`tests/test_mcp_transport_diagnostics.py`, `tests/test_phase_hygiene.py`, `tests/test_mcp_wrapper_protocol.py`, `tests/test_mcp_transport_soak.py`, `tests/test_mcp_transport_closure.py`)
    - live diagnostics gate artifact: `eval/reports/mcp_transport/mcp_transport_diagnostics_20260216_005047.json` (`results.gate.passed=true`)
  - transport incident replay automation verification:
    - `python -m py_compile eval/mcp_transport_incident_replay.py tests/test_mcp_transport_incident_replay.py`
    - `3 passed` (`tests/test_mcp_transport_incident_replay.py`)
    - live replay artifact: `eval/reports/mcp_transport/mcp_transport_incident_replay_20260216_010414.json` (`results.triggered=false`)
  - PR/release replay gate wiring verification:
    - `python -m py_compile eval/mcp_transport_diagnostics.py eval/phase_hygiene.py eval/mcp_transport_incident_replay.py tests/test_mcp_transport_diagnostics.py tests/test_phase_hygiene.py tests/test_mcp_transport_incident_replay.py`
    - `123 passed` (`tests/test_mcp_transport_diagnostics.py`, `tests/test_phase_hygiene.py`, `tests/test_mcp_transport_incident_replay.py`, `tests/test_mcp_wrapper_protocol.py`, `tests/test_mcp_transport_soak.py`, `tests/test_mcp_transport_closure.py`)
    - workflow wiring present: `.github/workflows/transport-incident-replay-gate.yml`
  - release-profile replay strictness + provenance verification:
    - `python -m py_compile eval/mcp_transport_diagnostics.py eval/phase_hygiene.py eval/mcp_transport_incident_replay.py tests/test_mcp_transport_diagnostics.py tests/test_phase_hygiene.py tests/test_mcp_transport_incident_replay.py`
    - `124 passed` (`tests/test_mcp_transport_diagnostics.py`, `tests/test_phase_hygiene.py`, `tests/test_mcp_transport_incident_replay.py`, `tests/test_mcp_wrapper_protocol.py`, `tests/test_mcp_transport_soak.py`, `tests/test_mcp_transport_closure.py`)
    - release-capable workflow profile wiring present: `.github/workflows/transport-incident-replay-gate.yml`
    - live strict-profile replay artifact: `eval/reports/mcp_transport/mcp_transport_incident_replay_20260216_012731.json` (`log_file.sha256` populated, `results.triggered=false`)
- Full-suite checkpoint:
  - `520 passed, 2 skipped, 1 warning`.

## Research and Feasibility Findings

### Ollama + Skill/Agent Markdown Feasibility

Feasible, with constraints:

- Ollama supports prompt/system behavior control via Modelfile primitives (`SYSTEM`, `TEMPLATE`) and runtime generation options.
- Ollama supports structured outputs and tool-calling paths in modern chat/generate APIs.
- Practical pattern for Muninn:
  - compile selected profile+skill Markdown into structured policy blocks,
  - inject policy as a deterministic system layer for local assistant sessions,
  - keep source profile/skill artifacts as auditable context in memory metadata.

### Standalone Packaging Feasibility

Feasible and low risk for baseline:

- PyInstaller supports both one-directory and one-file builds.
- Data files (like `dashboard.html`) require explicit inclusion.
- Recommended default for reliability: `onedir`; use `onefile` as optional advanced mode.

## Huginn Concept: Practicality Assessment

Concept: Add a deterministic planning/recommendation layer ("Huginn") on top of Muninn memory.

Assessment:

- High practical value if scope remains bounded:
  - inputs: user profile + project goal + recent retrieval traces + task metadata
  - outputs: concise ranked next-action recommendations with confidence and provenance
- Keep Huginn stateless per request initially; avoid autonomous writeback loops in v1.
- Gate recommendations with explicit confidence + governance thresholds before automated policy actions.

## Next-Phase Proposed Work (Phase 5B)

1. Add profile schema validation profiles (`strict`, `flexible`) with migration support.
2. Add profile-aware retrieval priors and rerank hooks.
3. Add Huginn v1 recommendation endpoint + MCP tool (`recommend_next_actions`).
4. Add skill-markdown compiler + Ollama runtime prompt adapter.
5. Add standalone packaging CI artifacts and integrity manifest checks.
6. Add oversized-result artifact mode for MCP tools (task pointer + fetch path) to avoid large stdio payload dependence on host transport.

## Sources

- Ollama docs (Modelfile): https://github.com/ollama/ollama/blob/main/docs/modelfile.md  
- Ollama docs (API): https://github.com/ollama/ollama/blob/main/docs/api.md  
- Ollama blog (structured outputs): https://ollama.com/blog/structured-outputs  
- PyInstaller usage docs: https://pyinstaller.org/en/stable/usage.html  
- PyInstaller runtime information: https://pyinstaller.org/en/stable/runtime-information.html  
- MemoryAgentBench paper: https://arxiv.org/abs/2507.05257  
- EMemBench trend reference: https://www.emergentmind.com/articles/2509.09957  
