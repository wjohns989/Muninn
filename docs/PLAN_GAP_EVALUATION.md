# Muninn v3.3 SOTA+ Plan Gap Evaluation (Codebase Snapshot)

Date: 2026-02-15
Evaluator: Codex

## TL;DR

- **Phase 1 correctness blockers are now fixed in-code** (Instructor wiring, Docker path behavior, trace score fidelity, adaptive score entropy, and version consistency).
- **ROI continuity tranche is now implemented** (Goal Compass + idempotent handoff bundles + MCP/API wiring).
- **Eval gate is now enforceable in CI** (latency + baseline regression checks).
- **Phase 2 core loop is now present** (conflict/dedup/weight-adapter + persisted feedback calibration path implemented).
- **Counterfactual calibration path is now available** (SNIPS-style estimator with rank/sampling propensity support and safeguards).
- **Competency-sliced eval reporting is now available** (optional per-track metrics in `eval/`).
- **Preset-driven release gate policy is now available** (track coverage gates + per-track regression checks + auditable gate config).
- **Paired significance/effect-size eval is now available** (bootstrap CI + permutation p-values + significant-regression gate).
- **Multiple-comparison correction is now available** (Bonferroni/Holm/BH with configurable test-family scope).
- **MCP conformance checks are now hardened** (method-not-found handling, lifecycle ordering, and param-shape validation).
- **Canonical benchmark artifact discipline is now available** (committed bundle + checksum manifest + reproducibility verifier).
- **Canonical artifact coverage is now multi-bundle** (baseline + robustness stress slice) with aggregate verification (`verify --all`).
- **OTel operational enablement is now documented** (runbook + collector config + privacy controls).
- **Legacy memory/chat migration flow is now shipped**: discovery + selection-based import across assistant logs and MCP memory stores, including chat-context parsing for JSONL and sqlite-backed stores.
- **Browser control center is now shipped**: root-served UI supports practical ingestion/reingestion/search/consolidation operations without CLI coupling.
- **Open PR review issues are now remediated in code**: ingestion allow-list enforcement, runtime chunking bounds, legacy root/path validation, SDK URL-segment encoding, duplicate-safe eval metrics, and `/ingest` HTTPException passthrough.
- **Phase 3 is now functionally complete at core package level**: Python SDK, multi-source ingestion, legacy migration, browser control center, and memory chains are now shipped.
- **Phase 4A is now started with production code**: browser control center preferences are now persistent (auto-save/save/reset) and model-profile tags are attached to ingestion operations.
- **Phase 4B baseline is now implemented in core extraction path**: profile-based Instructor route selection (`low_latency`, `balanced`, `high_reasoning`) is wired, with deterministic xLAM/Ollama fallback ordering and config/env controls.
- **Phase 4C startup/session adaptation baseline is now implemented**: MCP initialize now performs dependency readiness checks (Muninn + Ollama), auto-starts when possible, emits explicit startup prompts when not, and honors assistant-session profile overrides via `MUNINN_OPERATOR_MODEL_PROFILE`.
- **Phase 4D VRAM-aware policy baseline is now implemented**: extraction defaults are now budget-aware via `MUNINN_VRAM_BUDGET_GB`, with 16GB-safe high-reasoning defaults (14B-class) and 30B/32B limited to explicit high-budget tiers.
- **Phase 4E helper-first profile scheduling baseline is now implemented**: runtime/add/update defaults now stay on low-latency profile while ingestion/legacy-ingestion can use independently configured profiles (`MUNINN_RUNTIME_MODEL_PROFILE`, `MUNINN_INGESTION_MODEL_PROFILE`, `MUNINN_LEGACY_INGESTION_MODEL_PROFILE`) plus operation-specific MCP env overrides.
- **Phase 4F runtime profile-control tranche is now implemented**: profile policy can now be read/updated at runtime through memory core + REST (`/profiles/model`) + MCP tools (`get_model_profiles`, `set_model_profiles`) + SDK sync/async parity.
- **Phase 4G profile-policy audit visibility baseline is now implemented**: runtime profile mutations are now persisted as audit events and exposed through memory core + REST (`/profiles/model/events`) + MCP (`get_model_profile_events`) + SDK sync/async.
- **Phase 4H local model-matrix benchmarking baseline is now implemented**: versioned Ollama model matrix + prompt pack + sync/benchmark CLI are now shipped for reproducible local model selection under 16GB-class helper-first workflows.
- **Phase 4I model ability/resource benchmarking baseline is now implemented**: benchmark reports now include rubric-based ability scoring + ability-per-resource metrics, and a new legacy-ingestion benchmark mode can generate deterministic test cases from old project roots.
- **Phase 4J profile-promotion gate framework is now implemented in-branch**: profile gate policy file + `profile-gate` evaluator command now convert live/legacy benchmark evidence into deterministic pass/fail + recommendation decisions per profile tier.
- **Phase 4K phase-boundary hygiene gate baseline is now implemented**: `eval.phase_hygiene` now enforces one-open-PR policy, executes test commands with shell-safe tokenization, parses pytest JUnit summaries, and checks review/check + skipped-test/warning budgets with machine-readable reports.
- **Phase 4L development-cycle benchmark orchestration baseline is now implemented**: `dev-cycle` now runs live benchmark + legacy benchmark + profile gate in one operator-triggered command and emits role-based model recommendations for runtime/balanced/high-reasoning usage.
- **Phase 4M benchmark-to-policy bind baseline is now implemented**: `dev-cycle --apply-policy` now supports controlled runtime profile-policy mutation with pre-apply checkpoints, and `rollback-policy` restores previous defaults deterministically.
- **MCP transport framing compatibility fix is now implemented**: `mcp_wrapper` now accepts both newline-delimited JSON and `Content-Length` framed JSON-RPC payloads to prevent client transport disconnects caused by framing mismatch.
- **MCP startup + tray operational hardening is now implemented**: wrapper launch now triggers autostart bootstrap for Ollama/server when enabled, and Windows tray now exposes direct Browser UI + MCP health check + wrapper diagnostics actions.
- **Phase 4N policy-approval manifest baseline is now implemented**: `approval-manifest` now records explicit checkpoint approval/rejection with SHA-256 binding, and `apply-checkpoint` now enforces approved decision + integrity checks before profile-policy apply.
- **Phase 4O approval provenance-context baseline is now implemented**: approval manifests now capture optional PR/commit/branch provenance with validation + git fallback, and checkpoint apply reports now propagate the provenance context.
- **Phase 4P apply-checkpoint provenance enforcement baseline is now implemented**: checkpoint apply now supports required-provenance enforcement flags so policy mutation can be blocked when manifest provenance fields are missing.
- **Phase 4Q git-ancestry enforcement baseline is now implemented**: checkpoint apply now supports commit lineage verification (`--require-commit-reachable-from`) to block mutations when commit ancestry does not match required branch/ref.
- **Phase 4R MCP 2025-11-25 compatibility baseline is now implemented**: wrapper now advertises `tasks.list`, applies elicitation empty-object form defaults, supports deterministic `tasks/list`, and returns explicit task-support/tool-annotation metadata contracts.
- **Phase 4S MCP task lifecycle baseline is now implemented**: wrapper now handles `tasks/get`, `tasks/result`, and `tasks/cancel` with schema-aligned `taskId` validation and deterministic lifecycle error behavior.
- **Phase 4T task-augmented tools/call baseline is now implemented**: wrapper now supports task-backed `tools/call` execution with status notifications plus TTL/retention/pagination governance for task state.
- **Phase 4U blocking-result dispatch compliance baseline is now implemented**: `tasks/result` now remains lifecycle-blocking while wrapper dispatch routes blocking methods to background workers with stdout write-locking for concurrent transport safety.
- **Phase 4V task metadata/cursor compliance baseline is now implemented**: related-task metadata now uses `taskId`, task records now include `pollInterval`, and `tasks/list` now emits opaque cursor tokens.
- **Phase 4W MCP transport resilience baseline is now implemented**: malformed framed payloads are now recoverable, backend outages now use circuit-breaker fast-fail cooldown, dispatch queue saturation now returns explicit `-32001`, and broken-pipe writes are now transport-guarded.
- **Phase 4X transport soak + dispatch-policy baseline is now implemented**: deterministic MCP soak harness reports are now generated, `tools/call` background dispatch is now opt-in for transport determinism, and outage preflight start probes are now skipped when autostart is disabled or circuit is open.
- **Phase 4Y profile-governance telemetry + guardrail baseline is now implemented**: profile-gate now emits governance alerts with policy thresholds, governance enforcement is now available in gate/cycle commands, and policy apply can now require governance-clean gate output.
- **Packaging dependency mismatch is now corrected**: `pyproject.toml` now ships explicit optional extras for `conflict` and `sdk`, and `all` now includes those surfaces for reproducible installs.
- **Phase 4AA MCP tool-call timeout hardening baseline is now implemented**: wrapper tool calls now run under a bounded deadline budget (default `110s`), retry attempts now clamp request timeouts to remaining budget and abort deterministically on budget exhaustion, and `delete_memory` path segments are now URL-encoded in wrapper transport.
- **Phase 4AB startup-recovery budget gating baseline is now implemented**: preflight/retry startup recovery now checks remaining deadline budget and skips recovery when budget is below threshold (`MUNINN_MCP_STARTUP_RECOVERY_MIN_BUDGET_SEC`, default `28s`) to avoid timeout-window overruns.
- **Phase 4AC host-timeout-derived deadline budgeting baseline is now implemented**: when explicit deadline is not set, wrapper now derives tool-call budget from host timeout minus safety margin (`MUNINN_MCP_HOST_TOOLS_CALL_TIMEOUT_SEC` - `MUNINN_MCP_TOOL_CALL_DEADLINE_MARGIN_SEC`) with safe minimum clamping, improving cross-client timeout compatibility.
- **Phase 4AD explicit-deadline overrun guardrail baseline is now implemented**: explicit deadline values now default-clamp to host-safe budgets unless operator opt-out is set (`MUNINN_MCP_TOOL_CALL_DEADLINE_ALLOW_OVERRUN=1`), preventing misconfiguration-driven timeout-window overruns.
- **Phase 4AE guarded-dispatch fail-fast response baseline is now implemented**: guarded RPC dispatch now emits deterministic `-32603` replies for request IDs when unexpected dispatch exceptions occur, preventing silent hangs on background-dispatched request paths.
- **Restart artifact hygiene is now complete**: stale staged restart artifacts are cleared, obsolete `.bak` backup artifact removed, and unresolved conflict markers are no longer present in working docs.
- **SOTA+ quantitative decision framework is now documented**: final release verdict gates and benchmark normalization requirements are captured in `docs/plans/2026-02-15-sota-plus-quantitative-comparison-plan.md`.
- **Phase 4AF unified SOTA+ verdict baseline is now implemented**: `sota-verdict` now emits one deterministic release gate artifact with normalized eval/profile-gate/transport evidence and per-gate pass/fail outcomes.
- **Phase 4AG enhancement-first benchmark cadence baseline is now implemented**: `dev-cycle` now supports deferred benchmark mode (`--defer-benchmarks`) with reusable report freshness checks (`--max-reused-report-age-hours`) to keep active implementation tranches fast while preserving gate integrity.

## Status vs Plan

### Phase 1 (v3.1.0)

| Plan item | Status | Evidence |
|---|---|---|
| 1A Platform abstraction | **Mostly implemented** | `muninn/platform.py` exists with cross-platform dirs/process helpers. |
| 1A Docker support | **Implemented** | `Dockerfile` and `docker-compose.yml` exist. |
| 1B Instructor extraction | **Implemented + wired** | `MuninnMemory.initialize()` now passes `instructor_base_url/model/api_key` into `ExtractionPipeline`. |
| 1C Explainable recall traces | **Implemented + fidelity fixed** | Trace attribution now uses signal-native `raw_score` values, not rank proxy. |
| 1D Feature flags | **Implemented** | `muninn/core/feature_flags.py` present and used across retrieval/extraction/memory initialization. |
| 1.1 Version consistency | **Implemented** | Single source of truth in `muninn/version.py`; package/server/MCP versions aligned. |

### Phase 2 (v3.2.0)

| Plan item | Status | Evidence |
|---|---|---|
| 2A Conflict detection | **Implemented (feature-gated)** | `muninn/conflict/*` + memory integration present. |
| 2B Semantic dedup | **Implemented (feature-gated)** | `muninn/dedup/semantic_dedup.py` + memory add path integration present. |
| 2C Adaptive weights | **Implemented + improved** | `WeightAdapter` entropy now derives from native signal score distributions. |
| Retrieval eval gates | **Implemented (v1)** | `eval/run.py` now supports baseline-regression and p95 latency gating. |
| retrieval feedback persistence | **Implemented (feature-gated)** | `retrieval_feedback` table + API/MCP recording + adaptive multipliers are wired into retrieval path. |
| Counterfactual feedback estimator | **Implemented (feature-gated)** | SQLite multiplier computation supports `weighted_mean` and `snips`, with propensity clipping + effective sample checks; API/MCP accept optional `rank` and `sampling_prob`. |
| Preset-based track gate policy | **Implemented** | `eval/presets.py` + `eval/run.py` now support named policy defaults, per-track regression checks, and required track case gates. |
| Paired significance/effect-size gate | **Implemented** | `eval/statistics.py` + `eval/run.py` support paired bootstrap CI, permutation p-values, `cohens_d`, and optional significant-regression fail gate. |
| Multiple-comparison correction policy | **Implemented** | `eval/statistics.py` + `eval/run.py` now support `none`/`bonferroni`/`holm`/`bh` correction and `all`/`by_track` family scoping with adjusted p-value gate decisions. |
| Canonical artifact verifier | **Implemented** | `eval/artifacts/vibecoder_memoryagentbench_v1/*` + `eval/artifacts.py` provide checksum and reproducibility verification. |

### Phase 3 (v3.3.0)

| Plan item | Status | Evidence |
|---|---|---|
| 3A Memory chains | **Implemented (feature-gated)** | `muninn/chains/*` added; graph `PRECEDES/CAUSES` edges + memory add/update linking + hybrid chain signal are wired. |
| 3B Multi-source ingestion | **Implemented + expanded** | `muninn/ingestion/` now provides fail-open parser pipeline with provenance metadata, chat-context extraction for `.jsonl/.ndjson`, sqlite-backed source parsing (`.vscdb/.db/.sqlite*`), runtime guardrails (allow-list roots + chunk/file bounds), REST (`/ingest`, `/ingest/legacy/discover`, `/ingest/legacy/import`), MCP (`ingest_sources`, `discover_legacy_sources`, `ingest_legacy_sources`), and SDK parity methods. |
| 3C Python SDK | **Implemented** | `muninn/sdk/` now ships sync+async clients with typed errors; top-level exports include `Memory` and `AsyncMemory`. |
| 3D Browser control center | **Implemented** | `dashboard.html` rebuilt and served at `/` by `server.py`; includes legacy discovery/import selection, project-folder contextual ingestion with chronological ordering, and operational search/consolidation controls. |

## High-Impact Issues Discovered

1. **Graph retrieval argument mismatch (fixed):** `_graph_search` passed a string where graph store expects a list; now corrected with deterministic scoring.
2. **User-scope enforcement risk in retrieval (fixed):** hybrid retrieval now enforces user/namespace constraints in final record filtering.
3. **Legacy ingestion path traversal / arbitrary file-read surface (fixed):** user-provided roots and selected paths now validate against ingestion allow-list roots before discovery/import.
4. **Ingestion DoS vector through unconstrained chunk params (fixed):** runtime chunk/file limits now enforce bounded values and relation constraints (`overlap < chunk_size`, `min_chunk <= chunk_size`).
5. **Plan/dependency mismatch (fixed):** `pyproject.toml` now defines optional dependency groups for `conflict` and `sdk` and includes both in `all` install surface.
6. **Evaluation corpus breadth still incomplete (open):** gate mechanics and artifact coverage now include two bundles, but additional domain and noise/adversarial slices are still needed.
7. **Parser sandbox/process isolation still open (security hardening):** optional binary backends (`pdf/docx`) remain in-process and should be isolated for stricter threat models.
8. **Extraction/model policy partially open:** profile routing, UI profile persistence, session-level override wiring, operation-scoped runtime/ingestion profile defaults, runtime profile mutation API, mutation audit events, local model-matrix benchmarking harness, ability/resource benchmark scoring, controlled apply/rollback mutation flow, approval-gated checkpoint apply, PR/commit/branch provenance capture, apply-time provenance enforcement flags, git ancestry enforcement, and governance alert/guardrail controls are now implemented; remaining work is fully automated promotion scheduling/roll-forward policies for unattended operation.
9. **MCP Muninn transport reliability intermittency (partially mitigated, monitoring remains):** framing + parser resilience + queue/backoff + soak harness + tools/call deadline-budget controls + startup-recovery budget gating + host-timeout-derived budgeting + explicit-overrun guardrails + guarded-dispatch fail-fast replies are now in-code; remaining risk is host-side environment variability outside wrapper process control (monitor and tune `MUNINN_MCP_TOOL_CALL_DEADLINE_SEC`, `MUNINN_MCP_HOST_TOOLS_CALL_TIMEOUT_SEC`, `MUNINN_MCP_TOOL_CALL_DEADLINE_MARGIN_SEC`, `MUNINN_MCP_STARTUP_RECOVERY_MIN_BUDGET_SEC`, and `MUNINN_MCP_TOOL_CALL_DEADLINE_ALLOW_OVERRUN` as needed). Post-restart in-session signal: `get_model_profiles` and `add_memory` MCP calls succeeded, and latest framed transport soak run passed (`run_id=20260215_170548`), but continued runtime observation is still required before closing blocker status.
10. **Unified SOTA+ verdict automation (partially open):** one authoritative go/no-go verdict artifact is now wired (`sota-verdict`) with normalization hooks; remaining work is benchmark breadth adapters (LongMemEval/StructMemEval/Mem2Act + continuous-interaction slices), CI replay wiring, and signed promotion-manifest emission.
11. **CI cadence split (open):** deferred benchmark mode is now available for enhancement tranches, but CI still needs explicit fast-on-PR vs full-on-schedule/release workflow wiring.
12. **Policy-Aware Memory Governance (RL-Driven) (open):** Transition from static TTL/Rules to Reinforcement Learning (RL)-driven memory management. The system should autonomously learn when to Write, Delete, or Update memories based on retrieval reward signals, moving beyond predefined limits.
13. **E-mem Episodic Context Reconstruction (open):** Address "destructive de-contextualization." We need to enhance the existing `PRECEDES/CAUSES` graph edges to ensure logical integrity over extended temporal interactions.
14. **Cognitive Architecture (CoALA) Integration (open):** Introduce a modular decision-making loop bridging traditional ACT-R/SOAR cognitive structures with our memory components. This will help filter omissions and ground agent reasoning natively within the Muninn substrate.

## Validation Snapshot

- Full suite passes on `feature/sota-roadmap-outward` (2026-02-20): `1019 passed, 2 skipped, 4 warnings`.
- Crash-recovery verification completed: git integrity checks passed (`git fsck --full` with no corruption), and no open PR/comment backlog remained after restart.
- MCP protocol-focused tests: `12 passed` (`tests/test_mcp_wrapper_protocol.py`).
- Targeted changed-surface tests now pass:
  - `23 passed` (`eval_artifacts`, `eval_statistics`, `eval_presets`, `eval_run`, `eval_gates`, `eval_metrics`)
  - `21 passed` (`eval_statistics`, `eval_presets`, `eval_run`, `eval_gates`, `eval_metrics`)
  - `15 passed` (`eval_gates`, `eval_metrics`, `eval_run`)
  - `48 passed` (`sqlite_feedback`, `eval_metrics`, `mcp_wrapper_protocol`, `weight_adapter`, `eval_gates`)
  - `27 passed` (`memory_feedback`, `config`)
  - `32 passed` (`ingestion_parser`, `memory_ingestion`, `mcp_wrapper_protocol`, `sdk_client`)
  - `34 passed` (`ingestion_pipeline`, `memory_ingestion`, `mcp_wrapper_protocol`, `sdk_client`)
  - `83 passed` (`eval_metrics`, `sdk_client`, `ingestion_pipeline`, `ingestion_parser`, `ingestion_discovery`, `memory_ingestion`, `config`, `mcp_wrapper_protocol`)
  - `40 passed` (`memory_chains`, `hybrid_retriever`, `memory_update_path`, `config`, `memory_feedback`)
  - `40 passed` (`recall_trace`, `feature_flags`)
  - `20 passed` (`mcp_wrapper_protocol` startup/session-profile coverage)
  - `36 passed` (`config`, `extraction_pipeline` VRAM-policy coverage)
  - `69 passed` (`config`, `memory_ingestion`, `memory_update_path`, `mcp_wrapper_protocol`, `extraction_pipeline`)
  - `45 passed` (`memory_profiles`, `mcp_wrapper_protocol`, `sdk_client`)
  - `49 passed` (`memory_profiles`, `sqlite_profile_policy_events`, `mcp_wrapper_protocol`, `sdk_client`)
- Local model benchmark tooling smoke checks now pass:
  - `python -m eval.ollama_local_benchmark list`
  - `python -m eval.ollama_local_benchmark sync --dry-run`
- Phase 4I benchmark extensions now pass targeted checks:
  - `python -m py_compile eval/ollama_local_benchmark.py`
  - `8 passed` (`tests/test_ollama_local_benchmark.py`)
- Phase 4K hygiene-gate tranche now passes targeted checks:
  - `python -m py_compile eval/phase_hygiene.py tests/test_phase_hygiene.py`
  - `5 passed` (`tests/test_phase_hygiene.py`)
  - `14 passed` (`tests/test_ollama_local_benchmark.py`, `tests/test_phase_hygiene.py`)
- MCP transport framing compatibility tranche now passes targeted checks:
  - `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
  - `28 passed` (`tests/test_mcp_wrapper_protocol.py`)
- MCP startup + tray hardening tranche now passes targeted checks:
  - `python -m py_compile mcp_wrapper.py tray_app.py tests/test_mcp_wrapper_protocol.py`
  - `30 passed` (`tests/test_mcp_wrapper_protocol.py`)
- Phase 4M benchmark-policy apply/rollback tranche now passes targeted checks:
  - `python -m py_compile eval/ollama_local_benchmark.py tests/test_ollama_local_benchmark.py`
  - `11 passed` (`tests/test_ollama_local_benchmark.py`)
- Phase 4N policy-approval manifest tranche now passes targeted checks:
  - `python -m py_compile eval/ollama_local_benchmark.py tests/test_ollama_local_benchmark.py`
  - `16 passed` (`tests/test_ollama_local_benchmark.py`)
- Phase 4O approval provenance-context tranche now passes targeted checks:
  - `python -m py_compile eval/ollama_local_benchmark.py tests/test_ollama_local_benchmark.py`
  - `18 passed` (`tests/test_ollama_local_benchmark.py`)
- Phase 4P apply-checkpoint provenance enforcement tranche now passes targeted checks:
  - `python -m py_compile eval/ollama_local_benchmark.py tests/test_ollama_local_benchmark.py`
  - `23 passed` (`tests/test_ollama_local_benchmark.py`)
- Phase 4Q git-ancestry enforcement tranche now passes targeted checks:
  - `python -m py_compile eval/ollama_local_benchmark.py tests/test_ollama_local_benchmark.py`
  - `29 passed` (`tests/test_ollama_local_benchmark.py`)
  - security hardening now enforces `rev-parse --verify -- <ref>` + resolved-SHA ancestry checks for dash-prefixed ref safety.
- Phase 4R MCP 2025-11-25 compatibility tranche now passes targeted checks:
  - `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
  - `36 passed` (`tests/test_mcp_wrapper_protocol.py`)
  - `70 passed` (`tests/test_ollama_local_benchmark.py`, `tests/test_phase_hygiene.py`, `tests/test_mcp_wrapper_protocol.py`)
  - review follow-up corrected `idempotentHint` semantics for non-read-only idempotent tools.
- Phase 4S MCP task lifecycle tranche now passes targeted checks:
  - `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
  - `45 passed` (`tests/test_mcp_wrapper_protocol.py`)
  - `81 passed` (`tests/test_ollama_local_benchmark.py`, `tests/test_phase_hygiene.py`, `tests/test_mcp_wrapper_protocol.py`)
  - review follow-up now rejects terminal tasks without result payload and avoids reflecting raw unknown-task IDs in error strings.
  - workflow follow-up now hardens `eval.phase_hygiene` command decoding for mixed UTF-8/CP1252 output on Windows.
- Phase 4T task-augmented tools/call tranche now passes targeted checks:
  - `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
  - `49 passed` (`tests/test_mcp_wrapper_protocol.py`)
  - `85 passed` (`tests/test_ollama_local_benchmark.py`, `tests/test_phase_hygiene.py`, `tests/test_mcp_wrapper_protocol.py`)
  - `tasks/result` now blocks until terminal completion and includes related-task metadata for deterministic correlation.
  - task registry now enforces TTL purge + retention cap + cursor pagination semantics.
- Phase 4U blocking-result dispatch tranche now passes targeted checks:
  - `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
  - `52 passed` (`tests/test_mcp_wrapper_protocol.py`)
  - `88 passed` (`tests/test_ollama_local_benchmark.py`, `tests/test_phase_hygiene.py`, `tests/test_mcp_wrapper_protocol.py`)
  - blocking lifecycle semantics now run on background dispatch paths (`tasks/result`, `tools/call`) so wrapper channel remains responsive under concurrent requests.
  - review follow-up now bounds background dispatch with a thread pool and removes reflected exception text from guarded-dispatch logs.
- Phase 4V metadata/cursor compliance tranche now passes targeted checks:
  - `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
  - `52 passed` (`tests/test_mcp_wrapper_protocol.py`)
  - `88 passed` (`tests/test_ollama_local_benchmark.py`, `tests/test_phase_hygiene.py`, `tests/test_mcp_wrapper_protocol.py`)
  - related-task metadata now emits schema-aligned `taskId` key and `tasks/list` now uses opaque cursor tokens with backward-compatible numeric decode handling.
- Phase 4W MCP transport resilience tranche now passes targeted checks:
  - `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
  - `56 passed` (`tests/test_mcp_wrapper_protocol.py`)
  - `92 passed` (`tests/test_ollama_local_benchmark.py`, `tests/test_phase_hygiene.py`, `tests/test_mcp_wrapper_protocol.py`)
  - `python -m eval.phase_hygiene --max-open-prs 1 --pytest-command ""` -> PASS (`eval/reports/hygiene/phase_hygiene_20260215_064545.json`)
  - protocol tests now cover malformed framed payload recovery, circuit-open fast-fail behavior, dispatch queue saturation, and broken-pipe transport guarding.
- Phase 4X transport soak + dispatch-policy tranche now passes targeted checks:
  - `python -m py_compile eval/mcp_transport_soak.py tests/test_mcp_transport_soak.py mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
  - `98 passed` (`tests/test_mcp_transport_soak.py`, `tests/test_mcp_wrapper_protocol.py`, `tests/test_phase_hygiene.py`, `tests/test_ollama_local_benchmark.py`)
  - `python -m eval.mcp_transport_soak --iterations 6 --warmup-requests 1 --timeout-sec 12 --transport framed --server-url http://127.0.0.1:1 --failure-threshold 1 --cooldown-sec 30 --max-p95-ms 2500 --inject-malformed-frame` -> PASS (`eval/reports/mcp_transport/mcp_transport_soak_20260215_074136.json`)
  - `python -m eval.phase_hygiene --max-open-prs 1 --pytest-command ""` -> PASS (`eval/reports/hygiene/phase_hygiene_20260215_074404.json`)
  - `tools/call` background dispatch is now opt-in (`MUNINN_MCP_BACKGROUND_TOOLS_CALL=1`) while `tasks/result` remains background-dispatched by default.
- Phase 4Y profile-governance telemetry + guardrail tranche now passes targeted checks:
  - `python -m py_compile eval/ollama_local_benchmark.py tests/test_ollama_local_benchmark.py`
  - `30 passed` (`tests/test_ollama_local_benchmark.py`)
  - `profile-gate` now emits deterministic governance alerts and supports `--enforce-governance`.
  - `dev-cycle` now supports `--enforce-governance` and `--require-governance-clean` for apply gating.
- Phase 4AA tools/call timeout-hardening tranche now passes targeted checks:
  - `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
  - `62 passed` (`tests/test_mcp_wrapper_protocol.py`)
  - wrapper now enforces tool-call deadline budgets before host-side 120s channel limits and clamps retry timeouts to remaining budget.
- Phase 4AB startup-recovery budget-gating tranche now passes targeted checks:
  - `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
  - `64 passed` (`tests/test_mcp_wrapper_protocol.py`)
  - `71 passed` (`tests/test_phase_hygiene.py`, `tests/test_mcp_wrapper_protocol.py`)
  - wrapper now skips startup recovery preflight/retry work when remaining deadline budget is below configured threshold, reducing deadline overshoot risk near timeout windows.
- Phase 4AC host-timeout-derived deadline-budget tranche now passes targeted checks:
  - `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
  - `68 passed` (`tests/test_mcp_wrapper_protocol.py`)
  - `75 passed` (`tests/test_phase_hygiene.py`, `tests/test_mcp_wrapper_protocol.py`)
  - wrapper now derives default tool-call deadline budgets from host timeout minus margin with explicit override/disable semantics and safe minimum clamping.
- Phase 4AD explicit-deadline overrun-guardrail tranche now passes targeted checks:
  - `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
  - `70 passed` (`tests/test_mcp_wrapper_protocol.py`)
  - `77 passed` (`tests/test_phase_hygiene.py`, `tests/test_mcp_wrapper_protocol.py`)
  - explicit over-budget deadline values now clamp to host-safe budget by default unless overrun opt-out is explicitly enabled.
- Phase 4AE guarded-dispatch fail-fast response tranche now passes targeted checks:
  - `python -m py_compile mcp_wrapper.py tests/test_mcp_wrapper_protocol.py`
  - `71 passed` (`tests/test_mcp_wrapper_protocol.py`)
  - `78 passed` (`tests/test_phase_hygiene.py`, `tests/test_mcp_wrapper_protocol.py`)
  - `python -m eval.mcp_transport_soak --iterations 10 --warmup-requests 2 --timeout-sec 15 --transport framed --max-p95-ms 5000` -> PASS (`run_id=20260215_170548`)
  - guarded-dispatch path now returns deterministic `-32603` replies for request IDs when unexpected dispatcher exceptions occur.
- Phase 4AF unified SOTA+ verdict tranche now passes targeted checks:
  - `python -m py_compile eval/ollama_local_benchmark.py tests/test_ollama_local_benchmark.py`
  - `32 passed` (`tests/test_ollama_local_benchmark.py`)
  - `39 passed` (`tests/test_phase_hygiene.py`, `tests/test_ollama_local_benchmark.py`)
  - command now emits deterministic gate-family outcomes across quality/reliability/statistical/reproducibility/profile-policy dimensions.
- Phase 4AG deferred benchmark cadence tranche now passes targeted checks:
  - `python -m py_compile eval/ollama_local_benchmark.py tests/test_ollama_local_benchmark.py`
  - deferred-mode tests now validate benchmark-skip + report-reuse behavior and stale-report rejection (`tests/test_ollama_local_benchmark.py`)
  - combined targeted suite passes (`tests/test_phase_hygiene.py`, `tests/test_ollama_local_benchmark.py`).
- Initial cross-model quick-pass benchmark captured for 5 downloaded defaults (`xlam`, `qwen3:8b`, `deepseek-r1:8b`, `qwen2.5-coder:7b`, `llama3.1:8b`); snapshot and interpretation documented in `docs/plans/2026-02-14-phase4h-local-ollama-benchmarking.md`.
- Compile checks passed on all touched modules/tests.

## Newly Resolved Inaccuracies

1. User scope filtering no longer relies on brittle `metadata LIKE` patterns; JSON1 exact match is used with fallback.
2. MCP wrapper no longer hardcodes a single old protocol date; it negotiates and rejects unsupported protocol versions explicitly.
3. Eval harness now produces enforceable gate outcomes instead of report-only metrics.

## Unthought SOTA Enhancements (Recommended Additions)

1. **Continuous retrieval evaluation harness (must-have for SOTA claims)**
   - Add benchmark datasets + replay traces + nDCG@k / Recall@k / MRR + latency percentiles.
   - Required to validate adaptive-weights and explainability claims quantitatively.

2. **Policy-aware memory governance (RL-Driven)**
   - Per-memory retention TTL, PII tags, redaction/transformation policies, and auditable deletion proofs.
   - Transition to Reinforcement Learning (RL)-driven memory management. The system should autonomously learn when to Write, Delete, or Update memories based on retrieval reward signals, moving beyond predefined limits.

3. **Off-policy model upgrade over current SNIPS calibration**
   - Extend from scalar multipliers to feature-aware off-policy ranking updates (e.g., doubly robust estimators over trace features).
   - Improves long-horizon ranking quality beyond per-signal scalar adaptation.

4. **Trust/uncertainty propagation**
   - Carry confidence from extraction + contradiction scores + source reliability into final ranking and response generation.
   - Useful to avoid confidently surfacing low-trust facts.

5. **Memory compression/summarization layer for long-lived stores**
   - Periodic abstraction of dense episodic clusters into semantic memories with reversible provenance links.
   - Helps scale and keeps retrieval focused.
6. **MCP 2025-11 conformance tranche (follow-up now narrower)**
   - Baseline tasks/list/get/result/cancel + task-augmented `tools/call` + status notifications + retention/pagination governance + blocking-result responsive dispatch + metadata/cursor compliance are now implemented; remaining high-ROI work is advanced `input_required` task flows and optional persistent task-store backing.
7. **OpenTelemetry GenAI semantic instrumentation**
   - Standardized retrieval/add trace telemetry is implemented; next maturity step is dashboard/alert packs for regression triage.

## Suggested Plan Adjustments

1. Keep **Phase 1.1 Stabilization** as an explicit completed checkpoint in roadmap history:
   - Instructor wiring, Docker detection contract, trace raw-score fidelity, adaptive score-entropy, version consistency, and retrieval scope fixes.
2. Re-scope Phase 2 acceptance criteria around **measurable retrieval quality metrics** (not just feature existence).
3. Expand Phase 3 to include **ingestion safety hardening** (parser sandboxing, fail-open/skip semantics, and provenance metadata standards).
4. Add a cross-platform CI matrix + optional-dependency matrix as explicit deliverables before v3.2/v3.3 claims.
5. Add MCP 2025-11 interoperability and OTel GenAI instrumentation as cross-cutting release criteria.
6. Add a Phase 4 adaptive operator tranche for browser UI preferences + model profile routing (latency/quality/compute caliber control) with safe defaults.
7. Enforce single-PR workflow policy operationally: one branch/one open PR per phase, merge before next branch starts, with PR/comment checks at each phase boundary.
8. Add explicit Phase 4I acceptance criteria: ability-score + ability-per-resource thresholds for live and legacy-ingestion benchmark suites before profile default promotion.

## Vibecoder-Centric Additions (Multi-Assistant Continuity)

To align with the product intent (not enterprise-heavy), the highest-ROI additions are:

1. **Goal Compass + Drift Recovery**
   - Persist a project "north star" objective and constraints.
   - At `add/search`, detect semantic drift and inject short steering reminders.
   - Add `goal_relevance` as a retrieval signal.

2. **Cross-Assistant Handoff Bundles**
   - Deterministic import/export for moving between IDEs/agents.
   - Include goals, decisions, unresolved questions, and top-k memories.
   - Idempotent replay with event IDs for safe repeated imports.

3. **Release-Gated Retrieval Evaluation**
   - Require nDCG@k/Recall@k/MRR + latency budgets before SOTA claims.
   - A/B adaptive-vs-fixed and explain-on-vs-off to prove net value.

## Web Research Snapshot

Research notes and implementation guidance are documented in:

- `docs/WEB_RESEARCH_VIBECODER_SOTA.md`
- `docs/plans/2026-02-14-browser-ui-model-policy-design.md`
- `docs/plans/2026-02-14-phase4h-local-ollama-benchmarking.md`
- `docs/plans/2026-02-15-phase4i-ability-resource-benchmarking.md`
- `docs/plans/2026-02-15-phase4j-profile-promotion-gates.md`
- `docs/plans/2026-02-15-phase4k-hygiene-gate-and-roadmap-refresh.md`
- `docs/plans/2026-02-15-phase4l-dev-cycle-benchmark-orchestration.md`
- `docs/plans/2026-02-15-phase4m-dev-cycle-policy-apply-rollback.md`
- `docs/plans/2026-02-15-phase4n-policy-approval-manifest.md`
- `docs/plans/2026-02-15-phase4o-approval-provenance-context.md`
- `docs/plans/2026-02-15-phase4p-apply-checkpoint-provenance-enforcement.md`
- `docs/plans/2026-02-15-phase4q-git-ancestry-enforcement.md`
- `docs/plans/2026-02-15-phase4r-mcp-2025-11-25-compatibility.md`
- `docs/plans/2026-02-15-phase4s-mcp-task-lifecycle-baseline.md`
- `docs/plans/2026-02-15-mcp-transport-closed-recovery.md`
- `docs/plans/2026-02-15-phase4aa-mcp-tool-call-timeout-hardening.md`
- `docs/plans/2026-02-15-phase4ab-mcp-startup-recovery-budget-gating.md`
- `docs/plans/2026-02-15-phase4ac-mcp-host-timeout-derived-deadline-budget.md`
- `docs/plans/2026-02-15-phase4ad-mcp-explicit-deadline-overrun-guardrail.md`
- `docs/plans/2026-02-15-phase4ae-mcp-guarded-dispatch-fail-fast-response.md`
- `docs/plans/2026-02-15-sota-plus-quantitative-comparison-plan.md`
- `docs/plans/2026-02-15-phase4af-unified-sota-verdict-command.md`
- `docs/plans/2026-02-15-phase4ag-enhancement-first-benchmark-cadence.md`
- `docs/plans/2026-02-15-phase4l2-mcp-startup-tray-integration.md`

## Model-Caliber Research Update (2026-02-14)

Research-backed recommendation for SOTA+ operator profiles is to keep **caliber-based model selection** (not only think-level toggles):

1. `low_latency`: low-VRAM local model (`llama3.2:3b` baseline, optional `qwen3:4b`).
2. `balanced`: `qwen3:8b` default for quality/latency tradeoff.
3. `high_reasoning`: `qwen3:14b` default for 16GB-class workflows; reserve `qwen3:30b/32b` (or `deepseek-r1:32b`) for explicit high-VRAM opt-in sessions.
4. Keep xLAM as optional specialist endpoint for structured extraction/tool-calling workloads, not as mandatory global default.
5. For active coding sessions, pin runtime extraction to low-latency by default and split ingestion/legacy import into separate configurable profiles so background memory continuity does not consume planning-grade VRAM.
6. Local reproducibility is now codified with a versioned model matrix and benchmark harness:
   - `eval/ollama_model_matrix.json`
   - `eval/ollama_benchmark_prompts.jsonl`
   - `eval/ollama_local_benchmark.py`
   - `docs/plans/2026-02-14-phase4h-local-ollama-benchmarking.md`

Primary references for this recommendation:

- Ollama model availability/size envelopes (`qwen3`, `llama3.2`, `deepseek-r1`) and OpenAI-compatible API surface.
- xLAM-2 paper/repository for tool-agent design goals and function-calling orientation.
- Qwen3 technical report for current open-weight reasoning capability trajectory.

Key references reviewed include Elastic RRF docs, Qdrant/Pinecone hybrid search writeups, BEIR benchmark, Self-RAG, MCP specification, and idempotent receiver patterns.
