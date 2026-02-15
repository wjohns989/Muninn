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
5. **Plan/dependency mismatch (open):** `pyproject.toml` still lacks full roadmap optional dependency groups (`conflict`, `ingestion`, `sdk`) and release-profile surfaces.
6. **Evaluation corpus breadth still incomplete (open):** gate mechanics and artifact coverage now include two bundles, but additional domain and noise/adversarial slices are still needed.
7. **Parser sandbox/process isolation still open (security hardening):** optional binary backends (`pdf/docx`) remain in-process and should be isolated for stricter threat models.
8. **Extraction/model policy partially open:** profile routing, UI profile persistence, session-level override wiring, operation-scoped runtime/ingestion profile defaults, runtime profile mutation API, mutation audit events, local model-matrix benchmarking harness, ability/resource benchmark scoring, controlled apply/rollback mutation flow, and approval-gated checkpoint apply are now implemented, but profile-level telemetry/alert thresholds and auto-governance promotion controls still need completion before default-policy automation.
9. **MCP Muninn transport reliability intermittency (operationally mitigated):** framing mismatch is fixed and startup bootstrap is in place; stale closed handles still require session restart by design, now covered by explicit recovery runbook and tray health-probe tooling.

## Validation Snapshot

- Full suite now passes in-session: `418 passed, 2 skipped, 0 warnings`.
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

2. **Policy-aware memory governance**
   - Per-memory retention TTL, PII tags, redaction/transformation policies, and auditable deletion proofs.
   - Distinguishes enterprise/local-first deployments from generic OSS memory stores.

3. **Off-policy model upgrade over current SNIPS calibration**
   - Extend from scalar multipliers to feature-aware off-policy ranking updates (e.g., doubly robust estimators over trace features).
   - Improves long-horizon ranking quality beyond per-signal scalar adaptation.

4. **Trust/uncertainty propagation**
   - Carry confidence from extraction + contradiction scores + source reliability into final ranking and response generation.
   - Useful to avoid confidently surfacing low-trust facts.

5. **Memory compression/summarization layer for long-lived stores**
   - Periodic abstraction of dense episodic clusters into semantic memories with reversible provenance links.
   - Helps scale and keeps retrieval focused.
6. **MCP 2025-11 conformance tranche**
   - Align wrapper/protocol behavior with latest spec changes (tasks support trajectory, elicitation schema/default semantics, JSON Schema 2020-12 assumptions).
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
- `docs/plans/2026-02-15-mcp-transport-closed-recovery.md`
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
