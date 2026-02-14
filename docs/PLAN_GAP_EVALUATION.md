# Muninn v3.3 SOTA+ Plan Gap Evaluation (Codebase Snapshot)

Date: 2026-02-14
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
- **Phase 3 is substantially advanced**: Python SDK and multi-source ingestion are now shipped; memory chains package remains missing.

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
| 3A Memory chains | **Missing** | No `muninn/chains/` package in repository tree. |
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

## Validation Snapshot

- Full suite now passes in-session: `378 passed, 2 skipped, 1 warning`.
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

Key references reviewed include Elastic RRF docs, Qdrant/Pinecone hybrid search writeups, BEIR benchmark, Self-RAG, MCP specification, and idempotent receiver patterns.
