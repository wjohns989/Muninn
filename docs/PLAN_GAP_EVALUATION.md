# Muninn v3.3 SOTA+ Plan Gap Evaluation (Codebase Snapshot)

Date: 2026-02-14
Evaluator: Codex

## TL;DR

- **Phase 1 is mostly present, but with important correctness/integration gaps** (Instructor wiring, Docker path detection behavior, and recall trace fidelity).
- **Phase 2 is partially present** (conflict/dedup/weight-adapter modules exist) but not fully aligned with the plan’s dependency, data, and quality assumptions.
- **Phase 3 is largely missing** (no memory chains package, no ingestion pipeline package, no Python SDK package).

## Status vs Plan

### Phase 1 (v3.1.0)

| Plan item | Status | Evidence |
|---|---|---|
| 1A Platform abstraction | **Mostly implemented** | `muninn/platform.py` exists with cross-platform dirs/process helpers. |
| 1A Docker support | **Implemented** | `Dockerfile` and `docker-compose.yml` exist. |
| 1B Instructor extraction | **Partially implemented** | Instructor modules exist, but `MuninnMemory.initialize()` builds `ExtractionPipeline` without passing instructor config values, so configured Instructor endpoint/model are not actually wired in. |
| 1C Explainable recall traces | **Implemented with fidelity issues** | Trace models + hybrid retriever trace path exist, but raw score attribution uses rank proxies rather than signal-native scores. |
| 1D Feature flags | **Implemented** | `muninn/core/feature_flags.py` present and used across retrieval/extraction/memory initialization. |

### Phase 2 (v3.2.0)

| Plan item | Status | Evidence |
|---|---|---|
| 2A Conflict detection | **Implemented (feature-gated)** | `muninn/conflict/*` + memory integration present. |
| 2B Semantic dedup | **Implemented (feature-gated)** | `muninn/dedup/semantic_dedup.py` + memory add path integration present. |
| 2C Adaptive weights | **Implemented but mathematically weak** | `WeightAdapter` exists, but entropy currently derives from rank transforms, reducing discriminative value between signals with similar list lengths. |
| retrieval feedback persistence | **Missing** | Plan called for feedback table, but no retrieval feedback feature surfaced in current API/state flow. |

### Phase 3 (v3.3.0)

| Plan item | Status | Evidence |
|---|---|---|
| 3A Memory chains | **Missing** | No `muninn/chains/` package in repository tree. |
| 3B Multi-source ingestion | **Missing** | No `muninn/ingestion/` package, no `/ingest` server endpoint, no MCP ingest tool. |
| 3C Python SDK | **Missing** | No `muninn/sdk/` package; `muninn/__init__.py` does not export `Memory`/`AsyncMemory`. |

## High-Impact Issues Discovered

1. **Instructor integration gap (functional):** config contains Instructor fields, but `ExtractionPipeline` is initialized without instructor args in `MuninnMemory.initialize()`. Result: Instructor may never activate despite config/feature flag intent.
2. **Docker data-dir behavior mismatch:** tests currently expect Docker default `/data` when Docker-detected, but `get_data_dir()` only switches to Docker defaults when `MUNINN_DOCKER=1`, not when `is_running_in_docker()` is true. This creates an internal behavioral inconsistency.
3. **Explainability quality gap:** recall trace currently records `raw_score=float(rank)` for signals during fusion, which weakens “why” fidelity and undermines the uniqueness claim for explainable recall.
4. **Adaptive weighting signal-confidence quality gap:** entropy confidence is computed from rank-derived pseudo-scores, so confidence mostly tracks result-count shape rather than true retrieval certainty.
5. **Versioning inconsistency:** package version in `pyproject.toml` (3.1.0), MCP wrapper serverInfo (3.2.0), and `muninn.__version__` (3.0.0) are inconsistent.
6. **Plan/dependency mismatch:** `pyproject.toml` lacks the plan’s optional dependency groups (`conflict`, `ingestion`, `sdk`) and does not expose the roadmap-aligned install surfaces.

## Validation Snapshot

- Full test run is near-green, but one platform behavior test currently fails (`tests/test_platform.py::TestDataDir::test_docker_default`).

## Unthought SOTA Enhancements (Recommended Additions)

1. **Continuous retrieval evaluation harness (must-have for SOTA claims)**
   - Add benchmark datasets + replay traces + nDCG@k / Recall@k / MRR + latency percentiles.
   - Required to validate adaptive-weights and explainability claims quantitatively.

2. **Policy-aware memory governance**
   - Per-memory retention TTL, PII tags, redaction/transformation policies, and auditable deletion proofs.
   - Distinguishes enterprise/local-first deployments from generic OSS memory stores.

3. **Online learning-to-rank from implicit feedback**
   - Track clicks/uses/acceptance in downstream agent actions and optimize re-ranking/weighting over time.
   - Strong complement to current static heuristics.

4. **Trust/uncertainty propagation**
   - Carry confidence from extraction + contradiction scores + source reliability into final ranking and response generation.
   - Useful to avoid confidently surfacing low-trust facts.

5. **Memory compression/summarization layer for long-lived stores**
   - Periodic abstraction of dense episodic clusters into semantic memories with reversible provenance links.
   - Helps scale and keeps retrieval focused.

## Suggested Plan Adjustments

1. Add a **Phase 1.1 Stabilization** mini-phase:
   - Fix Instructor wiring, Docker detection contract, trace raw-score fidelity, and version consistency.
2. Re-scope Phase 2 acceptance criteria around **measurable retrieval quality metrics** (not just feature existence).
3. Expand Phase 3 to include **ingestion safety hardening** (parser sandboxing, fail-open/skip semantics, and provenance metadata standards).
4. Add a cross-platform CI matrix + optional-dependency matrix as explicit deliverables before v3.2/v3.3 claims.


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

