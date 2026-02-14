# Web Research: Missing Features + SOTA Implementation Guidance (Vibecoder Focus)

Date: 2026-02-14
Scope: Muninn roadmap updates for multi-assistant/IDE workflows with local-first memory.

## Product Lens (non-enterprise, high-leverage)

Muninn should optimize for solo/small-team vibecoders who:
- switch often between frontier assistants,
- move between IDEs and terminals,
- need continuity of project goals/constraints,
- want low-friction local-first operation with optional scale-up.

This implies prioritizing: **goal continuity, handoff portability, retrieval quality measurement, and deterministic merges** over enterprise governance overhead.

## Sources Reviewed (Web)

1. Elastic RRF reference (fusion mechanics and stability):
   - https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html
2. Qdrant hybrid retrieval design notes:
   - https://qdrant.tech/articles/hybrid-search/
3. Pinecone hybrid search overview (dense+sparse tradeoffs):
   - https://www.pinecone.io/learn/hybrid-search/
4. BEIR benchmark framing for retrieval evaluation:
   - https://arxiv.org/abs/2104.08663
5. Self-RAG reflection concept (retrieval + critique loop):
   - https://arxiv.org/abs/2310.11511
6. MCP specification (cross-tool interoperability):
   - https://modelcontextprotocol.io/specification/2025-06-18
7. Agent orchestration quality practices:
   - https://www.anthropic.com/engineering/building-effective-agents
8. Idempotent receiver pattern for safe replay/import:
   - https://martinfowler.com/articles/patterns-of-distributed-systems/idempotent-receiver.html
9. Practical eval instrumentation patterns:
   - https://docs.smith.langchain.com/evaluation
10. Trace-level observability for debugging complex retrieval:
    - https://opentelemetry.io/docs/concepts/signals/traces/

## New Missing Features Identified (Beyond Current Plan)

### A) Goal Compass + Drift Recovery (Highest ROI)

**What’s missing now:** No first-class project-goal primitive and no runtime drift checks.

**Why this matters:** In multi-assistant coding, users quickly lose thread continuity.

**Implementation (high performance):**
- Store one active `ProjectGoal` per `(project, namespace, user)`.
- Create embeddings for `goal_statement` + `constraints` and cache in memory.
- At `add/search`, compute cosine with request intent embedding.
- If similarity below threshold, inject short steering hint and add `goal_relevance` as retrieval signal.
- Keep this O(1) per request (single vector compare + optional template string).

### B) Portable Handoff Bundles (Assistant/IDE swapping)

**What’s missing now:** no deterministic import/export contract for switching tools.

**Implementation (high performance):**
- `export_handoff(project_id)` emits canonical JSON with:
  - goal, constraints, decisions, unresolved threads,
  - top-k memories from importance*recency,
  - source metadata and monotonic event IDs.
- `import_handoff(bundle)` is idempotent:
  - skip already seen event IDs,
  - merge deduplicated memories,
  - run conflict detection only on changed/novel entries.
- Complexity bounded by bundle size, not full database size.

### C) Evidence-grade Evaluation Harness (for SOTA claims)

**What’s missing now:** no reproducible benchmark loop for retrieval quality.

**Implementation:**
- Frozen eval corpus + query sets from real coding tasks.
- Run nDCG@k, Recall@k, MRR, p50/p95 latency on every release candidate.
- Add A/B runner for fixed vs adaptive weights and explainability overhead.
- Ship CLI: `python -m eval.run --preset vibecoder`.

### D) Observability for Explainability Trust

**What’s missing now:** recall traces exist, but no systematic operational visibility.

**Implementation:**
- Emit structured spans for retrieval stages (`vector`, `bm25`, `graph`, `temporal`, fusion, rerank).
- Persist top-k trace diagnostics for failed/low-confidence queries.
- Track drift-alert frequency to tune thresholds and reduce noisy reminders.

## Critical Issues/Accuracy Corrections

1. **Adaptive entropy currently rank-derived** (not score-derived): weak confidence semantics.
2. **Recall trace raw score fidelity is reduced** when rank proxy is used.
3. **Version strings diverged** across package/server/wrapper.
4. **Instructor config wiring appears incomplete** in initialization path.
5. **Docker detection contract mismatch** between behavior and test expectations.

## Recommended Plan Changes (Actionable)

1. Insert **Phase 1.1 Stabilization & Measurement Gate** before Phase 2.
2. Add **Goal Compass** in Phase 1.1 (feature-flagged, default ON in local mode).
3. Add **Cross-Assistant Handoff Pack** as a Phase 3 item.
4. Add release criteria requiring benchmark deltas and latency budgets.

## Practical Performance Budget Targets

- `search` p95 overhead from explainability: **< 12 ms** vs baseline.
- `add` p95 overhead with conflict+dedup ON: **< 35 ms** CPU-only (excluding model cold-start).
- Goal drift check overhead: **< 2 ms** once goal embedding is cached.
- Handoff export bundle size: **< 250 KB** default; hard cap configurable.

