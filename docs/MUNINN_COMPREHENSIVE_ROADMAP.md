# Muninn Comprehensive Implementation Roadmap (v3.1.0 → v3.3.0+)

**Date:** 2026-02-14  
**Audience:** Core maintainers and contributors  
**Primary product target:** Vibecoders using multiple frontier assistants/agents/IDEs, with local-first continuity and optional scale-up.  
**Compatibility goal:** Backward compatible, additive-by-default, optional advanced features.

---

## 1) Product North Star

Muninn should become the best **local-first cross-assistant memory substrate** for users who continuously switch between tools (Claude, ChatGPT, Cursor, Windsurf, Copilot, CLI agents, IDE extensions).

### Success criteria
1. **Goal continuity:** The system reliably preserves and re-surfaces project objective/constraints across sessions and tools.
2. **Decision continuity:** Important decisions and unresolved threads survive assistant switching and handoffs.
3. **Retrieval trust:** Results are explainable, measurable, and robust under drift/conflict/noise.
4. **Operational simplicity:** Works out-of-the-box on Windows/Linux/macOS and Docker.

---

## 2) Current-State Reality (Gap Baseline)

### What already exists (partially or fully)
- Platform abstraction, feature flags, recall trace primitives, conflict detection, semantic dedup, adaptive weight adapter, Docker assets.
- Existing gap audit identified key correctness issues and phase-completion mismatches.

### Highest-priority correctness gaps
1. Instructor config is not fully wired through memory initialization path.
2. Docker path behavior contract is inconsistent with current tests and detection semantics.
3. Recall trace `raw_score` fidelity is weakened by rank-proxy use.
4. Adaptive weighting confidence is rank-derived rather than score-derived.
5. Version numbers are inconsistent across package/server/wrapper.

These are blockers for trustworthy SOTA claims and should be fixed before deeper feature expansion.

---

## 3) Research-Grounded Design Principles

This roadmap follows concrete retrieval/agent-system practices:
- **RRF stability and hybrid retrieval:** use robust reciprocal-rank fusion and sparse+dense+graph composition.
- **Evaluation-first retrieval quality:** nDCG/Recall/MRR + latency budgets before claiming improvements.
- **Idempotent handoff/import semantics:** repeated imports must be safe and deterministic.
- **Traceability:** retrieval explanations and operational traces should be machine- and human-consumable.

Reference set used to shape this roadmap:
- Elastic RRF reference.
- Qdrant hybrid retrieval guidance.
- Pinecone hybrid search tradeoffs.
- BEIR retrieval benchmarking framing.
- Self-RAG retrieval+reflection paradigm.
- MCP specification for interop.
- Anthropic agent workflow guidance.
- Idempotent receiver pattern.
- LangSmith evaluation patterns.
- OpenTelemetry tracing concepts.

(Links are consolidated in `docs/WEB_RESEARCH_VIBECODER_SOTA.md`.)

---

## 4) End-State Capability Model (v3.3.0+)

After this roadmap, Muninn should provide:
1. **5-signal retrieval** (vector, BM25, graph, temporal, goal relevance).
2. **Explainable recall traces** with true per-signal score attribution.
3. **Conflict-aware and dedup-aware ingestion** with feature-gated cost controls.
4. **Cross-assistant handoff bundles** with deterministic idempotent import.
5. **Multi-source ingestion** and Python SDK parity across REST+MCP+SDK.
6. **Evaluation gate** as a hard release criterion.

---

## 5) Phased Delivery Plan

## Phase 1 (v3.1.0) — Foundation (already partially implemented)

### Scope
- Platform abstraction
- Instructor-based structured extraction
- Explainable recall traces
- Centralized feature flags

### Exit criteria (must verify now, not assume)
- All phase features actually wired (not only modules present).
- Tests pass across supported platforms.
- No version inconsistencies.

---

## Phase 1.1 (v3.1.1) — Stabilization & Measurement Gate (hard blocker)

**Objective:** Convert partial implementation into production-grade correctness and measurable quality.

### 1.1A Correctness Fix Bundle

#### Work item A1 — Instructor wiring fix
- Ensure `MuninnMemory.initialize()` forwards instructor config into `ExtractionPipeline`.
- Add integration test proving configured endpoint/model are used.

**Example (target behavior):**
```python
self._extraction = ExtractionPipeline(
    xlam_url=...,
    ollama_url=...,
    instructor_base_url=self.config.extraction.instructor_base_url,
    instructor_model=self.config.extraction.instructor_model,
    instructor_api_key=self.config.extraction.instructor_api_key,
)
```

#### Work item A2 — Docker data-dir contract fix
- Use actual container detection in path resolution (not only env override).
- Align runtime behavior with `tests/test_platform.py` expectations.

#### Work item A3 — Recall trace fidelity fix
- Capture native signal scores wherever possible:
  - vector cosine score,
  - BM25 score,
  - graph path/confidence score,
  - temporal decay/recency score.
- Keep rank as metadata, not as `raw_score` substitute.

#### Work item A4 — Version unification
- Single source of truth for versioning.
- Validate parity in `pyproject.toml`, package `__version__`, server metadata, MCP wrapper.

### 1.1B Retrieval Evaluation Gate

#### Deliverables
- `eval/` harness with fixed datasets and repeatable query suites.
- Metrics:
  - Relevance: `nDCG@k`, `Recall@k`, `MRR`
  - Performance: p50/p95 latency, memory footprint deltas
  - Integrity: contradiction FP/FN estimate on labeled slice
- A/B profiles:
  - fixed weights vs adaptive
  - explain off vs explain on

**Release policy:** no promotion to v3.2 unless metrics and latency budgets pass.

### 1.1C Goal Compass + Drift Guardrail (new, vibecoder-critical)

#### Problem
Users lose direction while switching assistants and contexts.

#### Solution
Introduce a `ProjectGoal` memory primitive plus drift checks at add/search time.

#### Design
- Store canonical goal object per `(project, namespace, user)`.
- Compute `goal_relevance` from intent-vs-goal embedding similarity.
- If drift is high, prepend concise steering reminder.
- Add goal relevance as 5th retrieval signal (feature-flagged).

**Example reminder style:**
> “You’re currently working toward ‘MCP memory server with cross-assistant continuity.’ This query appears tangential; do you want to continue or refocus on ingestion reliability?”

#### Performance constraints
- Goal check target overhead: `<2ms` with cached goal embedding.

### Phase 1.1 exit criteria
1. Zero known correctness mismatches from gap audit.
2. Cross-platform test matrix green (Linux/macOS/Windows).
3. Evaluation harness operational and producing stored reports.
4. Goal drift reminders demonstrate measurable continuity improvements in test sessions.

---

## Phase 2 (v3.2.0) — Intelligence

### 2A Conflict detection
- Keep NLI model feature-gated.
- Pre-filter candidates by similarity to cap runtime.
- Add explicit policy options:
  - `strict`: block contradiction without user confirmation,
  - `balanced`: flag/supersede heuristically,
  - `lenient`: annotate conflict only.

### 2B Semantic dedup
- Preserve conservative default thresholds.
- Add shadow-mode telemetry first (detect but do not merge) to estimate false positive risk before hard enablement.

### 2C Adaptive retrieval weights
- Replace rank-derived entropy with score-derived entropy where scores exist.
- Add fallback normalization strategy for scoreless channels.
- Add online feedback hooks but keep disabled by default until sufficient signal quality exists.

### Phase 2 exit criteria
1. Quality uplift over fixed baseline demonstrated on eval harness.
2. No unacceptable p95 latency regression in add/search.
3. Conflict and dedup false-positive rates within budget.

---

## Phase 3 (v3.3.0) — Ecosystem & Interop

### 3A Memory chains
- Temporal and causal link extraction.
- Controlled thresholds to reduce false chain creation.

### 3B Multi-source ingestion
- Parser adapters for txt/md/pdf/docx/html/json/csv.
- Fail-open by source item (skip bad files, continue pipeline).
- Mandatory provenance metadata per ingested chunk.

### 3C Python SDK
- Sync + async clients.
- Context manager ergonomics.
- Mem0-style compatibility surface where possible.

### 3D Cross-assistant handoff + interop pack

#### Why
This is core for vibecoders, not optional polish.

#### `export_handoff(project_id)` bundle spec
- Goal/constraints/do-not-break rules
- Recent decisions + unresolved questions
- Top-k memory slices by `importance * recency * goal_relevance`
- Provenance and checksum/signature
- Event watermark for idempotent replay

#### `import_handoff(bundle)` behavior
- Idempotent event application
- Dedup and conflict-aware merge
- Partial import reporting

**Example bundle shape:**
```json
{
  "project_id": "muninn",
  "goal": {"statement": "Ship v3.1.1 stabilization", "constraints": ["local-first", "backward-compatible"]},
  "decisions": [{"id": "d-102", "text": "Adopt score-based trace attribution", "status": "accepted"}],
  "open_questions": ["Should adaptive weights use per-signal calibration?"],
  "memories": [{"id": "m-1", "content": "Instructor wiring incomplete", "importance": 0.89}],
  "events": [{"event_id": "evt-991", "type": "decision"}],
  "checksum": "sha256:..."
}
```

### Phase 3 exit criteria
1. Handoff import/export works across REST+MCP+SDK.
2. End-to-end continuity scenario passes (Assistant A → IDE plugin → CLI agent).
3. Ingestion and SDK documented with runnable examples.

---

## 6) Cross-Cutting Workstreams (All Phases)

### A) Testing and validation
- Unit tests per module.
- Integration tests for add/search/update/delete/handoff.
- Regression test pack for known correctness bugs.
- Performance regression tests tied to budgets.

### B) Observability
- Structured logs for retrieval stage timing.
- Trace IDs attached to recall explanations.
- Optional OpenTelemetry instrumentation path.

### C) Compatibility and rollout
- All major additions behind feature flags.
- Defaults tuned for low friction local use.
- Progressive rollout: shadow mode → opt-in → default-on only after evidence.

### D) Documentation and examples
- “5-minute local setup” for vibecoder workflow.
- “Switch assistant safely” handoff tutorial.
- Goal Compass quickstart with real prompt examples.

---

## 7) Performance & Quality Budgets

## Search path budgets
- Explainable recall overhead: `<12ms` p95 over baseline.
- Adaptive weighting overhead: `<5ms` p95.
- Goal drift check overhead: `<2ms` p95 with caching.

## Add path budgets
- Dedup + conflict overhead: `<35ms` p95 CPU-only (excluding model cold start).

## Quality budgets
- nDCG@10 and Recall@10 must improve or remain neutral with confidence bounds.
- Contradiction false positive rate must stay below agreed threshold before default-on.

---

## 8) Risk Register + Mitigation

1. **Over-complexity risk** (too many features too quickly)
   - Mitigation: strict phase gates and feature-flag defaults.
2. **Latency creep risk**
   - Mitigation: budget enforcement in CI.
3. **False conflict/dedup actions**
   - Mitigation: shadow mode + conservative thresholds + policy modes.
4. **Interoperability drift across tools**
   - Mitigation: canonical handoff schema + conformance tests.
5. **SOTA claim mismatch**
   - Mitigation: publish evaluation artifacts with each release candidate.

---

## 9) Detailed Timeline (Revised)

- **Weeks 1–2:** Phase 1 hardening verification and remaining wiring checks.
- **Week 3:** Phase 1.1 stabilization + evaluation gate + Goal Compass MVP.
- **Weeks 4–5:** Phase 2 intelligence features with measured A/B validation.
- **Weeks 6–8:** Phase 3 ecosystem (ingestion, SDK, chains, handoff interop).

---

## 10) Immediate Next Actions (Execution Checklist)

1. Implement Phase 1.1 A1–A4 fixes.
2. Stand up eval harness + baseline report.
3. Implement Goal Compass in feature-flagged mode.
4. Define handoff bundle schema v1 and conformance tests.
5. Align versioning and release process to one source of truth.

If these five actions are completed with passing metrics, Muninn can credibly claim a robust SOTA+ trajectory for the intended vibecoder use case.
