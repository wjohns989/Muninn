# Muninn Comprehensive Implementation Roadmap (v3.1.1 → v3.3.0+)

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

### Execution status update (2026-02-14, tranche progress)
Completed since last update:
1. Goal Compass implemented and wired into add/search with drift diagnostics and goal-aware retrieval signal.
2. Handoff export/import implemented with checksum verification and idempotent ledger replay.
3. Eval harness now includes latency summaries and CI-grade gate checks for regressions/budgets.
4. MCP compatibility hardening added (protocol negotiation + lifecycle initialization gating + schema annotations).
5. Optional OTel GenAI instrumentation added behind feature flag with privacy-safe defaults.
6. Metadata scoping upgraded from `LIKE` to JSON1 exact matching (with fallback), eliminating edge-case user filter misses.
7. Retrieval feedback persistence and adaptive calibration path implemented (storage + API/MCP + adaptive weight multipliers).
8. Retrieval feedback calibration upgraded with optional SNIPS estimator, using inverse-propensity normalization with bounded propensity clipping and effective-sample safeguards.
9. Feedback API/MCP surface extended with optional `rank` and `sampling_prob` fields for counterfactual calibration.
10. Eval harness extended with optional per-track metrics and latency summaries for competency-sliced benchmarking.
11. Eval runner now supports preset policy profiles + required track coverage gates, with auditable `gate_config` in output reports.
12. Eval runner now supports paired significance/effect-size analysis against baseline predictions, with optional significant-regression gate enforcement.
13. Canonical vibecoder benchmark artifacts are now committed with checksum manifest and reproducibility verifier CLI (`eval.artifacts`).
14. Eval significance gating now supports multiple-comparison correction policies (`none`/`bonferroni`/`holm`/`bh`) with configurable family scope (`all`/`by_track`) and adjusted p-value audit fields.
15. MCP conformance behavior is now hardened and tested:
   - unknown JSON-RPC request methods now return `-32601`,
   - premature `notifications/initialized` is ignored before successful `initialize`,
   - invalid `initialize`/`tools/call` param shapes now return `-32602`,
   - protocol test suite now asserts lifecycle ordering and schema contract consistency.
16. Canonical benchmark artifact coverage now includes a second robustness slice preset (`vibecoder_memoryagentbench_stress_v1`) and aggregate verifier support (`python -m eval.artifacts verify --all`).
17. OTel operational runbook and example collector configuration are now committed, including privacy policy defaults and smoke-test guidance.
18. Phase 3C Python SDK is now implemented (`muninn/sdk`) with sync+async clients, typed errors, mem0-style aliases, and dedicated tests/docs.
19. Phase 3B multi-source ingestion is now implemented with feature-gated fail-open parsing, provenance-rich chunk metadata, REST/MCP/SDK surface wiring, and targeted tests.
20. Legacy assistant/MCP migration flow is now implemented with discovery + selection-based import, including parser support for chat JSONL and sqlite-backed state stores.

Verification:
- Full suite now passes in-session: `362 passed, 2 skipped, 1 warning`.
- Targeted verification for changed areas:
  - `23 passed` across eval artifacts/statistics/presets/run/gates/metrics tests.
  - `21 passed` across eval statistics/presets/run/gates/metrics tests.
  - `15 passed` across eval run/gate/metric policy tests.
  - `48 passed` across eval/adaptive/MCP protocol tests.
  - `27 passed` across feedback-memory/config/hybrid-flag tests.
- Compile checks passed for all touched modules/tests.

### What already exists (partially or fully)
- Platform abstraction, feature flags, recall trace primitives, conflict detection, semantic dedup, adaptive weight adapter, Docker assets.
- Existing gap audit identified key correctness issues and phase-completion mismatches.

### Highest-priority correctness status (2026-02-14 update)
Fixed in current implementation slice:
1. Instructor config wiring through memory initialization.
2. Docker path behavior contract (`is_running_in_docker()` honored by default path resolution).
3. Recall trace `raw_score` fidelity (native score attribution).
4. Adaptive weighting entropy inputs (score-derived, not rank-derived).
5. Version consistency across package/server/MCP wrapper via single source of truth.
6. Graph retrieval argument mismatch (entity list vs string) and deterministic score output.
7. Final retrieval scope enforcement for user/namespace constraints.

Still open and blocking SOTA claims:
1. Benchmark corpus breadth improved (now multi-bundle), but additional domain slices are still needed for broader external validity.
2. Phase 3 ecosystem packages are still incomplete (`chains` missing; ingestion hardening remains).

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
- MCP specification for interop (latest 2025-11 revision).
- Anthropic agent workflow guidance.
- Idempotent receiver pattern.
- LangSmith evaluation patterns.
- OpenTelemetry tracing + GenAI semantic conventions.
- MemoryAgentBench-style competency framing for memory systems.

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

### ROI-first order (active execution)
1. Goal Compass + drift reminders + goal retrieval signal.
2. Handoff bundle export/import with checksum + idempotent replay.
3. Eval gate completion and benchmark baselines.
4. MCP 2025-11 compatibility + OTel GenAI instrumentation.

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

**Execution status:** `1.1A` and `1.1C` are completed in code. `1.1B` remains pending (benchmark corpus + repeatable reports).

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

#### Work item A5 — Retrieval Scope + Graph Correctness
- Enforce user/namespace constraints in final search candidate filtering.
- Fix graph signal call contract and produce deterministic score outputs.

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

## Phase 1.2 (v3.1.2) — Interop & Observability Gate (new)

### 1.2A MCP 2025-11 Compatibility Tranche
- Validate wrapper behavior against latest MCP specification/changelog impacts:
  - schema assumptions (JSON Schema 2020-12 expectations),
  - evolving elicitation/task workflow compatibility path,
  - tool metadata and error semantics alignment.

### 1.2B OpenTelemetry GenAI Instrumentation (opt-in)
- Emit structured spans/events for retrieval stages and extraction calls.
- Add privacy-safe mode with content redaction controls for sensitive payloads.
- Track latency and confidence diagnostics required by release gates.
- Status update: feature-flagged instrumentation is implemented and operational docs/examples now exist (`docs/OTEL_GENAI_OBSERVABILITY.md`, `examples/otel/collector-config.yaml`); next step is backend-specific dashboards/alerts.

### 1.2C Memory-Agent Benchmark Extension
- Extend eval harness with competency tracks inspired by MemoryAgentBench:
  - accurate retrieval,
  - test-time learning,
  - long-range understanding,
  - selective forgetting.
- Status update: track-level reporting is now implemented; remaining work is benchmark corpus coverage and CI baseline curation.

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
- Retrieval feedback loop is now implemented (persisted feedback + bounded per-signal multipliers) and remains disabled by default until sufficient signal quality exists.

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
- Status update: implemented with feature-gated fail-open pipeline (`muninn/ingestion`), per-source/per-chunk reporting, chat-context adapters for `.jsonl/.ndjson`, sqlite-backed parser support (`.vscdb/.db/.sqlite*`), and REST/MCP/SDK parity for both baseline ingestion and legacy discovery/import flows.

### 3C Python SDK
- Sync + async clients.
- Context manager ergonomics.
- Mem0-style compatibility surface where possible.
- Status update: implemented (`MuninnClient`, `AsyncMuninnClient`, `Memory`, `AsyncMemory`) with typed exception hierarchy and SDK tests.

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
- **Week 4:** Phase 1.2 MCP compatibility + OpenTelemetry GenAI instrumentation baseline.
- **Weeks 5–6:** Phase 2 intelligence features with measured A/B validation.
- **Weeks 7–9:** Phase 3 ecosystem (ingestion, SDK, chains, handoff interop).

---

## 10) Immediate Next Actions (Execution Checklist)

1. Mark Phase 1.1 correctness bundle complete in release notes (A1–A5 now implemented).
2. Stand up eval harness + baseline report (BEIR-style retrieval metrics + vibecoder suites).
3. Implement Goal Compass in feature-flagged mode.
4. Add Phase 1.2 MCP 2025-11 conformance checklist and OTel GenAI opt-in tracing.
5. Define handoff bundle schema v1 and conformance tests.

If these five actions are completed with passing metrics, Muninn can credibly claim a robust SOTA+ trajectory for the intended vibecoder use case.
