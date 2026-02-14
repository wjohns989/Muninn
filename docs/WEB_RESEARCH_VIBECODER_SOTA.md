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
   - https://modelcontextprotocol.io/specification/2025-11-25
   - https://modelcontextprotocol.io/specification/2025-11-25/changelog
7. Agent orchestration quality practices:
   - https://www.anthropic.com/engineering/building-effective-agents
8. Idempotent receiver pattern for safe replay/import:
   - https://martinfowler.com/articles/patterns-of-distributed-systems/idempotent-receiver.html
9. Practical eval instrumentation patterns:
   - https://docs.smith.langchain.com/evaluation
10. Trace-level observability for debugging complex retrieval:
    - https://opentelemetry.io/docs/concepts/signals/traces/
11. GenAI telemetry semantics (events/attributes/PII warnings):
    - https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/
12. Memory-agent benchmark framing:
    - https://arxiv.org/abs/2507.05257
13. NLI model card + license validation for contradiction detection:
    - https://huggingface.co/api/models/cross-encoder/nli-deberta-v3-small
14. MCP adoption evidence in mainstream IDE workflows:
    - https://github.blog/changelog/2025-07-14-model-context-protocol-mcp-support-in-vs-code-is-generally-available/
15. MCP lifecycle initialization requirements:
    - https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle
16. MCP versioning semantics (date-based protocol versioning and compatibility):
   - https://modelcontextprotocol.io/specification/2025-11-25/basic#versioning
17. Counterfactual/online LTR grounding (for feedback-loop evolution):
   - https://arxiv.org/abs/1907.06412
   - https://arxiv.org/abs/2012.04426
   - https://arxiv.org/abs/2506.20854
18. MemoryAgentBench dataset card (split composition for track gating):
   - https://huggingface.co/datasets/ai-hyz/MemoryAgentBench
19. BEIR reproducibility and statistical-analysis guidance:
   - https://cs.uwaterloo.ca/~jimmylin/publications/Kamalloo_etal_SIGIR2024.pdf
20. Self-normalized off-policy estimators for ranking:
   - https://blondon.github.io/papers/london-consequences23.pdf
21. IR significance testing comparison (bootstrap/permutation guidance):
   - https://maroo.cs.umass.edu/getpdf.php?id=744
22. IR significance testing error analysis at scale:
   - https://arxiv.org/pdf/1905.11096
23. BEIR statistical/reproducibility framing:
   - https://cs.uwaterloo.ca/~jimmylin/publications/Kamalloo_etal_SIGIR2024.pdf
24. Multiple-testing correction reference (Holm/BH families):
   - https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html
25. Canonical correction-method definitions (`bonferroni`, `holm`, `BH/FDR`):
   - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html
26. JSON-RPC 2.0 method-not-found + request semantics:
   - https://www.jsonrpc.org/specification
27. Robustness in information retrieval systems (survey framing):
   - https://arxiv.org/abs/2404.10179
28. OTLP exporter environment configuration (Python):
   - https://opentelemetry-python.readthedocs.io/en/latest/exporter/otlp/otlp.html
29. OTel security guidance for sensitive data:
   - https://opentelemetry.io/docs/security/handling-sensitive-data/
30. Requests session guidance (connection reuse / advanced usage):
   - https://requests.readthedocs.io/en/latest/user/advanced/
31. HTTPX client lifecycle guidance (sync/async client reuse):
   - https://www.python-httpx.org/advanced/clients/
32. OWASP file upload hardening guidance (applicable to ingestion safety):
   - https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html
33. Python CSV standard-library behavior and limits:
   - https://docs.python.org/3/library/csv.html
34. Python HTML parser standard-library reference:
   - https://docs.python.org/3/library/html.parser.html
35. pypdf extraction and memory caveats:
   - https://pypdf.readthedocs.io/en/latest/user/extract-text.html
36. python-docx document API quickstart:
   - https://python-docx.readthedocs.io/en/latest/user/quickstart.html
37. Codex CLI config + state location (`CODEX_HOME`, defaults):
   - https://developers.openai.com/codex/config-advanced/
38. Codex CLI history growth issue referencing `CODEX_HOME/history.jsonl`:
   - https://github.com/openai/codex/issues/4963
39. Claude Code repository issue confirming local session JSONL paths:
   - https://github.com/anthropics/claude-code/issues/22365
40. VS Code user-data and platform path contract (portable mode docs):
   - https://code.visualstudio.com/docs/editor/portable
41. VS Code discussion referencing chat session storage under workspaceStorage:
   - https://github.com/microsoft/vscode-discussions/discussions/168163
42. VS Code issue referencing `workspaceStorage/.../state.vscdb`:
   - https://github.com/microsoft/vscode/issues/179882
43. Cursor community thread on local chat history location:
   - https://stackoverflow.com/questions/79398677/where-does-cursor-store-chat-history
44. Cursor chat browser implementation notes for workspace storage parsing:
   - https://github.com/alexjbuck/cursor-chat-browser/blob/main/README.md

## Legacy Chat/Memory Storage Research (2026-02-14)

### Confidence Matrix

- **High confidence**
  - Codex CLI: `CODEX_HOME` defaults to `~/.codex`; `history.jsonl` and session artifacts under that root.
  - Claude Code: session JSONL under `~/.claude/projects/...`.
- **Medium confidence**
  - VS Code/Copilot/Cursor-style stores: workspace/global state databases under `User/workspaceStorage` and `User/globalStorage` (sqlite `state.vscdb`, plus `chatSessions/*.json` where present).
  - Antigravity brain outputs under `.gemini/antigravity/brain/**/output.txt` (project-observed convention).
- **Low confidence**
  - Claude Desktop and ChatGPT desktop local artifact roots on each OS: directory conventions vary and official vendor docs are limited.

### Implementation Decision

To avoid brittle single-path assumptions, discovery was implemented as:
- provider-specific path patterns across Windows/macOS/Linux,
- confidence tagging per provider,
- user-supplied root scanning (`roots[]`) for unknown/custom layouts,
- parser-supported gating with explicit unsupported reporting.

This balances ingestion ROI against ecosystem path volatility and avoids hard-coding unverifiable assumptions as invariants.

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

### E) MCP 2025-11 Compatibility Tranche (High ROI)

**What’s missing now:** roadmap and wrappers still target older MCP assumptions.

**Why this matters:** current spec adds durable task flow support, updated elicitation schema behavior, and stronger schema expectations that improve interoperability.

**Implementation:**
- Add compatibility review against 2025-11 key changes (tasks, elicitation enum/default behavior, JSON Schema 2020-12 assumptions).
- Update wrapper schema metadata and validation behavior where required.
- Add conformance tests for tool error semantics and schema correctness.

**Implemented in current tranche:**
- Protocol negotiation now accepts supported versions and rejects unsupported protocol versions explicitly.
- Server now gates `tools/list` and `tools/call` until `notifications/initialized` is received.
- Tool schemas now carry JSON Schema 2020-12 declaration and read-only annotations where applicable.
- Unknown request methods now return explicit JSON-RPC `-32601` instead of silent ignore.
- `notifications/initialized` before successful `initialize` is now ignored to enforce lifecycle ordering.
- Invalid `initialize` and `tools/call` param shapes now return `-32602` to harden contract compliance.
- OTel runbook and collector example are now added for operational rollout (`docs/OTEL_GENAI_OBSERVABILITY.md`, `examples/otel/collector-config.yaml`) with privacy defaults and capture-bound policy.

### F) Memory-Specific Benchmarking (High ROI)

**What’s missing now:** corpus depth + reproducible baselines per competency track are still limited.

**Implementation:**
- Extend eval harness with MemoryAgentBench-style competency tracks:
  - accurate retrieval
  - test-time learning
  - long-range understanding
  - selective forgetting
- Track phase-level quality deltas and reject regressions before release tags.

**Implemented in current tranche:**
- `eval` now supports optional per-query `track` labels and emits per-track metrics/latency summaries in reports.
- This enables competency-sliced release checks without breaking existing global report format.
- Eval CLI now supports named preset policy profiles with explicit required track coverage and per-track regression budgets.
- Eval CLI now supports paired significance/effect-size reporting over baseline predictions (bootstrap CIs, permutation p-values, Cohen's d) and optional significant-regression fail gates.
- Canonical benchmark artifacts are now committed for the vibecoder preset with manifest-based checksum verification and reproducibility checks.
- Canonical artifact coverage now includes a second robustness stress slice preset and aggregate verifier support (`python -m eval.artifacts verify --all`) for CI-scale integrity checks.

### G) Retrieval Feedback Calibration Loop (implemented, high ROI)

**What was missing:** feedback signals were not persisted and could not calibrate ranking over time.

**Implemented now:**
- Persist retrieval outcomes in scoped storage (`user_id`, `namespace`, `project`).
- Record optional signal contributions per event.
- Compute bounded per-signal multipliers from historical outcome-weighted signal evidence.
- Apply multipliers in adaptive weighting path with TTL cache + invalidation on new feedback.

**Implemented upgrade (counterfactual path):**
- Feedback records now optionally persist `rank` and `sampling_prob`.
- Multiplier calibration now supports `weighted_mean` and `snips` estimators.
- SNIPS mode uses inverse-propensity normalization, clipped propensities, and effective-sample-size guardrails to reduce logged-policy exposure bias.
- Eval significance gating now supports multiple-comparison correction (`none`/`bonferroni`/`holm`/`bh`) with configurable family scope (`all`/`by_track`), and gate enforcement now uses adjusted p-values.

**Next research-backed upgrade path:**
- Evolve from per-signal scalar multipliers to feature-aware off-policy updates (e.g., doubly robust estimators over trace features).
- Expand canonical artifacts from two bundles to broader multi-domain suites (e.g., codebase-scale, temporal drift, and adversarial-noise slices) with the same manifest and reproducibility contract.

## Critical Issues/Accuracy Corrections

1. **Adaptive entropy currently rank-derived** (not score-derived): weak confidence semantics. ✅ Fixed in current code slice.
2. **Recall trace raw score fidelity is reduced** when rank proxy is used. ✅ Fixed in current code slice.
3. **Version strings diverged** across package/server/wrapper. ✅ Fixed with single source of truth (`muninn/version.py`).
4. **Instructor config wiring appears incomplete** in initialization path. ✅ Fixed in `MuninnMemory.initialize()`.
5. **Docker detection contract mismatch** between behavior and test expectations. ✅ Fixed by using runtime Docker detection.
6. **Graph retrieval argument mismatch** (`str` passed where entity list expected). ✅ Fixed.
7. **Search scope leakage risk** (`user_id` not enforced end-to-end in hybrid retrieval). ✅ Fixed with final-record scope filters.
8. **No retrieval-feedback persistence loop** (adaptive weighting could not learn online). ✅ Fixed with scoped feedback storage + calibration multipliers.
9. **No counterfactual correction for feedback bias** in ranking calibration. ✅ Fixed with optional SNIPS estimator + propensity-aware feedback fields.

## Recommended Plan Changes (Actionable)

1. Insert **Phase 1.1 Stabilization & Measurement Gate** before Phase 2.
2. Add **Goal Compass** in Phase 1.1 (feature-flagged, default ON in local mode).
3. Add **Cross-Assistant Handoff Pack** as a Phase 3 item.
4. Add release criteria requiring benchmark deltas and latency budgets.
5. Add an explicit MCP compatibility checkpoint for 2025-11 spec changes.
6. Add an OpenTelemetry GenAI instrumentation profile (opt-in) for production diagnostics.

## Practical Performance Budget Targets

- `search` p95 overhead from explainability: **< 12 ms** vs baseline.
- `add` p95 overhead with conflict+dedup ON: **< 35 ms** CPU-only (excluding model cold-start).
- Goal drift check overhead: **< 2 ms** once goal embedding is cached.
- Handoff export bundle size: **< 250 KB** default; hard cap configurable.

## Additional Optimization Discovered During Implementation

### JSON1-backed tenant scoping (high ROI, correctness + speed)

**Issue found:** user scoping in SQLite relied on `metadata LIKE` pattern matching, which can miss quoted values and performs poorly at scale.

**Upgrade implemented:**
- Use `json_extract(metadata, '$.user_id') = ?` when JSON1 is available.
- Keep fallback to LIKE for compatibility on limited SQLite builds.
- Create expression index on extracted user_id when JSON1 is available.

**Ecosystem impact:**
- More reliable multi-tenant isolation.
- Lower query cost for scoped retrieval/deletion paths.
- No API contract break; storage behavior improves transparently.

### Python SDK transport lifecycle (high ROI, performance + reliability)

**Issue found:** direct per-call HTTP usage increases connection churn and makes async integration awkward.

**Upgrade implemented:**
- Added first-party sync/async SDK clients with reusable `requests.Session` / `httpx.AsyncClient` transports.
- Added context manager ergonomics (`with` / `async with`) and typed exception hierarchy for deterministic error handling.
- Added mem0-style aliases (`Memory`, `AsyncMemory`) to reduce migration friction for existing users.

**Ecosystem impact:**
- Lower overhead for high-frequency programmatic integrations.
- Cleaner integration path for async agent runtimes.
- Better operability via typed connection/API error semantics.

### Ingestion safety + fail-open parsing (high ROI, correctness + operability)

**Issue found:** Phase 3B ingestion gap prevented controlled parsing of heterogeneous sources and increased risk of parser-coupled pipeline failure.

**Upgrade implemented:**
- Added feature-gated multi-source ingestion package (`muninn/ingestion`) with parser adapters for `txt/md/json/csv/tsv/html` and optional `pdf/docx`.
- Added source-level fail-open behavior: parser failures, missing paths, and oversize files are isolated per source without aborting the full run.
- Added provenance-rich chunk metadata (`source_path`, `source_type`, `source_sha256`, byte size, chunk offsets/count).
- Added strict chunking invariants (`overlap < chunk_size`, minimum chunk length) and bounded file-size controls.

**Ecosystem impact:**
- Reduces ingestion blast radius by turning source/parser failures into auditable partial failures.
- Improves forensic and dedup workflows through deterministic source checksums + chunk offset metadata.
- Creates a direct path for safe MCP/SDK ingestion automation while preserving local-first guarantees.
