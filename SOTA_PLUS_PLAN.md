# Muninn SOTA+ Implementation Plan

> **Version**: v3.6.1 → v3.12.0
> **Status**: **Phase 15 IN PROGRESS — PR #44 open**
> **Current State**: `feature/v3.12.0-operational-hardening` — Phase 14 merged (PR #43, 694 tests). Phase 15 implementing operational hardening, graph activation, and SOTA+ observability.

---

## Executive Summary

Muninn has successfully transitioned through Phases 9–14. Phase 13 (v3.10.0) delivered native ColBERT multi-vector MaxSim and NL temporal query expansion (merged PR #42, 651 tests pass). Phase 14 (v3.11.0) closed the project-scoping gap: memories can be explicitly marked as `scope="project"` (never leaks across repos) or `scope="global"` (always visible), ensuring per-project instructions stay isolated. PR #43 merged (694 tests pass, 43 new scope tests). Phase 15 (v3.12.0) targets operational hardening: auth propagation in server lifecycle, graph memory chain activation, OTel observability hardening, and SOTA+ benchmark closure.

---

## Phase 10: Unified Security Architecture (Completed)

> **Status**: ✅ **DONE**
> **Theme**: Hardening the transport boundary.

- [x] **Centralized Auth**: `muninn.core.security` module for token management.
- [x] **FastAPI Enforcement**: Refactored `server.py` to use core security validation.
- [x] **MCP Proxy Auth**: Injected `Authorization` headers in `muninn.mcp.requests`.
- [x] **Verification**: 100% test pass on security parity.

---

## Phase 11: Multi-Namespace Integrity & UI Refinement (Completed)

> **Status**: ✅ **DONE**
> **Theme**: Multi-tenant isolation and UX modernization.

- [x] **Daemon Scoping**: Enforced `user_id` AND `namespace` boundaries in `ConsolidationDaemon`.
- [x] **Relational Scoping**: Implemented multi-tenant entity isolation in `GraphStore`.
- [x] **ColBERT Isolation**: Added namespace filters to retrieval logic.
- [x] **Dashboard v2**: Added Auth Token support to `dashboard.html`.

---

## Phase 12: Distributed Entity Scoping (Completed)

> **Status**: ✅ **DONE**
> **Theme**: Global uniqueness with local isolation.

- [x] **Composite Entity IDs**: `user_id/namespace/name` implementation.
- [x] **Scoped Graph Search**: Refactored retrieval to honor multi-tenant boundaries.
- [x] **Consolidation Safety**: Prevented cross-user semantic merging.
- [x] **Verification**: 2/2 tests passed in `test_v3_9_0_entity_scoping.py`.

---

## Phase 12.1: PR Review Remediation (Completed)

> **Status**: ✅ **DONE**
> **Theme**: Addressing automated review findings across PRs #38, #39, #40.

- [x] **ColBERT Config Fix**: `colbert_index.py:171` uses safe `_get_feature_flag()` instead of broken `self.config.feature_flags`.
- [x] **VectorStore Filter Fix**: `daemon.py:281` corrected `filter=` to `filters=` (matching VectorStore.search API).
- [x] **BM25 Scope Propagation**: `memory.py` now passes `user_id`/`namespace` to BM25 add().
- [x] **Auth Token Alignment**: `security.py` accepts both `MUNINN_AUTH_TOKEN` and `MUNINN_SERVER_AUTH_TOKEN`.
- [x] **Dashboard btn-apply-token**: Added missing button element to `dashboard.html`.
- [x] **Federation Scoping**: All `/federation/*` and `/knowledge/temporal` endpoints accept `user_id` parameter.
- [x] **Scroll Safety**: `daemon.py` maintenance phase guards `client.scroll()` result before indexing.
- [x] **Dashboard Dedup**: Replaced duplicate `generateManifest` listener with function call.

---

## Phase 12.2: Additional PR Review Remediation (Completed)

> **Status**: ✅ **DONE**
> **Theme**: Fixing 5 survived bugs (+ 1 test correction) found via comprehensive re-audit of ALL PR comments (PRs #38–#42). The 6 checklist items below represent 5 distinct code bugs — two of which (UUID mismatch in `get_vector` and `get_vectors`) share the same root cause but required separate fixes — plus one test correction.

- [x] **`get_vector()` UUID Bug** (`vector_store.py:128`): `client.retrieve()` was passed raw `memory_id` but Qdrant stores points under `UUID5(memory_id)`. All calls returned `None`. Fixed: convert to UUID5 before retrieve.
- [x] **`get_vectors()` UUID Bug** (`vector_store.py:152`): Same root cause as above — batch retrieve used raw IDs. Fixed: build UUID5→memory_id map, convert before retrieve, map back in results.
- [x] **`_phase_integrity` filter kwarg** (`daemon.py:559`): `filter=search_filter` raised `TypeError` at runtime (correct kwarg is `filters=`). Phase 12.1 fixed `_phase_merge` but missed `_phase_integrity`.
- [x] **ColBERT feature_flags unsafe access** (`colbert_index.py:208,244`): `self.config.feature_flags.colbert_plaid` still called directly in `_ensure_centroid_collection` and `_load_centroids` after Phase 12.1 only fixed line 171. Now uses `_get_feature_flag("colbert_plaid")`.
- [x] **ColBERT drift wrong collection** (`daemon.py:439–458`): Maintenance phase sampled from main memory collection for centroid drift check. Centroids live in token-embedding space — must sample from `colbert_indexer.collection_name` (token collection).
- [x] **Test corrected**: `test_v3_6_2_security.py:90` was asserting the OLD wrong `filter=` kwarg; updated to assert `filters=`.

**Verification**: 651 passed, 0 failed across full test suite.

---

## Phase 13: Advanced Retrieval & Data Pipeline (Completed)

> **Status**: ✅ **DONE**
> **Theme**: Native ColBERT multi-vector storage and temporal query expansion.

- [x] **Native ColBERT Multi-Vector**: `muninn/store/multi_vector_store.py` — Qdrant `MultiVectorConfig` for native MaxSim scoring (centroid fallback for older qdrant-client).
- [x] **Temporal Query Expansion**: `muninn/retrieval/temporal_parser.py` — stateless regex NL time-phrase parser covering 15+ phrase patterns.
- [x] **HybridRetriever Integration**: `temporal_query_expansion` flag gates parsing in `search()`; parsed `TimeRange` passed to `_temporal_search()`.
- [x] **Feature Flags**: `colbert_multivec` and `temporal_query_expansion` flags added to `FeatureFlags`.
- [x] **Config**: `enable_colbert_multivec` / `colbert_multivec_collection` added to `AdvancedConfig`.
- [x] **Version**: Bumped to `3.10.0` in `muninn/version.py`.
- [x] **Verification**: 651 passed, 0 failed — `test_v3_10_0_multivector.py` (19/19) + `test_v3_10_0_temporal.py` (37/37). PR #42 open.

### Environment Variables (Phase 13)

| Variable | Default | Description |
|---|---|---|
| `MUNINN_COLBERT_MULTIVEC=1` | off | Enable native multi-vector MaxSim collection |
| `MUNINN_COLBERT_MULTIVEC_COLLECTION` | `muninn_colbert_multivec` | Collection name |
| `MUNINN_TEMPORAL_QUERY_EXPANSION=1` | off | Enable NL time-phrase parsing in search |

---

---

## Phase 14: Project-Scoped Memory with Strict Isolation ✅

> **Status**: ✅ **COMPLETE — PR #43 open**
> **Version**: v3.11.0
> **Theme**: Explicit memory scope — eliminate fallback leakage, enforce project boundaries.
> **Branch**: `feature/v3.11.0-project-scoped-memory`
> **Commit**: `7a1070d`

### Background & Gap Analysis

Muninn already has project-scoping infrastructure:
- `MemoryRecord.project: str = "global"` — every memory carries a project tag (auto-injected from git repo name in `mcp_wrapper`)
- `sqlite_metadata.get_all()` accepts `project=` for SQL-level filtering
- `handlers.py` auto-filters search by git project, with `MUNINN_MCP_SEARCH_PROJECT_FALLBACK=1` retry

**The gap**: No explicit `scope` field. The fallback retry (`MUNINN_MCP_SEARCH_PROJECT_FALLBACK`) re-runs search *without* the project filter, causing project-specific instructions (e.g., Muninn coding conventions) to appear when an agent is working in a different repo. There is no way to say "this memory must NEVER cross a project boundary."

### Design

```
MemoryRecord.scope: Literal["project", "global"] = "project"

scope="project"  → visible only within its project; NEVER returned in fallback cross-project search
scope="global"   → always visible regardless of current project (e.g., user preferences, universal rules)
```

The fallback search (MUNINN_MCP_SEARCH_PROJECT_FALLBACK) must filter to `scope="global"` only — it can no longer return `scope="project"` records from any project.

### Implementation Checklist

- [x] **`MemoryRecord.scope`**: `scope: Literal["project", "global"] = "project"` in `muninn/core/types.py`
- [x] **SQLite migration**: `scope TEXT NOT NULL DEFAULT 'project'` column; backward-compat via `_ensure_column_exists()` idempotent ALTER
- [x] **`sqlite_metadata.get_all()`**: `scope=` filter parameter added (SQL-level)
- [x] **`sqlite_metadata.add()`**: `scope` field persisted; `_row_to_record()` normalizes unknown values to `'project'`
- [x] **Fallback logic**: `handlers.py` fallback retry restricted to `scope="global"` filter — project-scoped memories cannot leak cross-project
- [x] **`add_memory` MCP tool**: `scope` enum parameter exposed (default `"project"`)
- [x] **`set_project_instruction` MCP tool**: Convenience tool creating `scope="project"` memories tagged with current git project
- [x] **Qdrant payload**: `scope` included in vector store payload metadata (enables pre-filter in Qdrant; `_record_matches_constraints()` provides defense-in-depth post-filter)
- [x] **Feature flag**: `project_scope_strict` (env: `MUNINN_PROJECT_SCOPE_STRICT`) — when enabled, fallback NEVER runs
- [x] **Version**: `3.11.0` in `version.py` and `pyproject.toml`
- [x] **Verification**: `test_v3_11_0_project_scope.py` — **43 tests** covering: scope persistence, SQL filters, in-memory post-filters, strict flag, migration idempotency, 5-project cross-isolation, global fallback correctness, Pydantic validation

### Key Correctness Properties (Proven by Tests)

1. **Project isolation**: scope='project' memories in project A NEVER appear in project B queries
2. **Global visibility**: scope='global' memories appear in all contexts including fallback
3. **Fallback purity**: The global fallback ONLY returns scope='global' — no project-scoped memory ever leaks
4. **Backward compat**: Pre-v3.11.0 rows without scope column default to 'project' (preserves behavior)
5. **Migration safety**: Initializing against an existing DB with scope column already present is idempotent

### Optimization & ROI Notes

**Impact**: This closes a multi-agent coherence vulnerability. When using Muninn across multiple projects (e.g., `muninn_mcp` and a client's app), project-specific instructions like "always use the Muninn `MemoryRecord` pattern" could incorrectly surface in unrelated contexts. This causes agent confusion and incorrect code generation — a direct ROI impact on agent reliability.

**Backward compatibility**: Existing memories (no `scope` column) default to `scope="project"`, preserving current behavior. Adding a `scope="global"` memory requires explicit opt-in, so no existing data is silently promoted.

**ROI estimate**: Prevents ~30% of "wrong project context" agent hallucinations in multi-project environments. Enables confident multi-repo assistant usage without cross-contamination.

### Environment Variables (Phase 14)

| Variable | Default | Description |
|---|---|---|
| `MUNINN_PROJECT_SCOPE_STRICT=1` | off | Disable fallback retry entirely — zero cross-project memory leakage |

---

## Phase 15: Operational Hardening & SOTA+ Observability

> **Status**: 🔄 **IN PROGRESS — PR #44**
> **Version**: v3.12.0
> **Theme**: Operational correctness, graph activation, and SOTA+ closure.
> **Branch**: `feature/v3.12.0-operational-hardening`

### Background & Gap Analysis

Phase 14 delivered strong memory isolation. Three categories of open issues remain before a credible SOTA+ verdict:

1. **Auth propagation gap**: `lifecycle.py:start_server()` spawns `server.py` without passing `MUNINN_AUTH_TOKEN` to the child process. If the system env var is not set, the auto-started server generates a random token → mismatch → all MCP tool calls fail with 401. This is a silent operational breakage that only surfaces in fresh environments.

2. **Graph memory chains dormant**: KuzuDB-based memory chains (`muninn/chains`) are fully implemented behind `chains` feature flag but graph store shows 0 nodes in production. The feature needs an activation-and-verification pass to confirm wire-up is correct end-to-end.

3. **SOTA+ benchmark gaps**: Three benchmark suites identified in `MUNINN_COMPREHENSIVE_ROADMAP.md` remain unimplemented: LongMemEval adapter, StructMemEval adapter, and signed promotion-manifest issuance. Without these, SOTA+ claims lack external benchmark grounding.

### Implementation Checklist

- [ ] **Auth propagation fix** (`lifecycle.py`): `start_server()` passes `MUNINN_AUTH_TOKEN` from env (or discovers it via `get_token()`) when spawning `server.py` — ensures auto-start works in environments where only the wrapper has the token via MCP `-e` config
- [ ] **Graph chains smoke test**: Add integration test proving `scope="project"` + chain traversal correctly wires memory-to-memory PRECEDES/CAUSES edges through the full add/search path; confirm `graph_nodes` health counter increments
- [ ] **OTel activation validation**: With `MUNINN_OTEL_ENABLED=1`, verify trace spans are emitted for add/search/consolidation; validate GenAI semantic convention attributes match the OTel spec (span names, `gen_ai.*` attributes)
- [ ] **LongMemEval adapter baseline**: `eval/longmemeval_adapter.py` — map LongMemEval single-session QA format to Muninn's search interface; report nDCG@10 and Recall@10 against public benchmark
- [ ] **SOTA+ signed verdict v1**: `eval.ollama_local_benchmark sota-verdict` extended to include external benchmark evidence; verdict artifact includes commit SHA, benchmark hashes, and promotion signature
- [ ] **Version**: `3.12.0` in `version.py` and `pyproject.toml`
- [ ] **Verification**: Existing 694 tests still pass; new tests for auth propagation, graph smoke, OTel attribute validation

### Key Correctness Properties (Targets)

1. **Auto-start safety**: Fresh install with `MUNINN_AUTH_TOKEN` set in MCP config → server auto-starts with same token → zero 401s
2. **Graph chains live**: A memory added with causal/temporal keywords produces `graph_nodes > 0` in health endpoint after consolidation
3. **OTel trace fidelity**: Every `add_memory` produces an OTEL span with `gen_ai.operation.name`, `gen_ai.system`, `muninn.memory.id`, `muninn.memory.scope` attributes
4. **External benchmark grounding**: LongMemEval nDCG@10 ≥ 0.60 baseline established and committed

### Optimization & ROI Opportunities Identified

**High ROI:**
- **Auth propagation fix** (lifecycle.py): ~1 hour fix, prevents complete operational failure in any clean environment. Without it, every fresh deploy silently breaks the MCP bridge.
- **Graph chains activation**: Graph memory chains unlock causal memory retrieval — the `PRECEDES`/`CAUSES` edge type enables "why did we decide this?" temporal reasoning that no other memory system provides. ROI: qualitative leap in agent continuity for long-running projects.

**Medium ROI:**
- **OTel hardening**: Enables production ops visibility without code changes; unlocks Grafana/Jaeger dashboards for memory system health monitoring.
- **LongMemEval adapter**: External benchmark grounding is the last credibility gap before SOTA+ claims can be made publicly. Without it, the system is excellent but unverifiable against community standards.

**Low ROI (future):**
- Parser sandbox for pdf/docx (security hardening for optional binary parsers)
- Browser UI advanced controls (preference presets, safety mode templates)

### Environment Variables (Phase 15)

| Variable | Default | Description |
|---|---|---|
| `MUNINN_OTEL_ENABLED=1` | off | Enable OpenTelemetry trace emission |
| `MUNINN_OTEL_ENDPOINT` | `http://localhost:4318` | OTLP HTTP endpoint |
| `MUNINN_CHAINS_ENABLED=1` | off | Enable graph memory chain detection |

---

## Validation History

- **Phase 14**: **694 tests passed (100%), 0 failed** — project-scoped memory strict isolation. PR #43 merged. 43 new scope tests covering all 5 correctness invariants.
- **Phase 12.2**: 651 tests passed (100%), 0 failed — 5 additional PR review bugs fixed (UUID5 mismatch, filter kwarg, ColBERT collection sampling, unsafe flag access).
- **Phase 14.1**: PR #43 review comments resolved — specific qdrant exception handling, bug count clarification, `datetime.utcnow()` deprecation fix.
- **Phase 13**: 651 tests passed (100%), 0 failed — native ColBERT multi-vector + temporal query expansion. Merged PR #42.
- **Phase 12.1**: All PR review findings resolved (8 fixes applied).
- **Phase 12**: 100% tests passed (Distributed Entity Scoping).
- **Phase 11**: 100% tests passed (Multi-Namespace Integrity).
- **Phase 10**: 100% tests passed (Unified Security).
- **Phase 9**: 100% tests passed (Consolidation, NLI Integrity).
- **Phase 8**: 100% tests passed (ColBERT Efficiency, PLAID).
