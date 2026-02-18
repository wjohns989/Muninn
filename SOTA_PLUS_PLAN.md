# Muninn SOTA+ Implementation Plan

> **Version**: v3.6.1 → v3.10.0
> **Status**: **Phase 13 (Advanced Retrieval & Data Pipeline) In Progress**
> **Current State**: `feature/v3.10.0-advanced-retrieval` implements native ColBERT multi-vector storage and temporal query expansion.

---

## Executive Summary

Muninn has successfully transitioned to **v3.9.0 (Entity Scoping Edition)**. Phases 9-12 implement consolidation integrity, unified security, multi-namespace isolation, and distributed entity scoping with composite IDs. Phase 13 (v3.10.0) extends the retrieval stack with native Qdrant multi-vector MaxSim and NL temporal query expansion.

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

## Validation History

- **Phase 13**: 651 tests passed (100%), 0 failed — native ColBERT multi-vector + temporal query expansion. PR #42 open.
- **Phase 12.1**: All PR review findings resolved (8 fixes applied).
- **Phase 12**: 100% tests passed (Distributed Entity Scoping).
- **Phase 11**: 100% tests passed (Multi-Namespace Integrity).
- **Phase 10**: 100% tests passed (Unified Security).
- **Phase 9**: 100% tests passed (Consolidation, NLI Integrity).
- **Phase 8**: 100% tests passed (ColBERT Efficiency, PLAID).
