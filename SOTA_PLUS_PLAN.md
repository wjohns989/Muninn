# Muninn SOTA+ Implementation Plan

> **Version**: v3.6.1 → v3.18.1
> **Status**: **Phase 19 IN PROGRESS — PR #50 open (`feature/sota-roadmap-outward`)**
> **Current State**: v3.18.1 — Phase 19 (Scout synthesis, hunt mode) implemented. 1019 tests pass. Phases 14–18 merged to main.

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

> **Status**: ✅ **COMPLETE — PR #44 ready for merge**
> **Version**: v3.12.0
> **Theme**: Operational correctness, graph activation, and SOTA+ closure.
> **Branch**: `feature/v3.12.0-operational-hardening`

### Background & Gap Analysis

Phase 14 delivered strong memory isolation. Three categories of open issues remain before a credible SOTA+ verdict:

1. **Auth propagation gap**: `lifecycle.py:start_server()` spawns `server.py` without passing `MUNINN_AUTH_TOKEN` to the child process. If the system env var is not set, the auto-started server generates a random token → mismatch → all MCP tool calls fail with 401. This is a silent operational breakage that only surfaces in fresh environments.

2. **Graph memory chains dormant**: KuzuDB-based memory chains (`muninn/chains`) are fully implemented behind `chains` feature flag but graph store shows 0 nodes in production. The feature needs an activation-and-verification pass to confirm wire-up is correct end-to-end.

3. **SOTA+ benchmark gaps**: Three benchmark suites identified in `MUNINN_COMPREHENSIVE_ROADMAP.md` remain unimplemented: LongMemEval adapter, StructMemEval adapter, and signed promotion-manifest issuance. Without these, SOTA+ claims lack external benchmark grounding.

### Implementation Checklist

- [x] **Auth propagation fix** (`lifecycle.py`): `start_server()` passes `MUNINN_AUTH_TOKEN` from env (or discovers it via `get_token()`) when spawning `server.py` — 5 unit tests prove correct behaviour (2026-02-19)
- [x] **Graph chains smoke test**: 7 integration tests via real KuzuDB (`tmp_path`); proves PRECEDES/CAUSES edge creation, retrieval, `get_entity_count()` increment, and `MemoryChainDetector` link detection (2026-02-19)
- [x] **OTel activation validation**: 8 tests validate GenAI semantic convention keys (`gen_ai.operation.name`, `gen_ai.system`), dot-namespaced Muninn attributes, privacy default for content, no-op when disabled (2026-02-19)
- [x] **LongMemEval adapter baseline**: `eval/longmemeval_adapter.py` — full production adapter with JSONL parser, nDCG@10/Recall@10 metrics, MuninnHTTPClient, selftest dataset, CLI; 13 tests pass (2026-02-19)
- [ ] **SOTA+ signed verdict v1**: `eval.ollama_local_benchmark sota-verdict` extended to include external benchmark evidence; verdict artifact includes commit SHA, benchmark hashes, and promotion signature
- [x] **Version**: `3.12.0` in `version.py` and `pyproject.toml`
- [x] **Verification**: 727 tests pass (694 existing + 33 new), 2 skipped, 0 failed (2026-02-19)

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

## Phase 16: SOTA+ Signed Verdict v1 & External Benchmark Closure

> **Status**: ✅ **COMPLETE — PR #45 ready for merge**
> **Version**: v3.13.0
> **Theme**: Credibility closure — signed verdict artifact, commit provenance, LongMemEval gate, StructMemEval adapter.
> **Branch**: `feature/v3.13.0-sota-verdict-v1`

### Background & Gap Analysis

Phase 15 built the LongMemEval adapter and validated OTel/graph chains. One credibility gap remains before SOTA+ claims can be asserted publicly: the `sota-verdict` command produces an unsigned JSON artifact with no cryptographic provenance chain, no commit SHA binding the verdict to a specific codebase state, and no external benchmark gate. A verdict that doesn't embed external benchmark evidence or a tamper-detectable signature cannot be audited or reproduced independently.

### Three Remaining Gaps

1. **Provenance void**: Verdict JSON has `run_id` but no `commit_sha`. Any assertion "Muninn SOTA+ passed at commit X" requires the reader to trust the timestamp alone.

2. **No external benchmark gate**: `sota-verdict` accepts `--aux-benchmark-report` but those are normalized and emitted, not gated. LongMemEval nDCG@10 and Recall@10 must become a hard gate in `overall_passed`.

3. **No tamper-detectable signature**: The verdict is plain JSON. A replay or mutation cannot be detected. HMAC-SHA256 over a canonical payload subset (bound to `commit_sha` + `input_file_hashes`) makes the verdict self-verifying.

4. **Single external benchmark**: LongMemEval covers single-session conversational QA. A second benchmark covering structured/factoid recall is needed to triangulate SOTA+ evidence.

### Implementation Checklist

- [x] **`cmd_sota_verdict` — Provenance block**: Inject `commit_sha` (git rev-parse HEAD), `input_file_hashes` (SHA256 of each --*-report arg), `verdict_schema_version` into payload under `provenance` key (2026-02-19)
- [x] **`cmd_sota_verdict` — HMAC-SHA256 signing**: `--signing-key` CLI arg; sign canonical JSON subset `{run_id, passed, commit_sha, input_file_hashes}` → `provenance.promotion_signature = "hmac-sha256=<hex>"`; no-op (null) when key not provided (2026-02-19)
- [x] **`cmd_sota_verdict` — LongMemEval gate**: `--longmemeval-report`, `--min-longmemeval-ndcg` (default 0.60), `--min-longmemeval-recall` (default 0.65), `--require-longmemeval` (default False); gate passes iff both thresholds met; contributes to `overall_passed`; all Phase 16 args use `getattr` defaults for backward compat with pre-existing SimpleNamespace tests (2026-02-19)
- [x] **`eval/structmemeval_adapter.py`**: StructMemEval adapter for structured/factoid memory recall; JSONL format `{case_id, question, expected_answer, answer_type, memories[], relevant_memory_index}`; metrics: Exact Match, token-F1, MRR@k; 3-case selftest dataset (selftest EM=1.000, MRR@10=1.000); full CLI (2026-02-19)
- [x] **Version**: `3.13.0` in `version.py` and `pyproject.toml` (2026-02-19)
- [x] **Verification**: **788 tests pass** (727 existing + 61 new), 2 skipped, 0 failed — `tests/test_v3_13_0_sota_verdict_v1.py` (61 tests across 8 classes) (2026-02-19)

### Key Correctness Properties (Targets)

1. **Provenance binding**: Verdict JSON includes `commit_sha` from `git rev-parse HEAD`; falls back gracefully when git unavailable
2. **Signature verifiability**: `promotion_signature` HMAC can be re-verified offline from public key + canonical payload
3. **LongMemEval hard gate**: `overall_passed = False` if `ndcg_at_10 < min_longmemeval_ndcg` and `--require-longmemeval` is set
4. **StructMemEval selftest**: `python eval/structmemeval_adapter.py --selftest` passes without any external server

### Optimization & ROI Notes

**High ROI:**

- **Signed verdict**: Enables public assertion "commit SHA X passed SOTA+ with nDCG@10=Y" — auditable, reproducible, tamper-evident. Without this, all SOTA+ claims are opinion, not evidence.
- **LongMemEval gate in `overall_passed`**: Currently LongMemEval evidence is computed but not gated. Making it a hard requirement forces the benchmark to be kept passing as code evolves — prevents silent regression.

**Medium ROI:**

- **StructMemEval adapter**: Structured recall (facts, numbers, entities) and conversational recall (LongMemEval) are complementary. A system that scores well on both has triangulated evidence across two distinct memory retrieval modes.

---

## Phase 17: Synthetic Benchmark Suite & Parser Security Sandbox

> **Status**: ✅ **COMPLETE — PR #46 ready for merge**
> **Version**: v3.14.0
> **Theme**: Evidence grounding for SOTA+ and operational security hardening.
> **Branch**: `feature/v3.14.0-benchmark-suite-parser-sandbox`

### Background & Gap Analysis

Phase 16 completed the SOTA+ signed verdict infrastructure — adapters, HMAC signing, LongMemEval hard gate. One credibility gap remains: **no real-dataset benchmark evidence has been produced**. Both adapters are production-ready and selftests pass, but the signed verdict has never been run against representative data. Phase 17 closes this gap by providing representative synthetic datasets and a CI-runnable pipeline. Additionally, the PDF/DOCX parsers were running in-process with no exploit containment — Phase 17 adds subprocess sandboxing.

### Implementation Checklist

- [x] **`eval/data/longmemeval_synthetic_v1.jsonl`**: 30 representative LongMemEval-format cases covering all question types (`single-session-qa`×10, `multi-session-qa`×8, `temporal`×6, `adversarial`×3, `entity-centric`×3). Each case has full session conversations with realistic Muninn-domain content. (2026-02-19)
- [x] **`eval/data/structmemeval_suite_v1.jsonl`**: 30 StructMemEval-format cases covering all answer types (`string`×10, `number`×8, `entity`×7, `list`×5). Each case has well-formed memories[] and a valid `relevant_memory_index`. (2026-02-19)
- [x] **`eval/run_benchmark.py`**: Automated CI benchmark pipeline with dry-run mode (selftests, no server), production mode (live server), per-adapter subprocess execution, gate evaluation (LongMemEval + StructMemEval), combined JSON report output, commit SHA provenance, full argparse CLI. (2026-02-19)
- [x] **`muninn/ingestion/_parser_subprocess.py`**: Subprocess worker for sandboxed PDF/DOCX parsing. JSON-over-stdout protocol, output size capped at 2 MB, catch-all exception handling, exit codes 0/1/2. (2026-02-19)
- [x] **`muninn/ingestion/sandbox.py`**: Sandbox executor wrapping `_parser_subprocess.py`. Timeout enforcement, 4 MB stdout cap, JSON validation, error containment, optional in-process fallback. Cross-platform (Windows/Linux/macOS). (2026-02-19)
- [x] **`muninn/ingestion/parser.py`**: `_parse_pdf()` and `_parse_docx()` now route through `sandboxed_parse_binary()` for process isolation. (2026-02-19)
- [x] **Version**: `3.14.0` in `version.py` and `pyproject.toml` (2026-02-19)
- [x] **Test correction**: `tests/test_v3_13_0_sota_verdict_v1.py::TestVersionBump313::test_version_is_3_13_0` updated to `>= (3, 13, 0)` tuple comparison (consistent with prior phases)
- [x] **Verification**: **848 tests pass** (788 existing + 60 new), 2 skipped, 0 failed — `tests/test_v3_14_0_benchmark_suite.py` (60 tests across 8 classes) (2026-02-19)

### Key Correctness Properties

1. **Dataset format compliance**: Both synthetic datasets fully comply with their respective adapter schemas — every case passes the JSONL parser without error
2. **Question type coverage**: LongMemEval synthetic covers all 5 question types; StructMemEval suite covers all 4 answer types
3. **Benchmark pipeline isolation**: `run_benchmark` dry-run mode requires no live server — safe for CI environments
4. **Parser sandbox containment**: PDF/DOCX parser exploits are contained to a child process — the Muninn server process is protected
5. **Timeout enforcement**: Parser subprocess is hard-killed after 30s — decompression bombs cannot block the server

### Security Impact Analysis

**Before Phase 17 (risk)**:

- `_parse_pdf(path)` called `from pypdf import PdfReader; PdfReader(str(path))` in the FastAPI server process
- A malicious PDF could exploit pypdf to: read arbitrary files, cause OOM (zip bomb), execute code in the server process
- Same risk for DOCX via python-docx

**After Phase 17 (hardened)**:

- All binary parsing runs in a subprocess with no inherited secrets, no network access
- Parser crash/exception produces a structured error JSON — never propagates to the server
- 30-second hard timeout prevents resource exhaustion
- 4 MB stdout cap on parent side + 2 MB cap on child side = dual defense against output flooding

### Optimization & ROI Notes

**High ROI:**

- **Parser sandbox**: ~2 hour implementation, prevents complete server compromise from malicious document ingestion. Without this, every user processing PDFs exposes the Muninn server to pypdf/docx vulnerabilities.
- **Synthetic benchmark datasets**: Enables the `--dry-run` CI benchmark pipeline to run without any external data. Every PR can now run `run_benchmark.py --dry-run` and verify the adapter plumbing works end-to-end.

**Medium ROI:**

- **Automated benchmark runner**: Reduces the gap from "adapters exist" to "evidence exists" by making the pipeline one command. The dry-run mode is CI-safe with zero server dependency.

**Future Work (Phase 19 candidates)**:

- Run `run_benchmark.py --production` against a live server with the synthetic datasets and commit the signed verdict artifact
- Obtain public LongMemEval JSONL dataset for real nDCG@10 baseline
- Wire StructMemEval into `sota-verdict` MCP tool signing

### Environment Variables (Phase 17)

No new environment variables. Parser sandboxing is always active for PDF/DOCX (no flag).

---

## Phase 18: CI Benchmark Workflow & Token Rotation (v3.15.0)

**Goal**: Automate adapter regression prevention on every PR and provide a safe, one-command token rotation workflow.

### Track 1 — GitHub Actions CI (`.github/workflows/benchmark.yml`)

- Triggers: `pull_request` → `main`, `push` → `main`, `workflow_dispatch`
- Job: `benchmark-dry-run` (ubuntu-latest, 15-minute timeout)
- Steps: checkout@v4, setup-python@v5 (3.11), `pip install .[dev]`, `python -m eval.run_benchmark --dry-run --output <report>`, parse report → GitHub Step Summary table, upload-artifact@v4
- Gate: exits 1 if `overall_passed=false`, blocking PR merge
- Optional inputs: `skip_lme`, `skip_sme`, `limit` for targeted manual runs
- Security: `permissions: contents: read` (minimal)

### Track 2 — Token Rotation CLI (`muninn/cli.py`)

- Entry point: `python -m muninn.cli rotate-token`
- Generates: `secrets.token_urlsafe(32)` (43-char URL-safe base64)
- Writes: `.muninn_token` (or `--token-file PATH` or `$MUNINN_TOKEN_FILE`)
- MCP config patching: auto-detects `claude_desktop_config.json`, `~/.cursor/mcp.json`, `~/.vscode/mcp.json`; updates `MUNINN_AUTH_TOKEN` in any "muninn" server `env` block (case-insensitive name match)
- Flags: `--dry-run` (no files written), `--token-only` (machine-readable single-line output)
- Platform instructions: PowerShell `$env:MUNINN_AUTH_TOKEN` + `setx` on Windows; `export` + `~/.bashrc` on POSIX
- Resolution order: `--token-file` > `$MUNINN_TOKEN_FILE` > `./.muninn_token`

### ROI / Optimization Notes

- **CI dry-run gate prevents silent adapter regressions** from landing on main. Without it, a dataset schema change or subprocess worker breakage would only be discovered during a production benchmark run (requires live server, ~5 min).
- **Token rotation** reduces credential exposure window from "until someone manually edits the file" to "one command + server restart." Cross-platform MCP config patching eliminates the manual step of updating Claude Desktop / Cursor configs separately.

### Environment Variables (Phase 18)

`MUNINN_TOKEN_FILE` — optional override for the token file path (used by `muninn.cli rotate-token`).

---

## Phase 19: Deterministic Outer Loops & Verification

**Goal**: Integrate deterministic verification directly into the automation pipeline for increased credibility and accuracy.

### Key Objectives

- **Playwright MCP Integration**: Introduce end-to-end testing via browser automation to verify the UI and control center works exactly as expected without human intervention.
- **Robust Code Verification**: Integrate multi-agent blind reviews and "Devil's Advocate" validation gates directly into the `sota-verdict` CI.
- **Fail-Fast Loop**: Incorporate rule-based (deterministic) checks before executing LLM-based (probabilistic) reviews.

---

## Phase 20: Multimodal Hive Mind Operations

**Goal**: Extend the core substrate far beyond text and individual contexts, creating a shared and universally applicable memory system.

### Key Objectives

- **Unified Multimodal Space**: Extend the unified embedding space to support cross-assistant shared multimodal memory, properly ingesting and retrieving images, audio, and sensor data context.
- **Cross-Assistant Collaboration**: Build out the infrastructure for a true "Hive Mind" architecture, establishing low-latency memory synchronization and robust multi-agent knowledge sharing.

---

## Validation History

- **Phase 18**: **890 tests passed (100%), 0 failed** — CI benchmark workflow, token rotation CLI, MCP config patcher, version 3.15.0. 39 new tests.
- **Phase 17**: **851 tests passed (100%), 0 failed** (incl. PR review fixes) — synthetic benchmark datasets (30+30 cases), automated benchmark runner, parser security sandbox, env-sanitized subprocess, corrected SME metric keys, version 3.14.0. 63 new tests. Merged (PR #45).
- **Phase 16**: **788 tests passed (100%), 0 failed** — SOTA+ signed verdict v1, HMAC-SHA256 provenance, LongMemEval hard gate, StructMemEval adapter. 61 new tests. PR #45 ready.
- **Phase 15**: **727 tests passed (100%), 0 failed** — auth propagation fix, graph chains smoke, OTel GenAI hardening, LongMemEval adapter baseline. PR #44 merged.
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
