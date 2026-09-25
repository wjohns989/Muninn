# Project Index: Muninn

> Generated 2026-09-25 for the `main` tree at `761b92d` (v3.24.0).
> Items tagged **[dev]** exist only on `fix/mcp-updates-and-fixes` (PR #57), which is
> 20 commits / 55 files ahead of `main` and is where active development lands.

Local-first, assistant-agnostic persistent memory for AI agents. One FastAPI
server (`server.py`, port `42069`) owns the stores; MCP, REST, SDK and the Huginn
browser UI are all clients of that single process.

---

## 📁 Layout

```
server.py              FastAPI backend: 47 REST routes (+2 [dev]) + dashboard; owns MuninnMemory
mcp_wrapper.py         Legacy stdio MCP facade → muninn/mcp/*
muninn_standalone.py   Huginn browser-first launcher
tray_app.py            System-tray controller (pystray)
silent_mcp.py          Windowless MCP launcher (Windows)
ingest_history.py      One-shot historical ingestion script
dashboard.html/.css    Huginn UI served at /
muninn/                Core package (see below)
eval/                  Benchmarks, SOTA verdict gates, MCP transport replay/soak
tests/                 101 test files (116 on [dev]); pytest, testpaths=tests
scripts/               build_standalone.py (PyInstaller), benchmark_colbert_quality.py, run_flaky.py
.github/workflows/     benchmark.yml (dry-run gate), transport-incident-replay-gate.yml
```

## 🚀 Entry Points

| Entry | Purpose |
|---|---|
| `python server.py` / `muninn-server` | REST API + UI at `http://localhost:42069` |
| `python mcp_wrapper.py` | stdio MCP (connects to the existing backend) |
| `POST /mcp` **[dev]** | Streamable HTTP MCP on the shared server (`muninn/mcp/http.py`) |
| `python muninn_standalone.py` | Huginn standalone UX |
| `muninn-tray` | Tray app |
| `python -m muninn.cli` | `rotate-token`, `doctor` operational CLI |
| `python -m eval.run_benchmark` | Benchmark harness; `eval.ollama_local_benchmark sota-verdict` for gates |
| `python -m eval.memory_profile_benchmark` **[dev]** | Isolated memory/resource profiling |

## 📦 Core Modules (`muninn/`)

**core/**
- `memory.py` (2.4k L): `MuninnMemory`, the engine that everything calls (add/search/update/delete, scoping, federation hooks)
- `config.py`: typed configs (`EmbeddingConfig`, `VectorConfig`, `GraphConfig`, …) built from env
- `types.py`: `MemoryRecord`, `SearchResult`, `MemoryType`, `Provenance`, `MediaType`
- `security.py`: token auth (`MUNINN_AUTH_TOKEN` / `MUNINN_API_KEY`), `verify_token`, `verify_api_token`
- `recall_trace.py`: explainable per-signal contribution traces
- `feature_flags.py`, `ingestion_manager.py`; `env_loader.py` **[dev]** loads the private project `.env`

**store/**: persistence (single-writer)
- `sqlite_metadata.py` (1.4k L): metadata, profiles, feedback, events
- `vector_store.py` (Qdrant), `multi_vector_store.py` (ColBERT), `graph_store.py` (Kuzu)
- `lock.py`: `StoreLock` cross-process write serialization

**retrieval/**
- `hybrid.py`: `HybridRetriever`, 5-signal RRF (vector, BM25, graph, temporal, goal/chain)
- `bm25.py`, `reranker.py` (FastEmbed cross-encoder; **[dev]** default `jina-reranker-v1-turbo-en`)
- `weight_adapter.py`: SNIPS adaptive weights; `temporal_parser.py`: NL time ranges
- `colbert_index.py`, `scout.py` + `synthesis.py` (agentic `hunt`), `benchmark_adapter.py`

**scoring/**: `importance.py` (recency/frequency/centrality/novelty/provenance/retrieval), `elo.py` (Elo → half-life), `entropy.py`

**consolidation/**: `daemon.py` (decay, merge, promote, integrity phases), `merge.py`, `promote.py`

**extraction/**: `pipeline.py` (rule-based → Instructor/Ollama), `rules.py`, `instructor_extractor.py`, `models.py`, `temporal_synthesis.py`, `vision_adapter.py`, `audio_adapter.py`

**ingestion/**: `pipeline.py` (fail-open multi-source), `parser.py`, `sandbox.py` + `_parser_subprocess.py` (subprocess-isolated binary parsing), `discovery.py` (legacy assistant chats), `periodic.py`, `legacy_scheduler.py`

**mcp/**: modular MCP server
- `definitions.py`: 36 tool schemas (37 on [dev], adding `add_image_memory`)
- `handlers.py`: initialize/list/call + task lifecycle; `tasks.py`, `state.py`, `requests.py` (deadline/retry), `lifecycle.py` (circuit breaker), `deps.py`, `protocol.py`, `metrics.py`, `utils.py`, `server.py`
- `http.py`, `sse.py` **[dev]**: bounded Streamable HTTP and legacy SSE transports

**Other packages**
- `mimir/`: interop relay to Claude Code / Codex / Gemini CLIs (`relay.py`, `routing.py`, `policy.py`, `reconcile.py`, `store.py`, `api.py`, `adapters/`)
- `advanced/`: `temporal_kg.py` (bi-temporal KG), `cross_agent.py` (federation), `colbert.py`
- `optimization/`: `distillation.py`, `clustering.py` (leader-follower `VectorClusterEngine`), `foraging.py`, `surgeon.py`
- `reasoning/omission.py`: gap detection; `goal/compass.py`: goal continuity; `chains/`: memory-chain detect/retrieve
- `conflict/` (NLI detector + resolver), `dedup/semantic_dedup.py`, `observability/otel_genai.py`
- `sdk/client.py`: `MuninnClient` / `AsyncMuninnClient`; `platform.py`: data/config dirs; `cli.py`
- `media/image_memory.py` **[dev]**: content-addressed managed image store

## 🔧 MCP Tools (36 on main)

`add_memory` `search_memory` `hunt_memory` `get_all_memories` `update_memory` `delete_memory`
`delete_all_memories` `set_project_instruction` `set/get_project_goal` `set/get_user_profile`
`get/set_model_profiles` `get_model_profile_events/alerts` `export/import_handoff`
`record_retrieval_feedback` `ingest_sources` `discover/ingest_legacy_sources`
`*_periodic_ingestion` (4) `detect_information_gaps` `trigger_distillation` `correct_fact`
`forage_knowledge` `get_temporal_knowledge` `*federation*` (4) `mimir_relay`, plus `add_image_memory` on [dev]

## ⚙️ Configuration

- `pyproject.toml`: package `muninn-mcp` 3.24.0, Python ≥3.10; extras: `instructor`, `tray`, `service`, `dev`, `observability`, `conflict`, `ingestion`, `sdk`, `all`
- `config.template.yaml`, `docker-compose.yml`, `Dockerfile`, `requirements.txt`, `uv.lock`
- Env vars: see README "Key environment variables" (`MUNINN_*`)
- Lint: ruff (py310, line 120, `E,F,W,I`)

## 📚 Documentation

| Doc | Status |
|---|---|
| `README.md` | Current. [dev] adds the HTTP topology, image memory, and resource env vars |
| `HANDOFF_SOTA_READY.md` **[dev]** | **Latest handoff/roadmap** (v3.24.0 → next steps) |
| `SOTA_PLUS_PLAN.md` | Phase 10–26 history; all phases marked done |
| `GEMINI.md` | Agent development conventions |
| `HANDOFF.md`, `REMEDIATION_HANDOFF.md`, `FINAL_REMEDIATION_REPORT.md`, `CHANGELOG_REMEDIATION.md`, `SESSION_COMPLETE.md`, `PR_UPDATE.md` | Historical (Feb 2026) |
| `docs/` (branch `feature/sota-plus-archive` only) | `SOTA_EXPERIMENTAL_REVIEW.md`, `plans/` (Phase 4–5 designs), `MUNINN_2026_VISION_AND_ROADMAP.md` |
| `eval/README.md`, `CITATIONS.md` | Benchmark docs, references |

## 🧪 Tests

- `pytest` from repo root; `tests/optimization/`, `tests/reasoning/` subpackages
- Last recorded: 1477 passed / 8 skipped on [dev] (PR #134); CI runs only the benchmark dry-run and the transport replay gate, not the full suite; the replay gate's two pytest calls end in `|| true`, so their failures can't fail CI

---

## 📌 Status Snapshot (2026-09-25)

### Current plan (`HANDOFF_SOTA_READY.md` + `docs/SOTA_EXPERIMENTAL_REVIEW.md`)

| Item | State in code ([dev]) |
|---|---|
| Fix Huginn UI ingestion regression (Ollama / `global_user` scoping) | Not verified fixed; UI and server still default to `user_id: global_user` |
| P0: SNIPS → importance loop | **Partial**: `retrieval_utility` weight (0.10) and Elo-scaled half-life exist; the review's "signal-specific multipliers into decay" is not done |
| P0: Centrality baseline for entity-free memories | **Open**: `centrality` still defaults to `0.0` (a 0.20-weight penalty) |
| P1: CoALA session inhibition in `HybridRetriever` | **Open**: no inhibition code |
| Distillation clustering TODO | Implemented (`VectorClusterEngine`); the TODO comment in `distillation.py` is stale |

The Aug 2026 work (PRs #134–#139, merged into the [dev] branch) was an off-roadmap
operational pass: memory/resource bounds, Streamable HTTP MCP, image memories,
`.env` loading, Codex config repair, and the Jina Turbo reranker.

### Pull requests (77 open)

**Integration PR:** #57 `fix/mcp-updates-and-fixes` → `main`: CI green (2/2), carries all Aug 2026 work. Merge this first; most other PRs target a stale `main`.

**Conflict with the [dev] branch (17):** #59, #60, #67, #68, #73, #77, #97, #105, #115, #116, #117, #119, #125, #126, #127, #128, #129

**Duplicate clusters (keep ≤1 each):**
- JSONL bulk parse: #118, #133
- SQLite index `executescript`: #117, #119 (+ Mimir DDL #115)
- Chain-link UNWIND batching: #128, #129
- Batch delete / N+1: #59, #73, #77
- Clustering: #68, #97, #131 are superseded by `VectorClusterEngine`; #94 (remove stale TODO) is the right fix. #131 also adds `scikit-learn` as a core dependency
- `mcp_wrapper.py` unused imports: #60, #64, #71, #84, #91
- `server.py` unused imports: #78, #82
- `run_benchmark.py` refactor: #87, #90, #95
- Platform/docker tests: #92, #96, #102, #104, #107
- Rule-based extraction tests: #111, #113, #122, #124
- Handoff SDK tests: #62, #75
- Blocking sleep → backoff/asyncio: #126, #132

**Security (review individually):** #65, #66, #76 (SQL construction in the SQLite store), #67 (`silent_mcp.py`, conflicts with the [dev] rewrite), #108 (CORS)

**Likely superseded:** #61 "modern MCP standards" (by the #134 Streamable HTTP work)
