# Muninn Development Handoff

> **Updated**: 2026-02-20
> **Branch**: `feature/v3.17.0-legacy-discovery-ingestion`
> **Version**: v3.17.2 (Phase 17 IN PROGRESS — PR #48 open)
> **Status**: 990 tests pass. Phases 14–16, 18 merged. Phase 17 PR open.

---

## Current State

### What's Working
- **990 tests pass** (100% pass rate)
- **Server**: FastAPI on `http://localhost:42069`, auth token via `MUNINN_AUTH_TOKEN`
- **MCP**: Registered as "muninn" in Claude Code user config with auth token baked in
- **Claude Desktop**: Already correctly registered as "muninn"
- **Phase 14 (v3.11.0)**: Project-scoped memory — **MERGED**
- **Phase 15 (v3.12.0)**: Operational hardening — **MERGED**
- **Phase 16 (v3.13.0)**: SOTA+ signed verdict v1 — **MERGED**
- **Phase 17a (v3.14.0)**: Synthetic benchmark suite + parser security sandbox — **MERGED**
- **Phase 18 (v3.15.0)**: CI benchmark workflow + token rotation utility — **MERGED** (PRs #46, #47)
- **Phase 17b (v3.17.x)**: Legacy discovery, dashboard overhaul, Scout — **PR #48 open**

### Server Quick Start

```bash
# Start server (Windows — run from repo root)
$env:MUNINN_AUTH_TOKEN = (Get-Content .muninn_token)
python server.py

# Or from bash
MUNINN_AUTH_TOKEN=$(cat .muninn_token) python server.py
```

**Auth token** is stored in `.muninn_token` (gitignored).
```
Token: ij0w9VmdPH5dxnE7vG-lZCXPPWhX9uU7HpBJODg0zoQ
```

### Data Directory
```
C:\Users\user\AppData\Local\AntigravityLabs\muninn\
├── metadata.db          # SQLite — 74 memories
├── qdrant_v8/           # Vector store — 73 vectors
└── kuzu_v12/            # Graph store — activated in Phase 15
```

---

## Phase 17b (v3.17.x) Summary — 2026-02-20

### v3.17.0 — Legacy Discovery Auto-Scan + Selective Ingestion UI
**Files**: `muninn/ingestion/discovery.py`, `dashboard.html`, `muninn/core/memory.py`, tests
- Added providers: `aider_chat`, `continue_dev`, `zed_ai` to `_provider_specs()`
- Updated `gemini_cli` patterns to include `~/.gemini/tmp/<hash>/chats/` paths
- Added `.zed` extension to `SUPPORTED_EXTENSIONS` (text parser)
- Added `_require_discovery_pipeline()` to bypass `multi_source_ingestion` flag for scans
- Fixed critical bug: legacy discovery now works in all deployments (no feature flag required)
- UI: 8-column legacy import table with provider/parser/confidence/size/modified/path columns
- Security: XSS prevention — `textContent` / DOM text nodes replacing `innerHTML` for server data

### v3.17.1 — Dashboard Overhaul + Retrieval/Consolidation Optimization
**Files**: `dashboard.html`, `muninn/store/graph_store.py`, `muninn/consolidation/daemon.py`,
          `muninn/retrieval/hybrid.py`, `muninn/retrieval/weight_adapter.py`,
          `muninn/retrieval/temporal_parser.py`, `docs/plans/BRAINSTORM_UI_UX.md`
- **Dashboard**: Complete 'Slate & Sky' redesign; sidebar nav (Overview/Ingestion/Legacy/System);
  responsive layout; magic search bar; stats grid; graph canvas
- **P0 Performance**: `graph_store.get_memory_node_degrees_batch()` — single batch Kuzu query
  for all N records, eliminating N individual degree lookups (was N+1 queries in `_phase_decay`)
- **P1 Performance**: `hybrid.py` — batch ColBERT token retrieval via `MatchAny` Qdrant scroll
  instead of per-document fetches; entity-grounded graph search optimization
- **Correctness**: `weight_adapter.py` — 'chain' + 'goal' added to `DEFAULT_WEIGHTS`
  (ensures all 6 RRF signals participate in entropy-based adaptation)
- **Windows compat**: All Unicode `→` arrows replaced with ASCII `->` to prevent console crashes
- `temporal_parser.py`: `TimeRange.__str__` now uses `->` separator

### v3.17.2 — Muninn Scout Agentic Discovery + hunt_memory MCP Tool
**Files**: `muninn/retrieval/scout.py` (new), `muninn/core/memory.py`, `muninn/mcp/definitions.py`,
          `muninn/mcp/handlers.py`, `muninn/retrieval/hybrid.py`, `muninn/store/vector_store.py`,
          `server.py`, `dashboard.html`
- **`MuninnScout`**: New agentic multi-hop retrieval engine
  - `hunt(query, depth=2)`: Initial hybrid search → entity expansion via graph → chain expansion
    (PRECEDES/CAUSES edges) → aggregate & rerank
  - Fallback: global scope search when project search returns nothing
  - Configurable depth: 0=no expansion, 1=single hop, 2=full multi-hop
- **`hunt_memory` MCP tool**: POST `/search/hunt`, depth + limit + namespaces params
- **`MuninnMemory.hunt()`**: Wrapper with OTel instrumentation, serializes `SearchResult` → dict
- **`HuntMemoryRequest`** Pydantic model: `query`, `user_id`, `agent_id`, `limit=10`, `depth=2`,
  `namespaces`
- **`VectorStore.search()`**: `memory_ids` filter key → `MatchAny` Qdrant condition
- **`HybridRetriever`**: `memory_ids` parameter plumbed through all 4 signals
  (`_vector_search`, `_graph_search`, `_bm25_search`, `_temporal_search`)
- **Test fixes (v3.17.2)**:
  - `test_temporal_range_str`: updated assertion from `→` to `->` (ASCII migration)
  - `test_colbert_logic_fix`: mock point `.payload` now set to dict so `.get("memory_id")` resolves

### v3.17.x Test Suite — `tests/test_v3_17_0_legacy_scout.py` (51 tests, 9 classes)
| Class | Count | Coverage |
|---|---|---|
| `TestMuninnScout` | 7 | hunt() multi-hop, entity expansion, chain follow, depth=0, fallback |
| `TestVectorStoreMemoryIdsFilter` | 5 | MatchAny filter construction, key, values, non-list filter |
| `TestGraphStoreBatchCentrality` | 8 | empty, zero-default, normalization, cap=1.0, range, absent ID, exception, batch=1 query |
| `TestDaemonPhaseDecayBatch` | 4 | batch call verified, no N+1, empty records, update call |
| `TestHuntMemoryMCPTool` | 8 | READ_ONLY_TOOLS, TOOLS_SCHEMAS, schema props, POST endpoint, payload, defaults |
| `TestHuntMemoryServerModel` | 4 | import, required field, defaults, custom values |
| `TestMuninnMemoryHunt` | 5 | method exists, raises when not init, dict serialization, required keys, depth/limit passthrough |
| `TestHybridRetrieverMemoryIds` | 6 | signatures, bm25 filter, graph filter via target_set |
| `TestVersionBump317` | 2 | >= 3.17.0, pyproject match |

---

## Phase 18 (v3.15.0) Summary — 2026-02-19

### Changes Delivered
- **GitHub Actions CI**: `.github/workflows/benchmark.yml` — dry-run on PRs, 15-min timeout
- **Token rotation**: `python -m muninn.cli rotate-token` — 32-byte urlsafe, writes `.muninn_token`,
  auto-patches Claude Desktop / Cursor configs
- **39 tests** in `tests/test_v3_15_0_ci_token_rotation.py`

---

## Architecture Notes

### Auth Flow
```
Claude Code → spawns mcp_wrapper.py (with MUNINN_AUTH_TOKEN from MCP env config)
           → mcp_wrapper.py uses token in Authorization: Bearer header
           → server.py validates token from MUNINN_AUTH_TOKEN env var
```

### MCP Server Registration (User-Scope)
```json
{
  "muninn": {
    "type": "stdio",
    "command": "C:/Users/user/miniconda3/python.exe",
    "args": ["C:/Users/user/muninn_mcp/mcp_wrapper.py"],
    "env": {
      "MUNINN_AUTH_TOKEN": "ij0w9VmdPH5dxnE7vG-lZCXPPWhX9uU7HpBJODg0zoQ"
    }
  }
}
```

### Scout Architecture
```
MuninnMemory.hunt()
  └─ MuninnScout.hunt(query, depth=2)
       ├── 1. HybridRetriever.search() — broad initial results
       ├── 2. (fallback) global scope search if empty
       ├── 3. Graph expansion: find_related_memories(entity_names)
       ├── 4. Chain expansion: find_chain_related_memories(seed_ids) [if memory_chains flag]
       └── 5. Final re-rank: search(filters={"memory_ids": all_ids})
```

### hunt_memory MCP Tool
```
hunt_memory → _do_hunt_memory() → POST /search/hunt → MuninnMemory.hunt()
```

---

## Test Suite
```bash
# Run full suite
pytest tests/ -q

# Run Phase 17b tests only
pytest tests/test_v3_17_0_legacy_scout.py -v

# Run legacy discovery tests
pytest tests/test_ingestion_discovery.py -v
```

**Expected**: 990 pass, 2 skipped, 0 fail

---

## Open Items / Next Steps

### Phase 17 Completion (PR #48)
- [ ] **Merge PR #48** into main after review

### Phase 19 Candidates
- [ ] **Live benchmark run + signed verdict artifact**: Run `eval/run_benchmark.py --production`
  against live server; commit signed verdict to `eval/reports/`
- [ ] **Public LongMemEval JSONL**: Obtain `longmemeval_oracle.jsonl` for real nDCG@10 baseline
- [ ] **StructMemEval in sota-verdict**: Wire StructMemEval adapter into verdict signing
- [ ] **Scout accuracy evaluation**: Measure `hunt()` recall vs plain `search()` on benchmark data
- [ ] **Scout LLM synthesis**: Use Claude API to synthesize discovery path explanation
  (currently aggregation only; LLM would generate "I found X because Y linked to Z")
- [ ] **Dashboard live search**: Wire magic search bar to `hunt_memory` endpoint
- [ ] **Background legacy scan scheduling**: Periodic auto-scan for new legacy AI data

### Known Remaining Gaps
- **SOTA+ production-run evidence**: Signed verdict against live server not yet committed
- **Scout LLM synthesis**: Step 5 is a no-op — logs elapsed time, no LLM-generated narrative
- **HANDOFF.md phase numbering**: "Phase 17a" = v3.14.x; "Phase 17b" = v3.17.x (internal git log)
  to distinguish the benchmark suite phase from the legacy discovery phase

---

## Validation History

- **Phase 17b**: **990 tests passed (100%), 0 failed** — legacy discovery (aider/continue/zed),
  Muninn Scout + hunt_memory, dashboard overhaul, batch centrality (N+1 fix), ColBERT batch
  optimization, MatchAny filter, 51 new tests, 2 test bug-fixes. v3.17.2. 2026-02-20.
- **Phase 18**: **890 tests passed (100%), 0 failed** — CI benchmark workflow, token rotation CLI,
  MCP config patcher. 39 new tests. v3.15.0. 2026-02-19.
- **Phase 17a**: **851 tests passed (100%), 0 failed** — synthetic benchmark datasets, automated
  benchmark runner, parser security sandbox. 63 new tests. Merged. v3.14.0. 2026-02-19.
- **Phase 16**: **788 tests passed (100%), 0 failed** — SOTA+ signed verdict v1, HMAC-SHA256
  provenance, LongMemEval hard gate, StructMemEval adapter. 61 new tests. v3.13.0. 2026-02-19.
- **Phase 15**: **727 tests passed (100%), 0 failed** — auth propagation fix, graph chains smoke,
  OTel GenAI hardening, LongMemEval adapter. Merged. v3.12.0. 2026-02-19.
- **Phase 14**: **694 tests passed (100%), 0 failed** — project-scoped memory strict isolation.
  43 scope tests. Merged. v3.11.0. 2026-02-19.
