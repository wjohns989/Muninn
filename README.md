<img src="assets/muninn_banner.jpeg" alt="Muninn — Persistent Memory MCP" width="100%"/>

# Muninn

> *"Muninn flies each day over the world to bring Odin knowledge of what happens."*
> — Prose Edda

**Local-first persistent memory infrastructure for coding agents and MCP-compatible tools.**

Muninn provides deterministic, explainable memory retrieval with robust transport behavior and production-grade operational controls. Designed for long-running development workflows where continuity, auditability, and measurable quality matter — across sessions, across assistants, and across projects.

---

## 🚦 Status

**Current Version:** v3.11.0 (Phase 14 Complete — Phase 15 In Progress)
**Stability:** Production Beta
**Test Suite:** 694 passing, 0 failing

### What's New in v3.11.0

- **Project-Scoped Memory** (`scope="project"` | `scope="global"`): Explicit scope field on every memory prevents cross-project leakage in multi-repo agent environments. Project-specific instructions stay isolated; global rules are always visible.
- **Strict Isolation Mode** (`MUNINN_PROJECT_SCOPE_STRICT=1`): Disables fallback search entirely — zero cross-project memory can ever surface.
- **`set_project_instruction` MCP tool**: Convenience tool for tagging project-scoped rules and coding conventions.
- **Schema migration hardening**: Idempotent `scope` column migration — safe upgrade from pre-v3.11.0 databases.
- **MCP server naming**: Registered as `muninn` (tools: `mcp__muninn__*`); `serverInfo.version` now dynamically reflects the installed version.

### Previous Milestones

| Version | Phase | Key Feature |
|---------|-------|-------------|
| v3.10.0 | 13 | Native ColBERT multi-vector MaxSim + NL temporal query expansion |
| v3.9.0 | 12 | Distributed entity scoping with composite IDs |
| v3.8.0 | 11 | Multi-namespace integrity + UI dashboard |
| v3.7.0 | 10 | Unified security architecture |

---

## 🚀 Features

### Core Memory Engine
- **Local-First**: Zero cloud dependency — all data stays on your machine
- **5-Signal Hybrid Retrieval**: Dense vector · BM25 lexical · Graph traversal · Temporal relevance · Goal relevance
- **Explainable Recall Traces**: Per-signal score attribution on every search result
- **Project Isolation**: `scope="project"` memories never cross repo boundaries; `scope="global"` memories are always available
- **Cross-Session Continuity**: Memories survive session ends, assistant switches, and tool restarts
- **Bi-Temporal Records**: `created_at` (real-world event time) vs `ingested_at` (system intake time)

### Memory Lifecycle
- **Consolidation Daemon**: Background process for decay, deduplication, promotion, and replay — inspired by sleep consolidation
- **ColBERT Multi-Vector**: Native Qdrant multi-vector storage for MaxSim scoring
- **NL Temporal Query Expansion**: Natural-language time phrases ("last week", "before the refactor") parsed into structured time ranges
- **Goal Compass**: Retrieval signal for project objectives and constraint drift

### Operational Controls
- **MCP Transport Hardening**: Framed + line JSON-RPC, timeout-window guardrails, protocol negotiation
- **Runtime Profile Control**: `get_model_profiles` / `set_model_profiles` for dynamic model routing
- **Profile Audit Log**: Immutable event ledger for profile policy mutations
- **Browser Control Center**: Web UI for search, ingestion, consolidation, and admin at `http://localhost:42069`
- **OpenTelemetry**: GenAI semantic convention tracing (feature-gated via `MUNINN_OTEL_ENABLED`)

### Multi-Assistant Interop
- **Handoff Bundles**: Export/import memory checkpoints with checksum verification and idempotent replay
- **Legacy Migration**: Discover and import memories from prior assistant sessions (JSONL chat history, SQLite state)
- **Federation**: Multi-instance memory synchronization with delta bundles
- **MCP 2025-11 Compliant**: Full protocol negotiation, lifecycle gating, schema annotations

---

## Quick Start

```bash
git clone https://github.com/wjohns989/Muninn.git
cd Muninn
pip install -e .
```

Set the auth token (shared between server and MCP wrapper):

```bash
# Windows (persists across sessions)
setx MUNINN_AUTH_TOKEN "your-token-here"

# Linux/macOS
export MUNINN_AUTH_TOKEN="your-token-here"
```

Start the backend:

```bash
python server.py
```

Verify it's running:

```bash
curl http://localhost:42069/health
# {"status":"ok","memory_count":0,...,"backend":"muninn-native"}
```

---

## Runtime Modes

| Mode | Command | Description |
|------|---------|-------------|
| **Muninn MCP** | `python mcp_wrapper.py` | stdio MCP server for active assistant/IDE sessions |
| **Huginn Standalone** | `python muninn_standalone.py` | Browser-first UX for direct ingestion/search/admin |
| **REST API** | `python server.py` | FastAPI backend at `http://localhost:42069` |
| **Packaged App** | `python scripts/build_standalone.py` | PyInstaller executable (Huginn Control Center) |

All modes use the same memory engine and data directory.

---

## MCP Client Configuration

Claude Code (recommended — bakes auth token into registration):

```bash
claude mcp add -s user muninn \
  -e MUNINN_AUTH_TOKEN="your-token-here" \
  -- python /absolute/path/to/mcp_wrapper.py
```

Generic MCP client (`claude_desktop_config.json` or equivalent):

```json
{
  "mcpServers": {
    "muninn": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_wrapper.py"],
      "env": {
        "MUNINN_AUTH_TOKEN": "your-token-here"
      }
    }
  }
}
```

> **Important**: Both `server.py` and `mcp_wrapper.py` must share the same `MUNINN_AUTH_TOKEN`. If either process generates a random token (when the env var is unset), all MCP tool calls fail with 401.

---

## MCP Tools

| Tool | Description |
|------|-------------|
| `add_memory` | Store a memory with optional `scope`, `project`, `namespace`, `importance` |
| `search_memory` | Hybrid 5-signal search with explainable recall traces |
| `get_all_memories` | Paginated memory listing with filters |
| `update_memory` | Update content or metadata of an existing memory |
| `delete_memory` | Remove a memory by ID |
| `set_project_goal` | Set the current project's objective and constraints |
| `get_project_goal` | Retrieve the active project goal |
| `set_project_instruction` | Store a project-scoped rule (`scope="project"` by default) |
| `get_model_profiles` | Get active model routing profiles |
| `set_model_profiles` | Update model routing profiles |
| `get_model_profile_events` | Audit log for profile policy changes |
| `export_handoff` | Export a memory handoff bundle |
| `import_handoff` | Import a handoff bundle (idempotent) |
| `ingest_sources` | Ingest files/folders into memory |
| `discover_legacy_sources` | Find prior assistant session files for migration |
| `ingest_legacy_sources` | Import discovered legacy memories |
| `record_retrieval_feedback` | Submit outcome signal for adaptive calibration |

---

## Python SDK

```python
from muninn import Memory

# Sync client
client = Memory(base_url="http://127.0.0.1:42069", auth_token="your-token-here")
client.add(
    content="Always use typed Pydantic models for API payloads",
    metadata={"project": "muninn", "scope": "project"}
)

results = client.search("API payload patterns", limit=5)
for r in results:
    print(r.content, r.recall_trace)
```

Async client:

```python
from muninn import AsyncMemory

async def main():
    async with AsyncMemory(base_url="http://127.0.0.1:42069") as client:
        await client.add(content="...", metadata={})
        results = await client.search("...", limit=5)
```

---

## REST API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server health + memory/vector/graph counts |
| `POST` | `/add` | Add a memory |
| `POST` | `/search` | Hybrid search |
| `GET` | `/get_all` | Paginated memory listing |
| `PUT` | `/update` | Update a memory |
| `DELETE` | `/delete/{memory_id}` | Delete a memory |
| `POST` | `/ingest` | Ingest files/folders |
| `POST` | `/ingest/legacy/discover` | Discover legacy session files |
| `POST` | `/ingest/legacy/import` | Import legacy memories |
| `GET` | `/profiles/model` | Get model routing profiles |
| `POST` | `/profiles/model` | Set model routing profiles |
| `GET` | `/profiles/model/events` | Profile audit log |
| `GET` | `/profile/user/get` | Get user profile |
| `POST` | `/profile/user/set` | Update user profile |
| `POST` | `/handoff/export` | Export handoff bundle |
| `POST` | `/handoff/import` | Import handoff bundle |
| `POST` | `/feedback/retrieval` | Submit retrieval feedback |
| `GET` | `/goal/get` | Get project goal |
| `POST` | `/goal/set` | Set project goal |

Auth: `Authorization: Bearer <MUNINN_AUTH_TOKEN>` required on all non-health endpoints.

---

## Configuration

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MUNINN_AUTH_TOKEN` | random | Shared secret between server and MCP wrapper |
| `MUNINN_SERVER_URL` | `http://localhost:42069` | Backend URL for MCP wrapper |
| `MUNINN_PROJECT_SCOPE_STRICT` | off | `=1` disables cross-project fallback entirely |
| `MUNINN_MCP_SEARCH_PROJECT_FALLBACK` | off | `=1` enables global-scope fallback on empty results |
| `MUNINN_OPERATOR_MODEL_PROFILE` | `balanced` | Default model routing profile |
| `MUNINN_OTEL_ENABLED` | off | `=1` enables OpenTelemetry tracing |
| `MUNINN_COLBERT_MULTIVEC` | off | `=1` enables native ColBERT multi-vector storage |
| `MUNINN_TEMPORAL_QUERY_EXPANSION` | off | `=1` enables NL time-phrase parsing in search |

---

## Evaluation & Quality Gates

Muninn includes an evaluation toolchain for measurable quality enforcement:

```bash
# Run full benchmark dev-cycle
python -m eval.ollama_local_benchmark dev-cycle

# Check phase hygiene gates
python -m eval.phase_hygiene

# Emit SOTA+ verdict artifact
python -m eval.ollama_local_benchmark sota-verdict
```

Metrics tracked: `nDCG@k`, `Recall@k`, `MRR`, p50/p95 latency, significance testing (Bonferroni/BH correction), effect-size analysis.

---

## Data & Security

- **Default data dir**: `~/.local/share/AntigravityLabs/muninn/` (Linux/macOS) · `%LOCALAPPDATA%\AntigravityLabs\muninn\` (Windows)
- **Storage**: SQLite (metadata) + Qdrant (vectors) + KuzuDB (memory chains graph)
- **No cloud dependency**: All data local by default
- **Auth**: Bearer token required on all API calls; token shared via env var
- **Namespace isolation**: `user_id` + `namespace` + `project` boundaries enforced at every retrieval layer

---

## Documentation Index

| Document | Description |
|----------|-------------|
| `SOTA_PLUS_PLAN.md` | Active development phases and roadmap |
| `HANDOFF.md` | Operational setup, auth flow, known issues |
| `docs/ARCHITECTURE.md` | System architecture deep-dive |
| `docs/MUNINN_COMPREHENSIVE_ROADMAP.md` | Full feature roadmap (v3.1→v3.3+) |
| `docs/AGENT_CONTINUATION_RUNBOOK.md` | How to resume development across sessions |
| `docs/PYTHON_SDK.md` | Python SDK reference |
| `docs/INGESTION_PIPELINE.md` | Ingestion pipeline internals |
| `docs/OTEL_GENAI_OBSERVABILITY.md` | OpenTelemetry integration guide |
| `docs/PLAN_GAP_EVALUATION.md` | Gap analysis against SOTA memory systems |

---

## Licensing

- Code: Apache License 2.0 (`LICENSE`)
- Third-party dependency licenses remain with their respective owners
- Attribution: See `NOTICE`
