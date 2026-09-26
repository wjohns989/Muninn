<img src="assets/muninn_banner.jpeg" alt="Muninn — Persistent Memory MCP" width="100%"/>

# Muninn

> *"Muninn flies each day over the world to bring Odin knowledge of what happens."*
> — Prose Edda

**Local-first persistent memory infrastructure for coding agents and MCP-compatible tools.**

Muninn provides deterministic, explainable memory retrieval with robust transport behavior and production-grade operational controls. Designed for long-running development workflows where continuity, auditability, and measurable quality matter — across sessions, across assistants, and across projects.

---

## 🚩 Status

**Current Version:** v3.24.0 (Phase 26 COMPLETE)
**Stability:** Production Beta
**Test Suite:** 1422+ passing, 0 failing

### What's New in v3.24.0

- **Cognitive Architecture (CoALA)**: Integration of a proactive reasoning loop bridging memory with active decision-making.
- **Knowledge Distillation**: Background synthesis of episodic memories into structured semantic manuals for long-term wisdom.
- **Epistemic Foraging**: Active inference-driven search to resolve ambiguities and fill information gaps autonomously.
- **Omission Filtering**: Automated detection of missing context required for successful task execution.
- **Elo-Rated SNIPS Governance**: Dynamic memory retention system mapping retrieval success to Elo ratings for usage-driven decay.

### Previous Milestones

| Version | Phase | Key Feature |
|---------|-------|-------------|
| v3.24.0 | 26 | Cognitive Architecture Complete |
| v3.23.0 | 23 | Elo-Rated SNIPS Governance |
| v3.22.0 | 22 | Temporal Knowledge Graph |
| v3.19.0 | 20 | Multimodal Hive Mind Operations |
| v3.18.3 | 19 | Bulk legacy import, NLI conflict detection, uncapped discovery |
| v3.18.1 | 19 | Scout synthesis, hunt mode |

---

## 🚀 Features

### Core Memory Engine

- **Local-First**: Zero cloud dependency — all data stays on your machine
- **Multimodal**: Native support for Text, Image, Audio, Video, and Sensor data
- **5-Signal Hybrid Retrieval**: Dense vector · BM25 lexical · Graph traversal · Temporal relevance · Goal relevance
- **Explainable Recall Traces**: Per-signal score attribution on every search result
- **Bi-Temporal Reasoning**: Support for "Valid Time" vs "Transaction Time" via Temporal Knowledge Graph
- **Project Isolation**: `scope="project"` memories never cross repo boundaries; `scope="global"` memories are always available
- **Cross-Session Continuity**: Memories survive session ends, assistant switches, and tool restarts
- **Bi-Temporal Records**: `created_at` (real-world event time) vs `ingested_at` (system intake time)

### Memory Lifecycle

- **Elo-Rated Governance**: Dynamic retention driven by retrieval feedback (SNIPS) and usage statistics
- **Consolidation Daemon**: Background process for decay, deduplication, promotion, and shadowing — inspired by sleep consolidation
- **Zero-Trust Ingestion**: Isolated subprocess parsing for PDF/DOCX to neutralize document-based exploits
- **ColBERT Multi-Vector**: Native Qdrant multi-vector storage for MaxSim scoring
- **NL Temporal Query Expansion**: Natural-language time phrases ("last week", "before the refactor") parsed into structured time ranges
- **Goal Compass**: Retrieval signal for project objectives and constraint drift
- **NLI Conflict Detection**: Transformer-based contradiction detection (`cross-encoder/nli-deberta-v3-small`) for memory integrity
- **Bulk Legacy Import**: One-click ingestion of all discovered legacy sources (batched, error-isolated) via dashboard or API

### Operational Controls

- **MCP Transport Hardening**: Framed + line JSON-RPC, timeout-window guardrails, protocol negotiation
- **Runtime Profile Control**: `get_model_profiles` / `set_model_profiles` for dynamic model routing
- **Profile Audit Log**: Immutable event ledger for profile policy mutations
- **Browser Control Center**: Web UI for search, ingestion, consolidation, and admin at `http://localhost:42069`
- **OpenTelemetry**: GenAI semantic convention tracing (feature-gated via `MUNINN_OTEL_ENABLED`)

### Multi-Assistant Interop

- **Handoff Bundles**: Export/import memory checkpoints with checksum verification and idempotent replay
- **Legacy Migration**: Discover and import memories from prior assistant sessions (JSONL chat history, SQLite state) — uncapped provider limits
- **Bulk Import**: `POST /ingest/legacy/import-all` ingests all discovered sources in batches of 50 with per-batch error isolation
- **Hive Mind Federation**: Push-based low-latency memory synchronization across assistant runtimes
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
| **Muninn MCP** | shared `http://127.0.0.1:42069/mcp` | Streamable HTTP MCP on the one machine-wide server |
| **Huginn Standalone** | `python muninn_standalone.py` | Browser-first UX for direct ingestion/search/admin |
| **REST API** | `python server.py` | FastAPI backend at `http://localhost:42069` |
| **Packaged App** | `python scripts/build_standalone.py` | PyInstaller executable (Huginn Control Center) |

All modes use the same memory engine and data directory.

---

## MCP Client Configuration

**Per-client setup** (ChatGPT desktop app with Codex, Claude Desktop and Claude
Code, Gemini CLI, Cursor, VS Code, LM Studio, Open WebUI, web ChatGPT): see
[`docs/CLIENTS.md`](docs/CLIENTS.md). All clients share one store; the server
instructions teach every agent to load `get_project_context` at the start and to
pass work on with `create_handoff` / `resume_handoff`.
Clients with tool limits or small models can load a smaller profile with
`?toolset=core` (or `readonly`, `chatgpt`) on the URL.

The preferred machine-wide topology is one verified `server.py` process and
HTTP clients connected to `http://127.0.0.1:42069/mcp`. Clients must not
auto-start private stdio copies when the shared endpoint is configured.

Generic Streamable HTTP client configuration:

```json
{
  "mcpServers": {
    "muninn": {
      "type": "http",
      "url": "http://127.0.0.1:42069/mcp",
      "headers": {
        "Authorization": "Bearer ${MUNINN_AUTH_TOKEN}"
      }
    }
  }
}
```

The legacy stdio wrapper remains available for clients without Streamable HTTP
support, but it connects to the existing backend and is not a second store owner.

The endpoint is dual-era: clients on MCP 2026-07-28 send stateless requests
(protocol version, client info and capabilities in `params._meta`, mirrored in the
`MCP-Protocol-Version`, `Mcp-Method` and `Mcp-Name` headers) and can call
`server/discover`; clients on 2025-11-25 and earlier keep using `initialize` and
`Mcp-Session-Id`. Stateless clients that want session inhibition pass a
`session_id` argument to `search_memory`.

Legacy wrapper registration:

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
| `get_project_context` | Session-start briefing: goal, open handoffs, project rules, recent memories by agent, global preferences |
| `create_handoff` | Leave work for another agent: summary, next steps, decisions, open questions, files, branch |
| `resume_handoff` | Claim the newest open handoff for a project (or a given id) |
| `complete_handoff` | Mark a resumed handoff done or cancelled, or release it |
| `add_memory` | Store a memory with optional `scope`, `project`, `namespace`, `media_type` |
| `add_image_memory` | Store a local image plus a searchable description and optional memory links |
| `search_memory` | Hybrid 5-signal search with `media_type` filtering and recall traces |
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
| `hunt_memory` | Agentic multi-hop search with a synthesized summary |
| `correct_fact` | Rewrite a wrong memory from a user correction |
| `detect_information_gaps` | List missing details (paths, credentials, hosts) a task needs |
| `forage_knowledge` | Follow graph links when a search is ambiguous |
| `trigger_distillation` | Condense clusters of episodic memories into semantic notes |
| `search`, `fetch` | ChatGPT connector tools (`chatgpt` profile only) |

The full list (41 tools) is returned by `tools/list`; federation, periodic
ingestion, model-profile and `mimir_relay` tools are omitted above.

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
    async with AsyncMemory(base_url="http://127.0.0.1:42069", auth_token="your-token-here") as client:
        await client.add(content="...", metadata={})
        results = await client.search("...", limit=5)
```

---

## REST API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Counts plus content-free resource, cache, and MCP transport utilization |
| `POST` | `/add` | Add a memory (supports `media_type`) |
| `POST` | `/add-image` | Copy an image into managed local storage and add its description |
| `GET` | `/images/{stored_name}` | Authenticated access to a managed image |
| `POST` | `/search` | Hybrid search (supports `media_type` filtering) |
| `POST` | `/mcp` | MCP Streamable HTTP transport |
| `GET` | `/get_all` | Paginated memory listing |
| `PUT` | `/update` | Update a memory |
| `DELETE` | `/delete/{memory_id}` | Delete a memory |
| `POST` | `/restore/{memory_id}` | Restore a memory that consolidation archived (merge, decay, temporal shadow) |
| `POST` | `/admin/reindex` | Rebuild vectors/BM25 from metadata (dry run by default) |
| `POST` | `/admin/import` | Import exported memories, keeping original timestamps (dry run by default) |
| `POST` | `/ingest` | Ingest files/folders |
| `POST` | `/ingest/legacy/discover` | Discover legacy session files |
| `POST` | `/ingest/legacy/import` | Import selected legacy memories |
| `POST` | `/ingest/legacy/import-all` | Discover and import ALL legacy sources (batched) |
| `GET` | `/ingest/legacy/status` | Legacy discovery scheduler status |
| `GET` | `/ingest/legacy/catalog` | Paginated cached catalog of discovered sources |
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
| `MUNINN_OTEL_ENDPOINT` | `http://localhost:4318` | OTLP HTTP endpoint for trace export |
| `MUNINN_CHAINS_ENABLED` | off | `=1` enables graph memory chain detection (PRECEDES/CAUSES edges) |
| `MUNINN_COLBERT_MULTIVEC` | off | `=1` enables native ColBERT multi-vector storage |
| `MUNINN_FEDERATION_ENABLED` | off | `=1` enables P2P memory synchronization |
| `MUNINN_FEDERATION_PEERS` | - | Comma-separated list of peer base URLs |
| `MUNINN_FEDERATION_SYNC_ON_ADD` | off | `=1` enables real-time push-on-add to peers |
| `MUNINN_TEMPORAL_QUERY_EXPANSION` | off | `=1` enables NL time-phrase parsing in search |
| `MUNINN_CONSOLIDATION_DRY_RUN` | off | `=1` computes consolidation changes and lists them in `/consolidation/status` without writing any store |
| `MUNINN_CONSOLIDATION_BATCH_SIZE` | `500` | Memories visited per phase per cycle; a persisted cursor pages through the whole store |
| `MUNINN_IMPORTANCE_MODEL` | `auto` | Self-supervised importance: `auto` learns continuously and uses the learned model only while it beats the hand-weighted score on its own outcomes; `shadow` learns and reports only; `legacy` disables learning |
| `MUNINN_ADAPTIVE_HORIZON_DAYS` | `7` | Window in which a memory must be re-retrieved by a new session to count as needed |
| `MUNINN_ADAPTIVE_SAMPLES_PER_CYCLE` | `200` | Predictions recorded per consolidation cycle for later self-labelling |
| `MUNINN_ADAPTIVE_MIN_EXAMPLES` | `200` | Resolved outcomes required before the learned model may take over |
| `MUNINN_SESSION_INHIBITION` | on | Demote memories already returned in the same agent session (requires `session_id` on search; MCP sends it) |
| `MUNINN_SESSION_INHIBITION_RANK_PENALTY` | `3` | Positions a repeated memory moves down in the final ranked pool |
| `MUNINN_SESSION_INHIBITION_TTL_SEC` | `1800` | How long a returned memory stays inhibited within a session |
| `MUNINN_RERANKER_ENABLED` | on | `=false` disables cross-encoder reranking |
| `MUNINN_RERANKER_MODEL` | `jinaai/jina-reranker-v1-turbo-en` | FastEmbed cross-encoder model; set the prior `jinaai/jina-reranker-v1-tiny-en` to roll back ranking behavior |
| `MUNINN_CONSOLIDATION_INTEGRITY_RESOURCE_MODE` | `cycle` | Load NLI integrity resources only for a consolidation cycle; `persistent` restores legacy eager lifetime |
| `MUNINN_RETRIEVAL_FEEDBACK_CACHE_MAX_ENTRIES` | `1024` | Hard bound for adaptive retrieval-feedback cache entries |
| `MUNINN_INGESTION_MAX_WORKERS` | `2` | Process-worker bound for multi-source ingestion (`1`-`8`) |
| `MUNINN_IMAGE_MAX_BYTES` | `104857600` | Maximum source image size copied into managed storage |
| `MUNINN_MCP_HTTP_MAX_SESSIONS` | `128` | Streamable HTTP session capacity |
| `MUNINN_MCP_HTTP_SESSION_TTL_SEC` | `3600` | Idle Streamable HTTP session expiry |
| `MUNINN_MCP_HTTP_MAX_BATCH_SIZE` | `100` | JSON-RPC batch bound |
| `MUNINN_MCP_HTTP_MAX_REQUEST_BYTES` | `1048576` | Streamable HTTP request-body bound |
| `MUNINN_MCP_HTTP_MAX_INFLIGHT` | `64` | Process-wide Streamable HTTP dispatch bound |
| `MUNINN_MCP_HTTP_MAX_INFLIGHT_PER_SESSION` | `8` | Per-session Streamable HTTP dispatch bound |
| `MUNINN_MCP_SSE_MAX_SESSIONS` | `128` | Legacy SSE session capacity |
| `MUNINN_MCP_SSE_QUEUE_SIZE` | `256` | Per-session legacy SSE response queue bound |
| `MUNINN_MCP_SSE_MAX_INFLIGHT` | `8` | Per-session legacy SSE dispatch-task bound |
| `MUNINN_AGENT_NAME` | client name | Agent label recorded on memories and handoffs for a stdio client (HTTP clients use `?agent=`) |
| `MUNINN_PROJECT` | git repo | Project for a stdio client started outside a repository (e.g. by Claude Desktop) |
| `MUNINN_MCP_TOOLSET` | `full` | Tool profile for stdio clients: `full`, `core`, `readonly` or `chatgpt` (HTTP clients use `?toolset=`) |
| `MUNINN_ALLOWED_ORIGINS` | - | Extra browser origins allowed besides localhost (comma-separated; `null` allows `file://`, `*` disables the check) |

`config.template.yaml` contains conservative, relative-path defaults. Keep real
tokens and machine-specific data paths in private environment/configuration files.

### Upgrading and migrating memories

`metadata.db` is the source of truth; vectors and the keyword index are derived from it.
Every command below talks to the running server and is a dry run unless `--apply` is given.

```bash
# Rebuild vectors and BM25 from metadata.db (after an embedding-model change add
# --recreate-vectors; also use after restoring metadata.db into a fresh install)
python -m muninn.cli reindex --apply

# Import memories exported from another system, including the pre-3.0 Mem0-based
# Muninn: JSONL, a JSON array, or a Mem0 GET /memories response. Original
# timestamps are kept, exact duplicates skipped, and the original user id is
# stored as metadata.legacy_user_id.
python -m muninn.cli import export.json --source mem0
python -m muninn.cli import export.json --source mem0 --apply
```

`/health` reports `legacy_stores` (booleans only) when an older `~/.muninn/data` or Mem0
store exists on the machine. Back up the data directory before any `--apply`.

### Reproducible memory profiling

The benchmark harness refuses port `42069`, strips credentials, disables
external model calls, and creates every store beneath a temporary directory:

```bash
python -m eval.memory_profile_benchmark \
  --output eval/reports/memory/local-report.json \
  --soak-seconds 1800 \
  --idle-soak-seconds 1800
```

Generated reports are intentionally ignored because they can contain local
temporary paths. Publish only reviewed aggregate measurements.

---

## Evaluation & Quality Gates

Muninn includes an evaluation toolchain for measurable quality enforcement:

```bash
# Run full benchmark dev-cycle
python -m eval.ollama_local_benchmark dev-cycle

# Check phase hygiene gates
python -m eval.phase_hygiene

# Emit SOTA+ signed verdict artifact
python -m eval.ollama_local_benchmark sota-verdict \
  --longmemeval-report path/to/lme_report.json \
  --min-longmemeval-ndcg 0.60 \
  --min-longmemeval-recall 0.65 \
  --signing-key "$SOTA_SIGNING_KEY"

# Run LongMemEval adapter selftest (no server needed)
python eval/longmemeval_adapter.py --selftest

# Run StructMemEval adapter selftest (no server needed)
python eval/structmemeval_adapter.py --selftest

# Run StructMemEval against a live server
python eval/structmemeval_adapter.py \
  --dataset path/to/structmemeval.jsonl \
  --server-url http://localhost:42069 \
  --auth-token "$MUNINN_AUTH_TOKEN"
```

Metrics tracked: `nDCG@k`, `Recall@k`, `MRR@k`, Exact Match, token-F1, p50/p95 latency, significance testing (Bonferroni/BH correction), effect-size analysis.

The `sota-verdict` command emits a signed JSON artifact with `commit_sha`, SHA256 file hashes, and HMAC-SHA256 `promotion_signature` — enabling auditable, commit-pinned SOTA+ evidence.

---

## Data & Security

- **Default data dir**: `~/.local/share/AntigravityLabs/muninn/` (Linux/macOS) · `%LOCALAPPDATA%\AntigravityLabs\muninn\` (Windows)
- **Storage**: SQLite (metadata) + Qdrant (vectors) + KuzuDB (memory chains graph)
- **No cloud dependency**: All data local by default
- **Auth**: when `MUNINN_AUTH_TOKEN` or `MUNINN_API_KEY` is set, every API and MCP call needs it as a Bearer token; without one, only local callers are expected
- **Browser origins**: requests from web pages other than `localhost` are rejected (blocks cross-site access and DNS rebinding); extend with `MUNINN_ALLOWED_ORIGINS`
- **Namespace isolation**: `user_id` + `namespace` + `project` boundaries enforced at every retrieval layer

---

## Documentation Index

| Document | Description |
|----------|-------------|
| `SOTA_PLUS_PLAN.md` | Active development phases and roadmap |
| `docs/plans/2026-09-25-post-codex-hardening-plan.md` | Current plan and status |
| `docs/archive/` | Historical handoffs and remediation reports (including `HANDOFF.md`) |
| `docs/ARCHITECTURE.md` | System architecture deep-dive |
| `docs/MUNINN_COMPREHENSIVE_ROADMAP.md` | Full feature roadmap (v3.1→v3.3+) |
| `docs/AGENT_CONTINUATION_RUNBOOK.md` | How to resume development across sessions |
| `docs/PYTHON_SDK.md` | Python SDK reference |
| `docs/CLIENTS.md` | Connecting Claude, ChatGPT, Codex, Gemini, Cursor, VS Code and local-model clients |
| `docs/INGESTION_PIPELINE.md` | Ingestion pipeline internals |
| `docs/OTEL_GENAI_OBSERVABILITY.md` | OpenTelemetry integration guide |
| `docs/PLAN_GAP_EVALUATION.md` | Gap analysis against SOTA memory systems |

---

## Licensing

- Code: Apache License 2.0 (`LICENSE`)
- Third-party dependency licenses remain with their respective owners
- Attribution: See `NOTICE`
