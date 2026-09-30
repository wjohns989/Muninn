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
- **Bulk Legacy Import (legacy mode only)**: One-click ingestion of discovered legacy sources via dashboard or API; strict vault-first mode does not import raw transcripts into ordinary memory

### Operational Controls

- **MCP Transport Hardening**: Framed + line JSON-RPC, timeout-window guardrails, protocol negotiation
- **Runtime Profile Control**: `get_model_profiles` / `set_model_profiles` for dynamic model routing
- **Profile Audit Log**: Immutable event ledger for profile policy mutations
- **Browser Control Center**: Web UI for search, ingestion, consolidation, and admin at `http://localhost:42069`
- **OpenTelemetry**: GenAI semantic convention tracing (feature-gated via `MUNINN_OTEL_ENABLED`)

### Multi-Assistant Interop

- **Handoff Bundles**: Export/import memory checkpoints with checksum verification and idempotent replay
- **Legacy Migration**: Explicit opt-in for the older plaintext importer; vault-first encrypted capture is the safe default
- **Bulk Import (legacy mode only)**: `POST /ingest/legacy/import-all` ingests discovered sources in batches of 50 with per-batch error isolation when explicitly enabled
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
# Windows PowerShell: prompt without putting the token in shell history
$secret = Read-Host 'Muninn auth token' -AsSecureString
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
try {
    $value = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
    [Environment]::SetEnvironmentVariable('MUNINN_AUTH_TOKEN', $value, 'User')
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
    Remove-Variable value, secret, ptr -ErrorAction SilentlyContinue
}

# Linux/macOS
export MUNINN_AUTH_TOKEN="your-token-here"
```

Start the backend:

```bash
python server.py
```

On Windows, `scripts/start_shared_local.ps1` starts or verifies one authenticated
loopback service. It reads `MUNINN_AUTH_TOKEN` from the Windows user environment
without placing the token in a command argument or client config. It refuses to
report success if anonymous access works or the token fails. Select your own
data directory and Python installation; no `D:` drive or specific model is
required. Initialize the encrypted history archive in that data directory
first (see below); the launcher treats an unready archive as a failed setup:

```powershell
.\scripts\start_shared_local.ps1 -PythonPath 'C:\path\to\python.exe' -DataDir 'C:\path\to\private-muninn-data'
```

For an existing installation, omit `-DataDir` to keep the current
`MUNINN_DATA_DIR` value. If unset, the Windows launcher uses `.muninn_runtime`
under the checkout. Keep that directory private and backed up. The launcher
sets strict history mode, deferred chat LLM use, zero Ollama residency, and
consolidation dry-run; it does not import historical chats or send them to a
remote model. Start the service explicitly before connecting MCP clients.

Verify it's running:

```bash
curl http://localhost:42069/health
# {"status":"ok","memory_count":0,...,"backend":"muninn-native"}
```

To check the authenticated, durable history-search queue against your own
archive without displaying transcript text or credentials, run
`python scripts/smoke_secure_search_live.py --query "a term you expect"`.
The probe reads `MUNINN_AUTH_TOKEN` from its environment (or the Windows user
environment), reports enqueue/search timing and safe match counts, and verifies
one redacted fetch when a match exists. It never prints the query, capability,
or fetched span. Search jobs are available through
`POST /history/secure/search/jobs` and authenticated poll/cancel endpoints.
For more context, `start_secure_history_transcript` queues a CPU-only encrypted
projection of the selected snapshot. Repeat that call while pending (or use
`poll_secure_history_transcript` in the full tool profile),
then pass its signed cursor to `read_secure_history_transcript_page` repeatedly.
Each page has at most 4,000 credential-redacted characters; the cursor is
one-use and expires after 10 minutes of inactivity, but there is no total
transcript-size/page-count limit. A renewed search can reopen a completed
projection after an expiry or service restart. Pages contain supported
user/assistant text only, not tool output, raw JSON, source paths, or credential
values. The ready response includes authenticated counts of included and
omitted records/messages. Unsupported formats fail closed; `fetch_secure_history`
remains the short-span compatibility path. Projection runs on CPU on demand and
does not load an Ollama model or reserve VRAM.
On a running installation, add `--transcript-pages 3` to the live search
probe above to check actual continuation pages without printing their text.
For the browser viewer, run `python scripts/smoke_dashboard_browser.py --transcript-query "a nonsecret term"`; add `--candidate-html` to test the
checked-out page against the real loopback backend before a service reload.
The browser probe reports counts only and never saves a transcript screenshot.
Before a service restart, the checked-out code can be tested directly against
your configured encrypted archive:

```bash
python -m scripts.smoke_secure_projection_archive --query "a nonsecret term" --max-size-kib 10240 --pages 2
```

This uses the local
archive unlock, may create an encrypted derived projection for one real hit,
and prints only provider, size, timing, coverage counts, and page lengths.
Optional `--analyze local` or `--analyze remote` exercises a model route without
printing its analysis. The remote option sends the authenticated raw hit window
to the configured ZDR OpenRouter route only when local consent and budget checks
pass; use it only for material you intend to send to that provider.

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
On Windows, desktop clients with stale process environments can launch
`<installed-python.exe> -E -P -m muninn_mcp_bridge` instead. That bridge reads
the current User `MUNINN_AUTH_TOKEN` at MCP launch, pins the shared loopback
endpoint and never starts a server or Ollama. Keep the bearer out of client
config; see [Windows authenticated stdio](docs/CLIENTS.md#windows-authenticated-stdio-for-codex-claude-code-and-gemini-cli).

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
| `get_thread` | Re-read an imported conversation in order, or list a project's threads |
| `import_agent_history` | Older plaintext import tool, disabled by default in strict history mode |
| `add_memory` | Store a memory with optional `scope`, `project`, `namespace`, `media_type` |
| `add_image_memory` | Store a local image plus a searchable description and optional memory links |
| `search_memory` | Hybrid 5-signal search with `media_type` filtering and recall traces |
| `search_secure_history` | Search an owner-only encrypted lexical index for archived transcript references and short-lived fetch grants |
| `fetch_secure_history` | Retrieve one authenticated, bounded, best-effort redacted transcript span from a search grant |
| `start_secure_history_transcript`, `poll_secure_history_transcript`, `read_secure_history_transcript_page` | Queue a CPU-only encrypted conversational projection and continue through signed, bounded, redacted pages |
| `search_credential_metadata` | Opt-in lookup of vault record existence and project-relative `.env` location; never a secret value |
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

The full list (43 tools) is returned by `tools/list`; federation, periodic
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
| `MUNINN_HISTORY_VAULT` | on | Keep a private copy of Claude Code/Desktop, Codex and Gemini CLI transcripts (the apps delete theirs); see `docs/CLIENTS.md` |
| `MUNINN_HISTORY_SECURITY` | `strict` | Vault-first protection is the default. `legacy` explicitly enables the older plaintext vault and file-ingestion paths; do not use it with secret-bearing material |
| `MUNINN_HISTORY_ARCHIVE_DIR` | `<data_dir>/history_secure_archive` | Owner-only encrypted history archive location; set this to a private directory with enough space, on any drive |
| `MUNINN_HISTORY_INDEX_AUTO` | off for direct server starts; on in Windows shared launcher | Build a resumable CPU-only encrypted transcript index in bounded batches; no model or GPU use |
| `MUNINN_SECURE_AUTO_ANALYSIS` | off for direct server starts; on in Windows shared launcher | After an authenticated search finds a pertinent encrypted snapshot, queue one durable, provisional interpretation job; capture and indexing remain CPU-only. Set `0` in the Windows User environment to disable |
| `MUNINN_HISTORY_SYNC_MINUTES` | `30` | Legacy sync cadence or strict-mode CPU-only discovery cadence for missed/changed chat transcripts; historical exports still need explicit encrypted sync |
| `MUNINN_HISTORY_AUTO_IMPORT` | off in strict mode | Legacy-mode automatic import switch; ignored by strict mode |
| `MUNINN_OPENROUTER_API_KEY` | - | Preferred environment key for optional ZDR OpenRouter use; a Windows user-scoped value is picked up by the running service without putting the key in repo files |
| `MUNINN_INSIGHTS_PROVIDER` | auto | `openrouter` or `ollama` for thread analysis (auto: OpenRouter when a key is set) |
| `MUNINN_INSIGHTS_MODEL` | `openai/gpt-6-luna-pro` | Primary model for thread analysis; falls back to DeepSeek V4 Flash, then Gemini 3.5 Flash-Lite (all zero data retention), including when a model refuses. A `:batch` suffix is dropped. `python -m muninn.cli openrouter set` keeps a prompted key in that process only and saves only nonsecret model settings; use an environment variable for persistent credentials |
| `MUNINN_INSIGHTS_WINDOW_TOKENS` | `200000` | Conversation per analysis call; larger threads are split and merged |
| `MUNINN_INSIGHTS_AUTO` | off | Legacy-mode analysis after automatic import; does not enable strict encrypted-history enrichment |
| `MUNINN_HISTORY_HOMES` | - | Extra home folders to scan for app history (e.g. the Windows home from WSL) |
| `MUNINN_DATA_DIR` | platform data directory | All Muninn stores; choose a private local directory with enough space for your history and reliable advisory file locking, not a repository checkout or network share |
| `MUNINN_PYTHON_PATH` | `python` on `PATH` | Windows shared launcher interpreter; set a user-scoped path to a trusted Python executable with Muninn's dependencies, or pass `-PythonPath` to the launcher. This setting authorizes execution of that file; do not point it at an untrusted checkout or download. No interpreter path is hard-coded in the repo |
| `MUNINN_OLLAMA_URL` | `http://127.0.0.1:11434` | Loopback-only HTTP Ollama endpoint for private analysis and status; no model directory or drive letter is assumed |
| `MUNINN_OLLAMA_MODEL` | `llama3.2:3b` | Model for explicitly requested Ollama analysis |
| `MUNINN_AUTO_LOCAL_MODEL_HINTS` | local measured defaults | Comma-separated installed model tags in preferred order for strict on-demand analysis; unlisted installed chat-capable models remain fallback candidates. Legacy analysis also accepts model-name fragments |
| `MUNINN_OLLAMA_KEEP_ALIVE` | `0` | Release an Ollama model after an analysis request instead of leaving it resident in VRAM |
| `MUNINN_STRICT_REMOTE_ANALYSIS` | off | Legacy standalone setting; the strict server fails remote use closed until consent is explicitly saved in the localhost Encrypted History tab. Explicit `analyze_secure_history` calls also need `allow_remote=true` |
| `MUNINN_OPENROUTER_MAX_DAILY_USD` | `10` | Legacy daily admission threshold until a policy is saved; values above $10 require explicit override |
| `MUNINN_OPENROUTER_MAX_MONTHLY_USD` | `100` | Legacy monthly admission threshold until a policy is saved; values above $100 require explicit override |
| `MUNINN_OPENROUTER_BUDGET_OVERRIDE` | off | Legacy explicit override for app thresholds; the provider key's own finite cap is still mandatory and independent |
| `MUNINN_CREDENTIAL_API_TOKEN` | unset (API disabled) | Dedicated 32+-character bearer token for loopback-only credential metadata search and explicit passphrase reveal; keep it in your local user environment, not a checked-in file |
| `MUNINN_CREDENTIAL_AGENT_SEARCH` | off | Set `1` to let authenticated loopback MCP/API clients search allowlisted vault metadata; secret-value reveal remains unavailable to agents |
| `MUNINN_MCP_TOOLSET` | `full` | Tool profile for stdio clients: `full`, `core`, `readonly` or `chatgpt` (HTTP clients use `?toolset=`) |
| `MUNINN_MCP_AUTO_START` | off | MCP clients only connect to the shared server; they do not launch a detached backend when it is down. Set `1` only if client-managed startup is explicitly desired |
| `MUNINN_ALLOWED_ORIGINS` | - | Extra browser origins allowed besides localhost (comma-separated; `null` allows `file://`, `*` disables the check) |

`config.template.yaml` contains conservative, relative-path defaults. Keep real
tokens and machine-specific data paths in private environment/configuration files.

### Importing your existing AI conversations

Strict history mode is now the default. It blocks the old gzip/plaintext vault,
legacy chat import and analysis, generic project-file ingestion, and automatic
legacy discovery. Ordinary memory features remain available. The encrypted
archive keeps raw snapshots locally, without invoking Ollama or OpenRouter or
writing transcript text to ordinary indexes. A rebuildable, owner-only encrypted
lexical index supports `search_secure_history`; each match includes an opaque
reference, source metadata, and a ten-minute fetch grant. `fetch_secure_history`
authenticates the complete archived snapshot before returning one bounded,
best-effort redacted span. For supported chat transcripts up to 8 MiB, it parses
the authenticated conversation before redaction so JSONL metadata on the same
physical line cannot erase a useful user/assistant message. Larger snapshots
use the bounded streaming fallback. Both operations require the normal
authenticated local service.
The index may initially be incomplete while the CPU-only worker catches up;
search reports coverage. Neither operation reveals the exact raw original.
The archive stores authenticated 1 MiB chunks. For large or dense valid UTF-8
snapshots, the rebuildable search index now writes fixed-size encrypted filters
per chunk instead of rejecting a whole file as "oversized". Builds use bounded
memory and publish a completion marker only after the entire snapshot verifies;
insufficient disk space defers the build for retry. Search still authenticates
candidate snapshots and may take proportionally longer for very large files.
Binary or invalid-UTF-8 formats need a format-specific parser and can remain
unsearchable; `unsearchable` does not by itself mean a file was too large.
After upgrading an old index, `python -m muninn.history.blind_index retry --root
'<your-private-data-dir>\history_secure_archive' --max-snapshots 1` upgrades
one legacy overflow at a time without rewriting archive ciphertext.
Redaction recognizes common credentials and suppresses suspicious lines, but
arbitrary unknown secrets cannot be proven absent from transcript text; treat
agent-visible spans as private data and do not use them for automatic credential
execution. Credential-value reveal remains a separate local-only operation.

Initialize the archive interactively with a recovery passphrase you keep outside
Muninn and its backups. Never pass the passphrase on the command line or in chat:

```powershell
python -m muninn.history.secure_archive init --root '<your-private-data-dir>\history_secure_archive'
python -m muninn.history.secure_archive status --root '<your-private-data-dir>\history_secure_archive'
python -m muninn.history.secure_archive plan --root '<your-private-data-dir>\history_secure_archive' --home '<your-home-dir>'
python -m muninn.history.secure_archive sync --root '<your-private-data-dir>\history_secure_archive' --home '<your-home-dir>'
python -m muninn.history.secure_archive catalog --root '<your-private-data-dir>\history_secure_archive'
python -m muninn.history.secure_archive verify --root '<your-private-data-dir>\history_secure_archive'
python -m muninn.history.secure_archive backup --root '<your-private-data-dir>\history_secure_archive' --backup-root '<new-private-backup-dir>'
```

On non-Windows systems, add `--portable` to `status`, `plan`, `sync`,
`catalog`, or `verify` to unlock with a local passphrase prompt. The prompted
passphrase is passed to the history service in memory, not stored in an
environment variable or config file. Unattended archive reopening and the
`backup` command currently require Windows user protection; a portable backup
can still be restored with its recovery passphrase.

`sync` is a manual, resumable, encrypted copy-only operation. It does not import
memories or analyze chats; live Codex `state_*.sqlite` files are reported as
skipped until an encrypted online-SQLite snapshot is available. Claude Code,
Codex and Gemini CLI hooks commit a private capture-journal row before acknowledging
a transcript event. A CPU-only worker archives it and replays pending work after
restart; a separate bounded scanner checks for missed/new chat transcripts on
the configured cadence. The journal contains encrypted source locators and is
included in authenticated portable archive backups. Queue acknowledgement is
not a claim that the source has already been copied. A missing source is retried
eight times, then remains visible as `unavailable` rather than consuming work
forever; if the file appears later, a hook or scan requeues it. The separate
CPU-only worker indexes encrypted snapshots
when `MUNINN_HISTORY_INDEX_AUTO=1`. Claude Code, Codex and Gemini CLI have
optional local lifecycle hooks installed with `python -m muninn.cli hooks install
--apply`; Claude Desktop's non-Code client uses MCP and scheduled sync instead.
It does not hold an Ollama model in VRAM.
With `MUNINN_SECURE_AUTO_ANALYSIS=1`, a successful authenticated search with a
matching archived transcript queues one durable analysis job for its newest
verified hit. This is search-triggered interpretation, not a historical sweep,
and its model output is provisional—not a verified memory or completion claim.
Capture, encrypted search, transcript retrieval, and ordinary memory services
work without resident models. A single background worker checks current GPU
headroom and installed Ollama chat models when a job is due, releases a local
model with `keep_alive=0`, and defers work when resources or the remote budget
are unavailable. Search replies include an opaque `analysis_job_id`; agents can
poll or cancel that job through authenticated API/MCP tools. Results are sealed
in the local archive journal and expire after one day. No raw transcript span
is returned by normal job polling. An agent can also call
`analyze_secure_history` on a pertinent search hit: Muninn authenticates the
expiring capability, selects a fitting installed Ollama completion model using
live GPU telemetry, and returns bounded, sanitized analysis without persisting
the prompt or answer. When locally enabled, a call with `allow_remote=true` may
use ZDR OpenRouter if no local model fits or if a completed local response
fails schema validation; both cases still require current consent, budget and
ZDR checks. `prefer_remote=true` explicitly selects that route for one hit.
Local transport, timeout, OOM and invalid-input failures never switch remotely.
Without `allow_remote=true`, an invalid local response is reported as deferred.
`MUNINN_INSIGHTS_AUTO` applies only
to legacy history import and must not be treated as enabling strict-mode enrichment.
The main local bearer token is an administrator capability shared by clients
on the same trusted Windows account. Anyone who holds it can inspect or cancel
another client's secure history job; Muninn does not provide per-client tenant
isolation within that token. Do not distribute it to untrusted clients.
To compare installed Ollama models on actual checked-in Muninn code without storing
results, run `python scripts/verify_live_model_routes.py --models <installed-tag>`.
To exercise authenticated search, bounded fetch, and actual analysis through the
running local service without printing transcript or model text, run
`python -m scripts.smoke_secure_history_routes --query <nonsecret-term>
--provider ollama` (or `--provider openrouter` only after configuring its
ZDR key, consent, and finite provider-enforced cap).
At `http://127.0.0.1:42069`, the authenticated Home view shows live service
health, generation-bound encrypted archive/index coverage, capture-intent states,
and each client's latest accepted hook; unknown or stale coverage is labeled as
such. It does not sample GPU or call a remote provider in the background. The
dashboard also has a separate **Encrypted History** tab that queues a durable search,
reports archive/index generations, indexed/total coverage, capture-intent/source-scan
status, accepted hook invocation counts, and partial results, and loads a bounded excerpt only when
selected. The UI labels current coverage unknown unless the index report is
bound to the ready archive's generation; hook counts alone do not prove a
host-origin event or completed archival. Excerpts receive best-effort redaction,
not a guarantee that all sensitive text is gone; the dashboard has no raw
transcript or credential reveal control. An on-demand local resource check uses
the authenticated `GET /history/secure/resources` endpoint to report GPU telemetry,
installed Ollama models, and Ollama residency without loading a model; an
unavailable probe is shown as unknown, not idle. **Ordinary Search** remains a different
corpus. This is a narrow operator slice, not the full control-center overhaul described in
`docs/plans/2026-09-28-local-control-center-overhaul.md`.
The separate **Credential Metadata** tab searches only allowlisted metadata from
already scanned or recorded vault sources after explicit local authentication and
opt-in. It does not return values, prove a missing credential does not exist, or
offer a reveal/use action; those retain their separate local authorization flow.
With optional Python Playwright and Microsoft Edge installed, `python -m
scripts.smoke_dashboard_browser --width 390` checks the authenticated History
view in an isolated, loopback-only browser context without searching transcripts
or printing the token. Add `--expect-resources-ready` after deploying the new
resource endpoint to verify its live UI response.
For authenticated CLI commands, token precedence is explicit `--token-file`,
`MUNINN_TOKEN_FILE`, process `MUNINN_AUTH_TOKEN`, Windows user environment,
then the default `.muninn_token`. An explicit missing/empty token file does not
fall back. `python -m muninn.cli doctor` verifies `/auth/check` with positive
and negative controls; `doctor --repair` never writes host configs unless that
check passes. HTTP MCP profiles with dynamic bearer references are reported as
unverified, not token drift, and generic repair/rotation does not inject stdio
environment fields into HTTP profiles.
Add `--archive-query <search-term>` to test a bounded, redacted excerpt of your
own encrypted history locally. OpenRouter validation requires an environment key,
provider-enforced daily cap, and ZDR route; `--openrouter` uses checked-in source
by default. Testing a private excerpt remotely additionally requires
`--allow-private-openrouter`, and remains best-effort credential-scrubbed rather
than proof that arbitrary secrets are absent. All Ollama requests use
`keep_alive=0`; the script reports post-request model residency. Model tags and
data directories are discovered or provided by the operator, not tied to a
particular drive or download directory.
OpenRouter API keys have one provider-enforced reset period (daily or monthly).
Muninn checks that finite key limit against the matching local admission
threshold and checks provider-reported usage for the other period before each
request. Application thresholds are **not hard spending caps**: an individual
request or overlapping requests can cross one. Use OpenRouter key and account
guardrails for hard limits. Changing Muninn's local threshold does not raise the
API key's own limit.
The dashboard shows the provider-enforced key limit, reset period, remaining
amount, and current route eligibility when remote use is enabled. This status
requires the main local token and does not expose the API key or key label.

Open the localhost dashboard's Encrypted History tab with the main local token
to save or revoke ZDR fallback and adjust daily/monthly thresholds without
restarting the service. Until that first save, strict-server remote use remains
off even if an older environment opt-in is present. Values above $10/day or $100/month require checking the
explicit override. The saved, audited policy lives under your configured
`MUNINN_DATA_DIR`, contains no API key or transcript, and overrides older user
environment settings. A damaged or missing managed policy fails remote use
closed; queued work retains its consent generation so revocation followed by
re-enabling does not grant old jobs remote permission. Requests already admitted
before revocation may finish. To use a larger one-time historical-import budget,
explicitly raise the local threshold and separately adjust the provider key
limit, then lower both again afterward.
A restored archive can be
opened with the recovery passphrase on another machine and rebound to that
Windows user with `python -m muninn.history.secure_archive rebind --root
'<restored-archive>'`. The `restore --backup-root '<backup>' --root
'<new-private-destination>'` action copies ciphertext into a new owner-only
directory, verifies every encrypted snapshot, and leaves the backup unchanged.
The `backup` action requires a new destination, holds the archive writer lock
while copying ciphertext, and authenticates the complete backup before success.
Keep the recovery passphrase outside both the archive and its backups.
Existing older gzip history copies and source transcripts are **not**
converted or removed by strict mode; protect them and their backups separately.

The older `python -m muninn.cli history import` workflow is available only by
explicitly setting `MUNINN_HISTORY_SECURITY=legacy`. It stores transcript text
in ordinary memories and gzip copies, and is unsafe for secrets. Do not select
legacy mode for vault-first use. Details of the older workflow remain in
[`docs/CLIENTS.md`](docs/CLIENTS.md#bring-in-your-existing-conversations).

The optional `muninn-mcp[credential-vault]` extra provides a separate local
encrypted credential store and `python -m muninn.cli credentials --help` for
interactive setup, metadata search, explicit reveal, and portable backup/restore.
An explicitly selected `credentials scan` prompts for the recovery passphrase
locally, reads supported text files under approved project roots (`.env`,
configuration, source, and documentation formats) and/or encrypted history
snapshots, and stores assignment-shaped credential candidates only inside the
vault. It streams inputs, rolls a source back on mutation, decoding, or
archive-integrity failure, and reports separate coverage (`complete`) and
ambiguity (`ambiguity_free`) claims without printing values. Rejected
assignment-shaped candidates enter an encrypted, resumable review queue in
the portable credential backup; reason and queue counts contain no candidate
text. A completed scan can still have unresolved ambiguities. Archive reports
also include path-free
`error_categories` counts (`archive_integrity`, `utf8`, `io`, `metadata`,
`vault`, `other`); these sum to `errors` and help diagnose incomplete scans
without exposing source names or exception messages. Project-file values rotate in place; historical
transcript observations are labeled historical, not asserted current. Each
inaccessible directory increments `walk_errors` and `errors` without stopping
accessible siblings or the independent archive scan; the final report remains
`complete=false` and the command exits with code 2 until every coverage gap is
resolved. Project reports likewise classify `root`, `walk`, `path`, `metadata`,
`utf8`, `unsupported_binary`, `io`, `source_changed`, and `other`; `walk_errors` remains a compatible
count of root and walk gaps and must not be added to `errors` again. Source hints retain valid long and Unicode relative paths rather
than rejecting them at an arbitrary short display limit. Each
fully verified archive snapshot commits an authenticated scan receipt in the
same vault transaction as its findings, including snapshots with no findings.
The queue-enabled scanner version re-evaluates older receipts once, then skips
unchanged snapshots on later runs. After the live archive grows, rerun from
offset zero; receipts skip already processed snapshots instead of decrypting
the whole history again. A bounded local-only review pass uses an installed
Ollama model only when GPU headroom allows. A one-page review uses
`keep_alive=0`; a multi-page pass uses a 30-second idle lease to avoid reloading
the model for every batch, then Ollama frees it after the lease. It never sends
candidate text to OpenRouter or automatically promotes a candidate to a usable
credential. Clear references can be rejected by rule, high-confidence model
non-credentials can be rejected, and uncertain/possible credentials are deferred
for explicit user review. A user can inspect queue metadata, explicitly reveal
one candidate in a private local terminal, then confirm its entire value before
atomic promotion to a normal encrypted credential, or reject a false positive.
Both decisions and reveals are audited; accept/reject each require a new,
validated encrypted pre-change backup. Do not run `review-reveal` under
`Start-Transcript` or any terminal logger because it prints the candidate. The portable
vault still requires a local passphrase prompt for each review run; it does not
keep an unattended unlock key. Example, with your own
private paths (no `D:` drive is required):

```powershell
python -m muninn.cli credentials scan --root '<your-private-data-dir>\credential_vault' --project-root '<your-project-root>' --archive-root '<your-private-data-dir>\history_secure_archive' --backup-before '<new-private-pre-scan-backup-dir>'
python -m muninn.cli credentials search API_KEY --root '<your-private-data-dir>\credential_vault'
python -m muninn.cli credentials review-status --root '<your-private-data-dir>\credential_vault'
python -m muninn.cli credentials review-list --root '<your-private-data-dir>\credential_vault' --review-state deferred
python -m muninn.cli credentials backup --root '<your-private-data-dir>\credential_vault' --destination '<new-private-backup-dir>'
python -m scripts.triage_credential_ambiguity --root '<your-private-data-dir>\credential_vault' --limit 60 --model qwen2.5:7b --max-pages 200 --backup-before '<new-private-pre-triage-backup-dir>' --backup-after '<new-private-reviewed-backup-dir>' --apply
```

For a specific opaque ID returned by `review-list`, use `credentials
review-reveal --record-id <id>` only in an unlogged private terminal if you
need to inspect the exact candidate. After independently checking its source,
use `credentials review-accept --record-id <id> --confirm-exact-candidate
--backup-before <new-private-backup-dir>` or `credentials review-reject
--record-id <id> --confirm-not-credential --backup-before
<new-private-backup-dir>`. These commands prompt locally for the passphrase;
do not pass secrets as arguments or paste revealed values into agent chat.
An empty or truncated candidate cannot be promoted as an exact value.
Triage exits with code 2 while pending or deferred items still require review;
this is not a backup failure. If an error interrupts triage, its terminal JSON
reports which backup stage completed without printing candidate text; the
post-triage backup is only validated after a successful processing pass.

Only the named project roots and archive are scanned. Generated/example `.env`
files, linked paths, and unsupported assignments are not silently treated as
recovered credentials. Project text scanning supports UTF-8, BOM-marked UTF-16,
and ASCII-shaped assignments surrounded by legacy 8-bit text; `.env` and
transcript decoding remain strict UTF-8. Binary-like project files are reported
as `unsupported_binary` coverage errors, not successful scans. Check the scan
report and the source before assuming a key is absent. Documentation examples and commented-out code
may be detected: every discovered record is marked `candidate_status=unverified`,
not proof that the key is live or valid. The passphrase is not stored for background
scanning, so newly changed project files require another explicit local scan.
Nested Git repositories under a selected collection root use the repository
folder as their project label; the file hint remains relative to the selected
root. Records may include a validated selected-root-relative file hint; the agent-facing
`search_credential_metadata` tool requires `MUNINN_CREDENTIAL_AGENT_SEARCH=1`
and returns metadata only. A vault search still cannot prove absence of a key
in an unscanned, ambiguous, or unsupported source.
Pending or deferred ambiguity entries appear as `candidate_status=needs_review`
in authorized metadata searches; they never expose candidate text and cannot be
revealed through the ordinary credential API or MCP tools.
The separate `/credentials/search` and `/credentials/reveal/{id}` API routes
are disabled unless a dedicated `MUNINN_CREDENTIAL_API_TOKEN` is configured;
reveal also requires the vault passphrase in a bounded JSON body and accepts
only loopback clients. These routes are not MCP tools. Choose suitable data and
model locations via the settings above; examples do not require a `D:` drive.

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

# Rows of an older Muninn metadata.db `memories` table, exported as JSONL, keep
# their project, metadata (JSON text is parsed), memory type and archived state.
python -m muninn.cli import old-muninn.jsonl --source muninn-legacy
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
- **Credential vault**: the separate encrypted store uses a locally prompted passphrase and owner-only filesystem access. An explicit `credentials scan` can discover unverified candidates from selected project text files and encrypted history snapshots; it is not an automatic background scan and does not make unscanned sources secret-free. Existing plaintext transcripts and older copies can contain secrets; restrict access to them and their backups.
- **Auth**: protected API, MCP, and dashboard operations require a Bearer token whenever security is enabled. Set `MUNINN_AUTH_TOKEN` or `MUNINN_API_KEY` before starting a normal service; the dashboard never injects or stores it in browser localStorage. An unconfigured fallback token is not logged. `server.py` refuses an effective `MUNINN_NO_AUTH=1` or tokenless `MUNINN_DEV_MODE=true` startup unless `--allow-no-auth` is explicitly passed with a loopback bind. The Windows `launch_muninn.ps1` requires a token and never launches in no-auth mode.
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
| `docs/architecture/local-operating-model-audit.md` | Current local lifecycle, timing contract, contradictions, and release gates (audit, not completion claim) |
| `docs/plans/2026-09-28-local-control-center-overhaul.md` | Dependency-ordered localhost UI/security overhaul plan; encrypted search and local ZDR policy controls are implemented, other screens remain planned |
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
