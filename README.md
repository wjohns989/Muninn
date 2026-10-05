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

On Windows, `scripts/check_or_start_local.ps1` is a health-aware desktop entry
point. It uses the same secure single-instance launcher: an existing server is
checked, not duplicated, and an absent server is started. It then checks
authenticated access, strict encrypted-history readiness, and nonsecret counts
of failed or unfinished capture/analysis jobs. Job problems are reported with a
nonzero exit code even when the service itself is reachable; the script does
not delete jobs, start Ollama, or enable remote inference. Pass `-RepoRoot` and
`-PythonPath` for your installation. A desktop `.bat` may call this script and
pause so warnings remain visible; its paths are local to each user, not part
of the shared configuration.

For an already running Windows installation, `python scripts/reload_shared_local.py`
is read-only by default. An explicitly authorized reload requires `--restart`
and `--expected-revision` with the tested Git commit hash. Use the same Python
interpreter as the service; `--repo` and `--port` support other checkout locations
and ports. By default this procedure requires capture enrichment to be off and
the durable queues to be idle; it is not an unattended upgrade mechanism.

For an already enabled installation, add `--preserve-capture-auto`
to an explicitly authorized reload. This requires both the authenticated
effective mode and owned-process flags to confirm automatic capture and its
current remote opt-in state.
It copies the existing launch environment without reading or writing the User
capture flags. Pending/retry jobs remain intact and resume after restart; active
capture, search, analysis or publication work blocks the stop. A journal writer
fence rechecks worker state and covers the encrypted database preimages through
owned-process termination; writes may briefly wait during this maintenance
window. Backup failure releases the fence without stopping the service. Preservation
cannot be combined with activation or activation finalization.
An unclaimed `publication_pending` item is durable queued work and is retained;
a live publication lease still blocks the stop.

Adding `--enable-capture-auto` opts in to automatic, local-only processing of
**new** captures. The helper preserves all other service settings, saves encrypted
database preimages and the two nonsecret capture-flag preimages, then verifies
authentication, strict history, single-service ownership and the effective
capture mode. Only after verification does it persist the two opt-in flags in
the Windows User environment. Partial setting writes are compensated without
overwriting a detected user revocation. Failures do not claim runtime rollback;
inspect the service and preserved private logs before another action.
Windows registry writes are not atomic; a concurrent edit between checking a
flag and writing it cannot be guaranteed preserved. Do not edit these flags
concurrently with this operator procedure. Untracked program files, changed
process identity/listener ownership and linked/reparse ancestors fail closed.
Database preimages are not a complete portable archive/vault backup. The helper does not
enable historical backfill or another service. For new-capture ZDR opt-in on
an already enabled installation, use `--restart --preserve-capture-auto
--enable-capture-remote --expected-revision <tested-commit>`. This saves a
private preimage of the nonsecret User flag, requires the managed ZDR policy
and its spending caps, verifies the new process, then persists the flag for
future launcher starts. A failed remote-enabled reload stops its own new
process. Existing local-only queued windows remain parked while this mode is on.
Automatic claims are remote-only even when the local quiet/max-wait gate opens;
disabling the remote capture flag restores ordinary local eligibility. The opt-in does
not clear unresolved cost admissions or guarantee that remote inference is
available; ordinary captures and encrypted search continue locally.

For temporary catch-up while chats remain active, add
`--backlog-drain-minutes 60` to that remote-enabled reload (valid range: 1–180).
It bypasses quiet time only while at least 100 enrichment sources remain, keeps
the ordinary 30-second minimum attempt interval and single consumer, and selects
remote-bound windows without local GPU fallback. Consent/generation changes,
spending limits or uncertain accounting halt catch-up. Its absolute expiry is
passed only to the child process, not saved as a permanent setting; normal
remote-only capture cadence resumes afterward while remote opt-in stays enabled.
Existing local-bound failures are not
silently retried or converted. Status exposes the deadline and halt reason under
`capture_enrichment.backlog_drain`; this mode alone does not prove full historical
coverage or resolve credential ambiguity.

An explicitly authorized operator reload normally requires every worker to be
idle. If continuous raw transcript capture prevents that, the opt-in
`--recover-active-capture` option on `scripts/reload_shared_local.py` requires
`--restart --preserve-capture-auto` and permits at most one interrupted CPU
archive capture. Model, search and publication claims still block the reload.
Encrypted preimages and the second writer-fenced check precede the owned stop;
startup requeues the raw capture and preserves committed snapshots. Unfinished
encrypted staging files remain unreferenced and are not deleted.

Capture windows deferred as `source_not_remote_safe` remain durable and parked
while automatic remote mode is enabled. They become eligible for ordinary/local
processing when remote mode is turned off and local capacity is available. They
do not occupy planning slots or get retried by the remote-only consumer. Status
reports them separately as `parked_private_windows`; this is scheduling only and
does not grant permission to send their contents remotely. Other deferred work
still counts against the existing runnable-work limits.

Exact unchanged windows may reuse a direct earlier acknowledged interpretation
locally before a model call. OpenRouter reuse is explicitly historical coverage,
not a claim that a mutable cloud model would produce the same answer today. It
requires the same authenticated source occurrence, text, role, timestamp,
prompt/schema and an allowed original model, with settled billing and durable
original citations. It creates no new memory entries and reads no API key or
provider budget on a hit. Source rewrites and changed interpretation contracts
remain misses. This direct-parent optimization does not yet reuse through chains
of reused snapshots or merge separate sources/events with similar wording.

Older archives without commit-generation markers are not silently enrolled by
the new-capture worker. To explicitly queue historical work, use the latest-only
enrollment helper with your configured archive directory:

```bash
python scripts/enroll_history_backlog.py --archive-root /your/data/history_secure_archive
python scripts/enroll_history_backlog.py --archive-root /your/data/history_secure_archive --apply --max-batches 1
```

The first command previews at most 128 sources without migrating or writing the
journal. The second commits one bounded batch; increase `--max-batches` or repeat
it to finish enrollment. It pins an authenticated manifest and selects one latest
snapshot per source, preserving older raw snapshots and the independent live
capture baseline/cursor. Resume is idempotent; missing or altered pin/grant evidence
fails closed, including on portable restore. Windows can use the existing local
archive unlock; elsewhere add `--prompt-passphrase` for a hidden local prompt.

This helper does not call models, relax credential screening, or change provider
consent/budgets/cadence. If automatic interpretation is enabled, the existing
worker consumes enrolled windows under those controls. Status exposes
`capture_enrichment.historical_enrollment`; its `complete` flag means enrollment
finished, **not** that every window was interpreted successfully. A successful
helper exit means the requested batches committed, not that backlog processing
has finished. This latest-only pass does not extract deleted text from older
rewritten snapshots; those originals remain available for explicit retrieval.

Failed local interpretation does not require resetting an entire source. The
operator recovery helper previews eligible unsent local failures without writes:

```bash
python scripts/recover_capture_windows.py --archive-root /your/data/history_secure_archive --limit 4
```

To re-admit selected failures through the currently approved remote route, add
`--apply --expected-generation N --backup-before /your/private/fresh-preimage`,
using the consent generation shown in the preview. The helper first validates a
private encrypted journal preimage, then rechecks each job and consent. It
preserves source/window identity, completion counters and successful results;
uncertain or dispatched requests, publication work and integrity failures are
not reset. It never makes a model request itself, but the enabled service can
consume the recovered work and incur charges under its existing privacy and
budget controls. Outside Windows local unlock, add `--prompt-passphrase`.
Capacity or changed preview/consent can leave some selected windows untouched;
read the reported outcomes rather than assuming all were queued.

That helper is selective recovery. Historical processing also has a separate
OpenRouter Batch API route, not a `:batch` suffix on synchronous requests. It
uses Luna on the pinned OpenAI provider and requires an explicit, revocable
temporary-retention opt-in in addition to managed remote consent:

```bash
python scripts/configure_history_batch.py --root /your/data
python scripts/configure_history_batch.py --root /your/data --enable --max-batches 1
python scripts/configure_history_batch.py --root /your/data --disable
```

`--root` is the existing runtime directory containing `remote_policy` and
`history_secure_archive`; it is not a model directory. The mutation commands
verify the existing shared installation and take a private policy preimage.
The default opt-in is one pilot batch. A larger `--max-batches` permits automatic
catch-up under the same daily/monthly spending limits; it does not raise them.
The next batch waits for the preceding batch's settled bill, validated exact
source citations and durable publication checkpoint. An unknown submission is
not blindly retried. Inputs and results remain encrypted locally; this code
never deletes or cancels provider batches.

Each batch holds at most 128 screened windows and an 8 MiB serialized request.
If individual returned items fail schema or citation validation, Muninn retains
the original batch and successful publications, and retries only the failed
items in an encrypted, linked child batch. There are at most two repair rounds
per checkpoint, each with fresh consent/quota/budget checks and a separate bill.
Unknown submissions are never blindly resent. The parent checkpoint cannot
advance until every item has a validated, billed and durably published result;
exhausted repairs remain visibly held. No batch is deleted or cancelled.
Additional windows stay queued for later batches, without a source-size cutoff.
The service gathers screened work for at most 90 seconds, submits a full batch
immediately, then releases a partial batch at the deadline. Gathering never
delays recovery of an already owned or submitted checkpoint, and revoking either
permission resets the gathering cycle. This reduces tiny submissions without
making a full batch a prerequisite for progress.
While an unexhausted opt-in is enabled, CPU planning can prepare 128 runnable
windows instead of the normal 24, preserving eight foreground analysis slots
even after permission is revoked. CPU capture, transcript search and retrieval
continue while the provider works; optional paid interpretation remains serial.
Selection orders already planned historical windows by their source timestamps;
this is not a guarantee of global chronology across unplanned sources.
Credential/private input is excluded from this retained route; its separately
authorized ZDR handling remains distinct. Batch is not ZDR: OpenRouter retains
artifacts temporarily, and its completion window is 24 hours. Lower token rates
do not guarantee a shorter elapsed time.

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
Polls that encounter a transient SQLite writer return no-store HTTP 503 with
`Retry-After: 1`; retry polling within your existing deadline. This is not a
new durable job state, a lost result, or permission to replay a model request.
The MCP bridge sends immediate `analyze_secure_history` calls and one-use
`read_secure_history_transcript_page` calls once, without automatic HTTP replay.
A timeout or lost response can leave the outcome unknown; it does not prove
that generation was uncharged or a cursor unconsumed. Read-only job polling
keeps its existing retry behavior. Reopen an interrupted transcript through its
source capability; do not blindly repeat a potentially completed model request.
This bridge safeguard does not prevent an external client from independently
submitting another request, and is not a hard spending-cap guarantee.
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
| `search_cited_memories`, `get_cited_memory` | CPU-only lookup of encrypted source-cited candidates; retain provisional/review and truth labels, never credential values |
| `get_cited_memory_source` | Follow a memory ref to exact source coordinates and safe bounded context, then use its expiring grant to page through the credential-redacted transcript |
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
| `MUNINN_CAPTURE_ENRICHMENT` | off | Encrypted new-capture outbox with immutable enable watermark and crash reconciliation. Alone it does not run models or backfill historical snapshots |
| `MUNINN_CAPTURE_AUTO_ANALYSIS` | off | With capture enrichment enabled, automatically prepare and interpret new captures after quiet time. One CPU-only planner and one shared resource-aware model consumer; foreground search takes priority. Local-only unless capture-specific ZDR is enabled |
| `MUNINN_CAPTURE_AUTO_REMOTE` | off | Separately opt in to remote-only automatic interpretation under the managed ZDR policy and daily/monthly spending caps. Denied or private windows stay pending/parked without local inference; an uncertain/sent request never auto-retries. Old local-only jobs stay parked while this mode is enabled. Does not start a historical bulk run |
| `MUNINN_CAPTURE_QUIET_SECONDS` | `300` | Quiet period after a successful archive operation and fresh startup grace. Rejected or merely queued requests do not reset it. Valid finite range: greater than zero through 86400 |
| `MUNINN_CAPTURE_MAX_WAIT_SECONDS` | `1800` | Maximum wait before new-capture planning and a model attempt become eligible despite continuous chat capture. Must be at least the quiet period; resource and spending gates still apply. This prevents the global quiet timer from starving analysis |
| `MUNINN_CAPTURE_INTERVAL_SECONDS` | `30` | Minimum interval between capture analysis attempts, including deferred/reuse attempts; foreground analysis is not throttled. Same valid range as quiet seconds |
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

Strict-history analysis status preserves fixed local validation failure codes:
`local_output_json`, `local_output_cited_schema`, `local_output_citation`,
`local_output_quote` (missing or ambiguous exact quote), and
`local_output_analysis_schema`. These are terminal diagnostics, not permission
to weaken citations or automatically replay failed jobs. Unknown categories
remain `local_output_invalid`; rejected model text is not stored in the journal.
Older generic failures are not retrospectively reclassified. Resource deferrals
remain retryable, and an uncertain remote dispatch remains `outcome_unknown`.

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
request or other clients sharing the key can cross one. Changing Muninn's local threshold
does not raise the API key's own limit. Do not treat provider-side limits as proof
of zero overshoot: OpenRouter documents that already-dispatched requests can
finish past a workspace budget. Its multiple-interval workspace budgets are an
Enterprise feature, not a prerequisite or an assumed capability of this local
setup. See [OpenRouter workspace budget enforcement](https://openrouter.ai/docs/guides/features/workspaces/workspace-budgets).
Muninn does not yet reserve a verified upper cost bound per paid request or
claim exact simultaneous daily/monthly enforcement; automatic paid capture
fallback remains disabled until that separate accounting boundary is closed.
The dashboard shows the provider-enforced key limit, reset period, remaining
amount, and current route eligibility when remote use is enabled. This status
requires the main local token and does not expose the API key or key label.

Strict ZDR analysis additionally uses a private durable admission journal in
the managed policy database. One unresolved request blocks all further strict
paid admissions for that data directory, even across process restarts or UTC
rollovers. A pre-send reservation alone expires after 15 minutes, atomically
fencing its old worker before another admission can proceed. A reservation that
has reached the pre-POST unknown marker never expires automatically. Failed
queue cleanup cannot strand a pre-send reservation; uncertain billing still
requires explicit reconciliation. Temporary admission contention defers
catch-up instead of permanently stopping it.

Complete responses settle their reported `usage.cost`, rounded upward
to micro-USD without first converting its JSON decimal to a binary float, before
model output validation. The daily/monthly admission floor is the greater of
reported key usage and this journal's settled cost, not their sum. Calls crossing
a day/month boundary count conservatively in both periods. Proven-unsent calls
release their admission; timeout, cancellation after POST, missing billing data
or unknown billing markers remain blocking, never assumed free. Revocation
stops new admissions but does not prevent settlement or proven-unsent cleanup.
This is not a verified per-request upper cost reservation, does not cover legacy
analysis/direct external clients, and does not activate paid capture fallback.
Missing established accounting markers/tables fail closed rather than resetting
spend. Preserve the complete `remote_policy` directory and a SQLite-consistent
policy database backup, not a copied live database file or just its marker.

The authenticated local `GET /history/secure/remote-policy/accounting` returns
only cost floors and unresolved counts, without a key, transcript or admission
identifier. For operator reconciliation, these local commands reveal only
nonsecret accounting metadata:

```powershell
python -m muninn.history.remote_accounting status --root '<your-private-data-dir>'
python -m muninn.history.remote_accounting reconcile --root '<your-private-data-dir>' --admission '<id-from-local-status>' --cost-usd '<verified-actual-cost>' --confirm-outcome-reviewed
```

Reconcile only after confirming that the call is no longer in flight and checking
its actual charge (zero only when verified unbilled). Do not delete the journal,
blindly retry the model or reconcile an active request to clear a warning. The
confirmation changes the cost ledger only; it neither changes consent/budgets
nor sends an inference request. OpenRouter describes `usage.cost` as the total
account charge in its [usage-accounting documentation](https://openrouter.ai/docs/cookbook/administration/usage-accounting).

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
The `backup` action requires a new destination and holds the archive writer lock
while copying ciphertext. Backup and restore first use a private
`.incomplete-<random-id>` sibling directory, authenticate the copied stores, and
check that publication receipts identify the exact cited memories present in the
copied ledger. Only validated copies are published with an atomic no-replace
rename. A failed stage stays private and incomplete; existing destinations are
never replaced. Finalization currently supports Windows and Linux with
`renameat2(RENAME_NOREPLACE)`; unsupported systems/filesystems fail closed.
The journal and encrypted sidecars use SQLite online snapshots, not raw live DB
copies. These checks establish reference consistency, not a single snapshot time
across all writers. When managed remote accounting exists, this command creates
a self-contained runtime bundle: `history_secure_archive/`, a disabled
`remote_policy/`, and an authenticated encrypted accounting snapshot bound to
the history recovery key. Paid publication receipts, exact batch ownership,
settled costs and unresolved billing holds are verified together. Restore uses
the encrypted accounting, not the plaintext sibling database; both remote and
batch permissions remain disabled until explicit reconsent. For this format,
`restore --root` names the new **runtime directory**, and its archive is inside
`history_secure_archive/`. Managed-accounting recovery requires SQLite
serialization support (normally Python 3.11+); unsupported runtimes fail before
copying. Archives without managed accounting retain the legacy archive-root
format. This is not a full-installation backup: the separate credential vault,
API keys, environment/model settings and older stores are **not** included.
Ensure sufficient space for both backup and restore before a recovery drill. See
[the recovery decision](docs/architecture/adr-archive-backup-reference-integrity.md).
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
python -m scripts.triage_credential_ambiguity --root '<your-private-data-dir>\credential_vault' --archive-root '<your-private-data-dir>\history_secure_archive' --policy-root '<your-private-data-dir>' --provider openrouter --check-readiness
python -m scripts.triage_credential_ambiguity --root '<your-private-data-dir>\credential_vault' --archive-root '<your-private-data-dir>\history_secure_archive' --policy-root '<your-private-data-dir>' --provider openrouter --limit 60 --max-pages 200 --backup-before '<new-private-pre-triage-backup-dir>' --backup-after '<new-private-reviewed-backup-dir>' --apply
```

Credential backup authenticates records with streaming cursors and compares the
complete pinned SQLite snapshot, including receipts, audits and sequence state.
Restore copies the database through SQLite's online backup API into private
staging; it has no arbitrary database byte ceiling. Fixed-format header limits,
link/WAL checks, passphrase authentication and record integrity checks remain.
Publication never replaces an existing destination and does not reopen the
store after publication. Atomic no-replace publication supports Windows and
Linux; unavailable platform/filesystem support fails closed. These are vault
component backups, not coordinated history/remote-policy installation bundles.

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

Credential model review joins every original ambiguous assignment occurrence to
its authenticated snapshot and source unit. Project evidence and provider event
time are carried separately from archive capture time and file mtime. An absent
timestamp remains unknown; a later Codex cwd never rewrites earlier turns.
The classifier cannot reject a whole cross-source value group based on one
representative. Every matching occurrence must support rejection; uncertainty
stays pending and later rows remain reachable through keyset pagination.
Encrypted per-context decisions are reusable only for the same immutable source
and model/request identity (local model weights for Ollama). Unsupported source shapes remain pending, and integrity
failures abort. No candidate or model reply is printed by the triage runner.

The OpenRouter route uses the configured primary model, synchronous strict ZDR
and the existing managed spending caps. It never probes Ollama. Values from the
target and neighboring assignments are withheld locally; variable-name metadata,
coarse value shape and safely screened source/time context remain available to
the interpreter. Unscreenable contexts stay pending for local review and do not
block later safe candidates. Possible values cannot be automatically promoted.
Encrypted occurrence-bound dispatch receipts retain decisions before billing
settlement, so restarts and portable restores reuse received results rather than
repeat paid calls (the matching managed accounting store is also required).
Unknown transport or billing outcomes stop further dispatch.
The shared paid-admission fence defers this route while a historical batch is
outstanding; readiness checks happen before a passphrase prompt or backup.
PowerShell triage wrappers now default to OpenRouter; `-PolicyRoot` overrides the
default parent of `-VaultRoot`, and an omitted `-Model` uses the configured remote
primary. Explicit `--provider ollama` / `-Provider ollama` retains local-only
review for users who choose it. This route is not yet an automatic background
vault-unlock service or proof that the historical ambiguity queue is cleared.

The private `source-evidence` and `credential-context` sidecars are encrypted,
rebuildable, and included in portable archive backup/restore verification. Their
raw pages are not public transcript endpoints. Conversational extraction streams
large JSON strings, including pretty-printed Gemini container JSON, without
imposing a whole-message byte limit; normal agent pages stay bounded and redacted.
An archive-attached `memory-ledger` now provides encrypted durable candidates
and append-only review decisions with authenticated source citations. Only an
exact whole-user-message observation with known project/event evidence can
auto-file; even then its read contract says `unverified_assertion`, not verified
truth. Excerpts and typed model interpretations remain provisional, and possible
secrets remain pending without public text. Screening streams the entire source
unit before releasing a selected window, including across chunk boundaries.
The ledger is included in portable archive backup/restore verification.
The authenticated History dashboard reports source-version coverage separately
from recorded transcript-window job states. A model window is at most 3,000
characters; a retained batch contains up to 128 windows and can be smaller.
Retry totals include privacy-parked work, which is not runnable remote work.
Counts cover all capture-lane jobs, not just the current run, and do not prove
independently verified publication or the size of unplanned history. Linked
repair batches show their own item count alongside the original checkpoint size.
The explicit Windows reload helper fences paid admission, batch storage and the
journal before stopping the owned service. An uncertain submission or in-flight
authorization blocks reload; known authenticated provider IDs resume after it.
Worker preparation can reuse an encrypted whole-unit screening attestation
across worker instances, bound to the exact archived version, sealed parser
attempt, unit metadata and screening-policy version. It contains hashes, not
transcript text. Selected inputs remain authenticated and outgoing requests are
screened afresh. Ordinary agent reads retain their stronger whole-unit recheck;
cache reuse does not weaken that boundary. Portable verification checks the
cache too, including obsolete policy entries.

With `MUNINN_SECURE_AUTO_ANALYSIS` enabled, pertinent secure-search jobs now
bind immutable cited inputs before inference, stage validated replies encrypted,
and publish model-origin candidates before reporting success. A recovered reply
replays publication without another model call. Model interpretations remain
provisional; exact quotes do not verify truth. Local calls release their model,
and ZDR requests require persisted consent, budget admission, whole-unit privacy
screening, and final request screening. Existing running services must load the
updated code before this behavior is active.

Agents can pass an ID from `poll_secure_history_analysis.memory_refs` to
`get_cited_memory`, or discover safe claims with `search_cited_memories`.
To browse unresolved noncredential candidates, use `search_cited_memories`
with `review_only: true` and no `query`. Continue with its opaque `next_cursor`
and the same `limit` (1–20). Pages keep a stable candidate prefix while applying
current review decisions; newly appended candidates join a fresh traversal.
Credential-risk and withheld text are excluded. Reading does not approve a
claim or resolve an ambiguity. The queue opens existing stores read-only and
checks the complete authenticated ledger chain; output is bounded, but the
verification cost grows with the ledger.
Local users can browse the same queue with `python -m muninn.cli memories
review-list --archive-root '<your-private-data-dir>\\history_secure_archive'`
and resolve an exact candidate with the existing `memories review` command.
That separate mutation requires local passphrase authentication, confirmation,
the expected current state and an encrypted ledger preimage. Agents cannot
make that decision through the read-only queue.
For evidence beyond the claim, call `get_cited_memory_source`; it returns exact
source-version/unit/fragment coordinates and bounded context when the complete
unit passes the privacy check. Use its short-lived `transcript_capability` with
`start_secure_history_transcript`; if the projection is pending, check
`poll_secure_history_transcript`, then read the relevant page cursors. Reading
one page does not establish coverage of the complete transcript.
An unsafe unit returns metadata and access to the redacted projection, not raw
context. These tools require the main local token, stay loopback-only, use no
model, and do not change a provisional claim into verified truth. Keep private
context and capabilities out of logs/publication. Search is lexical over safe
candidate text/type, not a plaintext index or a full historical claim backfill.
The compact `core` profile exposes 20 tools, including the transcript-build poll
needed to finish a pending full-transcript read. Its startup instructions explain
the separate stored-memory, cited-history, transcript and credential-metadata
recall paths. Project goals are included in `get_project_context`; the standalone
`get_project_goal` is available in `full`. `update_memory`,
`set_project_goal`, and `correct_fact` are available in `full` instead of loading
those mutation schemas into every ordinary retrieval session.

Dedicated authenticated agent search and source-following tools are available;
they are not yet federated into ordinary `search_memory`. **Automatic historical
fact/task/conflict enrichment and the ledger review UI remain incomplete**.

The default-off capture outbox preserves exact committed snapshot identities even
if queue insertion is interrupted and the chat file grows before retry. Receipts,
the starting watermark and a pinned-generation resume checkpoint are encrypted
inside the portable archive journal. Reconciliation examines at most 128 entries
per batch; old/unsupported entries also consume that examination bound. It does
not invoke models or prove processing coverage, and a zero-insert batch is not
necessarily complete. Its internal typed-window queue authenticates exact plan
ordinals, reserves foreground search capacity and acknowledges coverage only after
durable publication. Automatic capture is local-only unless the separate remote
capture flag and managed remote consent are both enabled. Local/GPU processing
uses the quiet/resource gate. Approved remote processing can continue during
chat activity after the existing attempt cooldown, selecting only remotely
admitted windows and retaining foreground search priority. CPU planning likewise
need not wait for quiet when remote consent is active. Privacy failures remain
local; a remote refusal never silently runs a GPU model in this automatic remote
mode, even at quiet/max-wait or after temporary catch-up expires. Private and
local-only windows stay parked. Disable the separate remote capture flag to
resume ordinary local quiet/max-wait opportunities, still subject to foreground
priority, queue capacity and GPU availability. A resource-deferred local
opportunity resets that wait; GPU work is never forced while another application
needs it.
These workers default off; enabling both capture flags above activates local
cadence without a manual batch command. No temporary catch-up deadline or budget
is extended by persistent remote scheduling.
Completed and no-context sources leave pending selectors without deleting their
encrypted receipts. Sealed scheduler totals and per-source plans detect inconsistent
exclusion hints. New strictly appended snapshots can carry an encrypted,
authenticated same-source byte-prefix certificate. A private window matcher
requires an existing parent plan, unchanged input/partition and exact source-unit
provenance; it never deduplicates by text across occurrences or marks analysis
complete. The interpretation-contract identity is independently recomputable
and includes prompt/schema/version/model weights. Durable direct-parent reuse
acknowledgments and quiet-time cadence are implemented; measured historical
backlog scaling remains unproven. Existing manifest metadata is still loaded in full. This internal
integration is not automatic historical backfill.
Preparation failures have required encrypted per-source retry/blocked state,
so a failed source can yield to healthy work without advancing its coverage.
State changes compare the observed plan and retry seals; missing established
state fails closed. The private source status distinguishes deferred, retry-ready
and review-blocked preparation. This is scheduler fairness, not durable parser
resumption: interrupting an incomplete large-source projection can still discard
its staging. There is no source-size cutoff, but restart-cost/scaling remains
an explicit activation gate.
No model directory, drive letter or user home is assumed by this component.
New window plans derive from the already authenticated, sealed source-unit
stream instead of decrypting the original archive a third time. Publication
still requires the complete pinned fragment sequence and matching coverage
metadata. This inherits the original raw EOF proof, not a fresh raw-file check.

For a certified Codex/Claude JSONL append with compatible sealed parent evidence
and a newline-ended boundary, source preparation copies and re-encrypts the
authenticated parent units, then parses only the new suffix. Both current raw
passes still decrypt/hash the entire snapshot through authenticated EOF. Blank
lines, omitted units, per-turn project context and original timestamp coordinates
are preserved. Missing parent evidence, legacy/rewrite captures, non-newline
boundaries and Gemini use the full parser. Corruption or cancellation cannot seal
a partial child. This saves prefix parsing, not full-file I/O or encrypted parent
copying; interrupted cold parsing still restarts, and no whole-backlog speedup is
claimed from isolated regression tests.

The internal capture worker can acknowledge an unchanged window as `reused`
instead of calling a model again. Admission requires the direct parent to have an
original completed analysis, an authenticated publication ACK and matching durable
memory refs. Source occurrence, window partition and prompt/schema must match.
Local reuse also requires matching selected model weights and generation options;
OpenRouter reuse requires an allowed original model, settled accounting and current
remote consent, and records historical coverage rather than current-model equivalence.
The job keeps its current source target but exposes the original memory refs/citations,
creates no duplicate ledger events and reports reuse separately from new inference.
Local weights are rechecked after source/ledger proof; either receipt and its coverage
commit together under the current lease. Interrupted, rewritten, changed-contract or
already-reused parents do not qualify. Portable verification revalidates original
evidence, including the policy/accounting ledger for remote originals; a standalone
archive copy without that ledger is not a proved remote-history recovery. Later
model settings do not invalidate historically admitted coverage. This
default-off capture lane has quiet-time cadence, but multi-growth reuse and
measured historical backlog scaling remain incomplete; these source changes
are not new live-model tests or automatic historical-backfill activation.

Its first publication API verifies the whole event chain; large background
batches need measured amortization before activation. It detects event corruption
and deletion, but cannot detect substitution of a valid older whole database.

To check bounded real-source samples without sending inference or printing text:

```powershell
python -m scripts.smoke_source_evidence_archive --archive-root '<your-private-data-dir>\history_secure_archive'
python -m scripts.smoke_memory_ledger_archive --root '<your-private-data-dir>\history_secure_archive'
```

The ledger check previews at most eight bounded real snapshots. Adding `--apply`
stores at most one encrypted source observation and verifies durable reopening;
it sends no inference and prints no source text. This is a component check, not
a full historical import. Preview may build encrypted source-evidence sidecars,
but does not publish a memory candidate. Paths and installed model tags remain
user-configurable.

For an explicitly requested **single real-source inference check**, without
publishing memories or changing policy:

```powershell
python -m scripts.smoke_memory_ledger_archive --root '<your-archive-dir>' --local-analysis-preview
python -m scripts.smoke_memory_ledger_archive --root '<your-archive-dir>' --zdr-analysis-preview --policy-root '<your-private-data-dir>'
```

The ZDR check can incur a provider charge and requires an enabled managed policy
at the explicit data directory. It never infers consent from an archive location.
Output contains counts/status, not original chat text. A short source preview
does not prove historical coverage or extraction accuracy on the backlog.

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
