# Post-Codex Hardening Plan (v3.25.0)

> Date: 2026-09-25. Baseline: `main` after PR #57, which carries the latest Codex work
> (PRs #134–#139, all created and merged 2026-08-24). Nothing newer reached GitHub between
> 2026-08-24 and 2026-09-25; any later Codex work exists only on the operator's machine.

## Where Codex left off

Codex's August pass modernized resource lifecycle and transport (bounded Streamable HTTP,
image memories, `.env` loading, Codex config repair, Jina Turbo reranker) and closed every
review thread it received. Two items were explicitly left pending:

1. **Release verification.** PR #137: "The live machine-wide service ... remains on merge commit
   `d869003` pending merge and isolated release verification." Deployment needs an approved restart.
2. **Vector-quality evidence.** PR #134: the local FastEmbed cache was corrupt, so its quality
   benchmark ran on lexical fallback. It proved no consolidation regression, not retrieval quality.

Codex's working conventions, which this plan keeps: adopt only findings reproduced against
current code with a failing regression test first; bounded state reported content-free in
`/health`; Ruff on touched code, `git diff --check`, and a gitleaks scan before push; no
machine paths, credentials or runtime artifacts committed; a rollback note per change.

## Already landed on `claude/kind-cray-qifgx9`

- `aiohttp` declared (clean installs could not import `muninn.core.memory`).
- Huginn dashboard reports real ingestion results and server errors; bulk legacy import counts fixed.
- Consolidation persists decay/merge/promote/shadow results (all were no-ops); merge no longer
  deletes the surviving record when the secondary wins.
- Centrality floored at one graph relation for entity-free memories.
- CoALA session inhibition (opt-in via `session_id`; MCP forwards HTTP/SSE/stdio session keys),
  with content-free utilization in `/health`.

## Status (updated 2026-09-25, PR #140)

| Item | State |
|---|---|
| Dense vector recall under the default `archived=False` filter | **Fixed** (found during this work; the vector signal returned nothing for ordinary memories) |
| 1. Consolidation coverage | **Fixed**: persisted id cursor per phase |
| 2. Replay | **Fixed**: `update_vector` preserves payload; archived rows skipped |
| 3. Decay threshold / retention | **Fixed**: stored novelty; decay, merge and temporal shadow archive (reversible, `POST /restore/{id}`); dry-run mode |
| 4. Deletes leaking index entries and files | **Fixed** for working-memory TTL (the only remaining hard delete) |
| 5. Merged survivor re-indexing | **Fixed**: survivor re-embedded, payload content and BM25 updated |
| 6. CI | **Done**: `tests.yml` (full suite on locked deps + clean-install import check, which also surfaced undeclared `sse-starlette`) |
| 7. Test hygiene | **Done** |
| 8. `/ingest` 500 on disabled flag | **Fixed**: 409 with the flag to set |
| Self-supervised adaptive importance | **Done**: see below |
| BM25 after restart | **Fixed** (found this round): the startup rebuild dropped user/namespace scope, so user-scoped searches skipped every pre-restart memory; it also stopped at 10,000 records |
| Consolidation crash on archive/merge | **Fixed** (found this round): `VectorStore.delete` rejected the list the daemon passes; covered by a real-store contract test |
| Learner feedback loop | **Fixed**: the learned score decides retention only; ranking keeps the hand-weighted importance |
| Reindex / legacy import / legacy detection | **Done**: `python -m muninn.cli reindex|import`, `/admin/*`, `legacy_stores` in `/health` |
| MCP 2026-07-28 | **Done**: dual-era endpoint (stateless modern requests, `server/discover`, header validation; legacy sessions unchanged) |
| MCP client accessibility | **Done** (see "Client accessibility review" below) |
| 9. Benchmarks with real vectors | Open |
| 10. Session inhibition follow-ups | **Done**: `inhibited` flag on results, SDK `session_id`, explicit `session_id` tool argument |
| 11. Feedback weighting | Open (measure once the learner has data) |
| 12–13. Release, PR backlog | Open |
| Kùzu replacement, embedding upgrade, lint pass, branch pruning | Open (see "Next cycle") |

### Self-supervised adaptive importance

Muninn learns importance from its own retrieval history without human feedback
(`muninn/scoring/adaptive.py`). Each search logs access events; each cycle snapshots
ACT-R activation features for a sample of memories; after the horizon, the LEARN phase
labels each snapshot by whether a new session retrieved the memory again, scores the
stored prediction, then updates an online logistic model. Decay uses the learned score
only while its rolling AUC beats the hand-weighted score, and only for memories older
than the horizon. Memories archived during a window are censored.

Known risks to watch once it is live:
- **Feedback loop.** Once active, learned importance feeds the ranking boost
  (`0.7 + 0.3 x importance`), which influences future retrievals and therefore labels.
  Mitigations: labels require a new session, session inhibition diversifies results,
  and the gate falls back to legacy if AUC degrades. If `/health` shows drift, add
  small exploration (occasional unboosted ranking) and position-bias weighting from
  the logged ranks.
- **Access log size.** About one row per returned result, pruned after
  max(4 x horizon, 90 days); watch it on busy stores.

### Client accessibility review (2026-09-25)

Checked with the official MCP Python SDK client (`mcp` 2.2.0) over stdio and
Streamable HTTP in legacy, auto and 2026-07-28 modes; all 12 combinations pass.

| Finding | Fix |
|---|---|
| `set_project_goal`, `detect_information_gaps`, `trigger_distillation`, `correct_fact`, `forage_knowledge` were advertised but returned "Method not found" (dispatch entries lost in the modular refactor) | Routed; a test now fails if any listed tool lacks a dispatcher |
| 2026-07-28 list and discover results lacked the required `cacheScope`/`ttlMs`, so the official SDK rejected every modern connection | Added; `tests/test_mcp_wire_schema.py` validates every method and version against `mcp-types` |
| Tool failures were JSON-RPC errors the model never sees | Returned as `isError` results (spec 2025-11-25), including backend 4xx bodies |
| Every tool claimed `openWorldHint: true`; overwrite tools were not marked destructive | Correct hints plus `title` |
| No Origin validation and CORS `*`, with no token by default: any web page could read or delete memories | `OriginGuardMiddleware` (localhost plus `MUNINN_ALLOWED_ORIGINS`), CORS on the same allow-list |
| 37 tools exceed Cursor's 40-tool budget with other servers; ChatGPT outside Developer Mode needs `search`/`fetch` | Tool profiles `full`/`core`/`readonly`/`chatgpt` via `?toolset=` or `MUNINN_MCP_TOOLSET`; ChatGPT `search`/`fetch` with `outputSchema` and `structuredContent`; `GET /memory/{id}` |
| No per-client setup guide | `docs/CLIENTS.md` |

### Cross-agent handoffs and built-in prompting (2026-09-26)

Goal: Claude Desktop (and its Code tab), Codex and ChatGPT Work in the ChatGPT
desktop app, Claude Code and any other local agent share one store and pass
projects to each other.

| Finding | Fix |
|---|---|
| The project came from the MCP process's working directory: over HTTP that is the server's own folder for every client, and desktop apps start stdio servers outside the repository | `project` argument on the tools; HTTP never uses the server's directory; stdio uses git only inside a repository, else `MUNINN_PROJECT`, else `global` |
| Memories never recorded which agent wrote them (`source_agent` was always `unknown`) | Agent from MCP `clientInfo` (aliases such as `claude-ai` → `claude-desktop`), `?agent=` or `MUNINN_AGENT_NAME` |
| Handoffs existed only as export/import bundles between stores; notes over 1000 characters were split into chunks | `agent_handoffs` table (outside consolidation) with open → claimed → done lifecycle; tools `create_handoff`, `resume_handoff`, `complete_handoff`; REST `/handoffs` |
| No session-start routine | `get_project_context` / `GET /context`: goal, open handoffs, rules, recent memories by agent, global preferences |
| Server instructions were one sentence; prompts were empty stubs | The shared-memory protocol ships as server instructions (read by Codex, Claude Code, Claude Desktop, Gemini CLI), per toolset; MCP prompts `start`, `resume`, `handoff`, `remember` on every transport |

Verified with the official MCP SDK: Codex over HTTP (2026-07-28) stores a
memory and a handoff; Claude Desktop over stdio, started outside any
repository, sees both in its briefing, claims the handoff and completes it.

### Local conversation history (2026-09-26)

Legacy discovery found transcript files but imported them as fixed-size raw
chunks, all under one project, stamped with the import time. It also scanned
folders that hold credentials: Gemini's `oauth_creds.json` matched
`~/.gemini/**/*.json`, and the Claude and ChatGPT app-data scans matched
`claude_desktop_config.json` and similar files. And it ignored `CLAUDE_CONFIG_DIR`,
`CODEX_HOME`, Codex `archived_sessions` and `.zst` rollouts.

| Piece | What it does |
|---|---|
| `muninn/history/locations.py` | App locations from the apps' own settings (`CLAUDE_CONFIG_DIR`, `CODEX_HOME`, per-OS app data, `MUNINN_HISTORY_HOMES`) plus their retention settings |
| `muninn/history/vault.py` | Compressed private copy in `<data dir>/history_vault`; never deletes; keeps versions when a file shrinks; marks files the app deleted |
| `muninn/history/parsers.py` | Claude Code (queued mid-turn messages, compaction summaries, Desktop titles), Codex (current and early layouts, `.zst`, thread titles), Gemini, ChatGPT/Claude exports, prompt histories |
| `muninn/history/importer.py` | One memory per turn (ordered parts, not truncated) with original time, project (git remote or repository; worktree-aware), directory, branch, agent, thread id and turn number; compaction and thread-summary memories; secrets redacted; incremental |
| `muninn/history/service.py` | Sync every 30 min; automatic import after the first applied import |
| Interfaces | `/history/*`, `python -m muninn.cli history`, MCP `get_thread` and `import_agent_history`, `recent_threads` in `get_project_context` |

Legacy discovery now honours the relocation variables, never reads credential
files, and its "import all" skips the agent transcripts the importer handles.
Verified on a real Claude Desktop transcript: 19 turns and a compaction
summary, all under project `Muninn`, re-read in order and found by search.

### Hooks and thread understanding (2026-09-26)

- `POST /hooks/{agent}` plus `python -m muninn.cli hooks install`. Claude Code
  uses `http` hooks; Codex uses command hooks through `muninn/hook_client.py`
  (standard library only, about 50 ms startup, always exits 0). SessionStart
  injects the project briefing. PreCompact, Stop (throttled) and SessionEnd
  vault and import that one thread in the background; the endpoint answers in
  about 10 ms.
- `history analyze` (`muninn/history/insights.py`): per thread, a model
  extracts a summary, status, topics and insights (decision, preference,
  convention, fact, fix, open_item). Each insight is stored as a semantic
  memory dated to its source turn. Recent in-progress threads with open items
  become handoffs. Providers: OpenRouter, with `provider.zdr = true` and
  `data_collection = deny` on every request, or local Ollama. The thread
  catalog gains status and topic filters.
- Import speed: batching fastembed gave no gain on the test CPU (8.1 s one at
  a time vs 9.2 s batched for 64 turns), so it was not added. Imported turns
  skip the per-turn LLM entity extraction instead. Cloud embeddings would force
  a store-wide re-embed and send every future memory to the cloud, so they are
  documented as a trade-off rather than enabled.

### One timeline across apps; validated analysis (2026-09-26)

- Turn fingerprints (original time + text) stop resumed or forked sessions
  from storing copied turns twice. The new thread records `continues_thread`.
  The project timeline (`history timeline`, `/history/timeline`, `get_thread`
  with `timeline=true`) interleaves every app's turns with handoff events and
  agent switches.
- Analysis reads each project's threads oldest first, with "Meanwhile"
  markers and other threads' timestamped insights. Superseded insights are
  archived; re-analysis replaces a thread's insights.
- OpenRouter: the CLI asks for a key on first run (verified, saved with
  owner-only permissions). Default `openai/gpt-6-luna` with DeepSeek V4 Flash
  and Gemini 3.5 Flash-Lite fallbacks, chosen from the live ZDR endpoint list.
  The previous default, `google/gemini-2.5-flash`, expires 2026-10-20.
  Requests use a strict JSON Schema with `require_parameters`, and send only
  parameters all three models' ZDR endpoints accept. Every reply is validated
  before storage.
- Whole conversations go in one call. The old 2,500-character cap per turn
  dropped about 60% of a real long session before any model saw it.

Next for this area:

- MCP resources (project briefing as `muninn://project/{name}`) for hosts that
  attach resources without a tool call.
- Show handoffs and agent labels in the Huginn dashboard.
- An `.mcpb` Desktop Extension for one-click Claude Desktop install.
- OAuth 2.1 on `/mcp` so ChatGPT and claude.ai can connect directly without a
  tunnel.
- The stdio wrapper logs the generated temporary token to stderr, and hosts
  keep stderr in log files; log that a token was generated, not its value.
- Drop the non-existent `2025-11-05` from `SUPPORTED_PROTOCOL_VERSIONS`.

## P0 — consolidation correctness (silent failures)

Each of these was reproduced against current `main`.

### 1. Consolidation only ever sees the 500 most important memories
- **Evidence:** `SQLiteMetadataStore.get_for_consolidation` runs
  `ORDER BY importance DESC LIMIT ?`; decay and merge pass 500, promote passes 500, and replay passes 100.
  Once a store exceeds 500 memories, low-importance records, which are the ones decay exists for,
  are never evaluated.
- **Fix:** page through the table with a persisted rowid cursor (`set_meta`/`get_meta`) so every
  record is visited over successive cycles; keep the per-cycle bound. Promote should filter in SQL
  (`min_access_count`) instead of post-filtering a top-importance slice.
- **Validate:** real-store test with 1,200 memories: after three cycles every record has been
  re-scored once; per-cycle work stays at the bound.

### 2. Replay phase never re-embeds anything
- **Evidence:** `_phase_replay` calls `vectors.upsert(doc_id=..., vector=..., payload=...)`, but
  `VectorStore.upsert` takes `(memory_id, embedding, metadata)`. With an autospec store every call
  raises `TypeError`, which the phase logs and swallows (`re_embedded: 0`). Existing tests use plain
  `MagicMock`, which accepts any keywords.
- **Trap:** fixing only the signature would replace the point payload with four fields and drop
  `user_id`, `project`, `scope` and `media_type`, hiding re-embedded memories from scoped searches.
- **Fix:** add `VectorStore.update_vector(memory_id, embedding)` using Qdrant `update_vectors`
  (payload preserved). Use `create_autospec` in daemon tests.

### 3. Decay can never delete: threshold is unreachable
- **Evidence:** decay passes `max_similarity=0.0`, so novelty is always 1.0 (0.25 of importance).
  The lowest reachable importance is 0.295 before the centrality floor and 0.34 after (10-year-old,
  never accessed, `INGESTED` provenance), against `decay_threshold = 0.1`. With the real stored
  novelty of a near-duplicate (similarity 0.95) the same memory scores 0.103.
- **Fix:** score decay with the persisted `record.novelty_score` computed at ingestion
  (`calculate_importance` gains an optional `novelty` override).
- **Decision needed (operator):** retention policy. Recommended: archive (`archived = 1`,
  column exists since v3.24.0, excluded from search, recoverable) instead of hard delete, with the
  threshold calibrated from the importance distribution of the memory-profile benchmark corpus.
  Hard deletion stays available behind an explicit setting.

### 4. Consolidation deletes leak index entries and files
- **Evidence:** expired working memories are removed from SQLite and Qdrant only (BM25 and graph keep
  them). Neither decay nor expiry calls `cleanup_managed_images`, so image files outlive their
  memories, which is the gap Codex closed for API deletes in #137/#138.
- **Fix:** route every consolidation removal through one helper that mirrors `MuninnMemory.delete`
  (metadata, vectors, graph, BM25, managed images), or archive per item 3.

### 5. Merged survivor is not re-indexed
- **Evidence:** merge rewrites the survivor's content but leaves its vector, vector payload `content`
  and BM25 document at the pre-merge text, so the absorbed text is unsearchable by keyword or vector.
- **Fix:** re-embed via `update_vector`, `set_payload` for `content`, and re-add to BM25.

## P1 — engineering reliability

### 6. CI does not run the test suite
- `benchmark.yml` and `transport-incident-replay-gate.yml` run no unit tests beyond two protocol
  files, and those end in `|| true`. Every Codex PR relied on local runs.
- **Fix:** add `tests.yml`: `uv sync --frozen --extra dev --extra sdk --extra ingestion`,
  `PYSTRAY_BACKEND=dummy`, `pytest --timeout=180`. Remove `|| true`. Add a clean-install import smoke
  (`pip install .` then `python -c "import server"`), which would have caught the `aiohttp` gap.

### 7. Test hygiene
- `tests/test_concurrency.py` asserts multi-process access to embedded Qdrant/Kuzu, which both
  libraries refuse by design and which the single-owner topology forbids. It fails on `main` and
  before #57. Rewrite it to assert the single-owner `StoreLock` behavior.
- `test_shared_fastapi_server_exposes_streamable_http_route` compares one route's method set; FastAPI
  0.141 registers methods per route, so it fails although the transport works (the other 15 HTTP
  transport tests pass on 0.141). Assert by dispatching requests instead of introspecting routes.

### 8. `/ingest` returns 500 for a disabled feature
- `MUNINN_MULTI_SOURCE_INGESTION` is opt-in; when off, `/ingest` raises a 500. SDK and MCP callers
  see a server error. Return 409 with the flag name (the dashboard now shows `detail` either way).

## P2 — quality validation and enhancements

### 9. Retrieval-quality benchmark with real vectors
- Re-run `eval.memory_profile_benchmark` with a working FastEmbed cache and compare Recall/MRR/nDCG@5
  and p95 latency for: Jina Tiny vs Turbo (`MUNINN_RERANKER_MODEL`), centrality floor, and session
  inhibition penalties 0/3/limit on multi-turn query sequences. Publish aggregates only.

### 10. Session inhibition follow-ups
- Mark demoted results in recall traces (`inhibited: true`) so agents and Huginn can explain ordering.
- Add `session_id` to the Python SDK `search()`; consider opt-in inhibition for `hunt_memory`.

### 11. Feedback weighting check
- Helpful feedback now raises importance through both the Elo half-life and the SNIPS utility term.
  Measure on the benchmark whether this over-weights frequently rated memories; tune Elo `k_factor`
  or the `retrieval` weight if needed.

## P3 — release and backlog

### 12. Release v3.25.0
- Bump `pyproject.toml` and `muninn/version.py`, add a changelog entry, and run Codex's isolated
  verification (memory-profile benchmark with 30-minute workload and idle soaks on a random port and
  temporary stores). Then restart the machine-wide service with operator approval. That step is
  local-only.

### 13. PR backlog (54 open)
- Once CI (item 6) exists, let it validate the clean test-only PRs and merge the green ones; rebase
  or close the nine perf PRs that conflict with `main` (#59, #105, #115, #116, #119, #125, #126,
  #127, #129). Keep Codex's rule: adopt only changes reproduced against current code.

## Suggested order

1 → 2 → 4 → 5 (mechanical, test-first), then 6 → 7 so CI guards everything after. Item 3 waits on
the retention decision. Items 9 and 11 need a working vector cache. Item 12 closes the cycle.

---

# Next cycle: ecosystem changes, migration, cleanup (researched 2026-09-25)

## Ecosystem changes that affect Muninn

| Change | Impact on Muninn | Plan |
|---|---|---|
| **Kùzu archived** 2025-10-10; 0.11.3 is the final release. Community fork **LadybugDB** (v0.19.1 as of 2026-08) continues it; on-disk files are not a portable interchange format. | The graph store and TKG sit on an unmaintained engine (no security fixes, no new Python/platform wheels). | Pinned `kuzu==0.11.3` (done). Then put the graph behind an interface and migrate by **logical rebuild from SQLite** (not file copy), either to LadybugDB or to SQLite-backed entity/relation tables. Muninn's graph use (entity links, degree, temporal edges) is small enough that SQLite would also remove the second single-process store. Decide with a benchmark. |
| **MCP spec 2026-07-28** is stateless: `Mcp-Session-Id` and the `initialize` handshake are removed; protocol version, client info and capabilities travel in `_meta` on every request; HTTP+SSE transport, Roots, Sampling and Logging are deprecated. | The Streamable HTTP transport negotiates up to 2025-11-25; session inhibition keys off the MCP session id; the legacy SSE router is now a deprecated transport. | Add 2026-07-28 support alongside 2025-11-25 (per-request `_meta` version, `UnsupportedProtocolVersionError`), derive the inhibition key from an explicit tool argument or client identity instead of the removed session header, and schedule the SSE router for removal. |
| **Benchmarks**: LongMemEval (500 questions), LoCoMo (1,540) and BEAM (1M/10M tokens) are the reference set; published results are ~94–96% on LongMemEval and ~92–94% on LoCoMo. | Muninn only runs a synthetic LongMemEval-style set, and vector recall was broken until PR #140, so it has no comparable number. | Run the real LongMemEval-S and LoCoMo through the existing adapter with a working vector cache; add BEAM-1M later. This becomes the regression gate and the offline check for the adaptive learner. |
| **Local embeddings**: `nomic-embed-text` remains a sound small default; `embeddinggemma` (768-d, same dimension) and `qwen3-embedding:0.6b` (MRL, adjustable dims) score higher on MTEB. | Changing models means re-embedding every memory. | Needs the reindex tool below, then an A/B on the benchmark before switching the default. |

## Migrating an existing local install

The repository starts at v3.0.0 (2026-02-11), which replaced the Mem0-based Muninn. What an upgrade needs
depends on which generation the local install is:

- **3.x install** (data under the platform data dir, e.g. `%LOCALAPPDATA%\AntigravityLabs\muninn`, containing
  `metadata.db`, `qdrant_v8/`, `kuzu_v12/`): upgrades in place. SQLite adds new columns and tables on
  startup, BM25 is rebuilt from SQLite, and the vector and graph directory names have not changed since
  3.0. The vector-recall fix is query-side, so existing points need no rewrite. Back up the data dir, stop
  the service, update, start with `MUNINN_CONSOLIDATION_DRY_RUN=1`.
- **Pre-3.0 (Mem0-based) install** (served on port 8000, `mem0ai` dependency, data in Mem0's own store):
  there is no migration path today. The 3.0 design called for a Mem0 → native migration script and a
  `MIGRATION.md`; neither was written.

Tooling to build (always operating on a backup copy, dry-run by default):
1. `muninn reindex` — rebuild vectors, BM25 and graph from `metadata.db`. Also the mechanism for an
   embedding-model change and for leaving Kùzu.
2. `muninn import-mem0` — read memories from a running Mem0-era server's API (or its store), preserve
   original `created_at`, user and metadata, tag provenance `legacy_import`, skip duplicates by content
   hash, report counts before writing.
3. Wire the unused `platform.get_legacy_data_dir()` into startup so a legacy layout is detected and reported
   in `/health` instead of silently starting empty.

## Repository cleanup

- **Privacy guard (done):** gitleaks in CI and pre-commit, tracked-but-ignored check, home-path check,
  `.gitignore` for the relative data dir and images; untracked outputs containing a real profile path.
  History contains no secrets; three old commits contain a Windows profile path, which only a history
  rewrite on `main` would remove (not recommended).
- **Root clutter (done):** historical handoffs and remediation reports moved to `docs/archive/`;
  the unused root `package.json`/`package-lock.json` (Node Claude Agent SDK) removed. `fix_fastembed.py`
  (referenced by the engine for corrupt FastEmbed caches) and `ingest_history.py` (current, port 42069)
  stay.
- **Lint:** 1,173 auto-fixable Ruff findings (whitespace, unsorted/unused imports). One mechanical PR,
  then Ruff in CI so it stays clean.
- **Branches:** 110 remote branches, most from closed or merged PRs. Delete after review.
- **PR backlog:** 54 open; CI now validates them.

## Sources
- Kùzu archive and LadybugDB: https://oneuptime.com/blog/post/2026-08-12-kuzu-archived-pin-0-11-3-fork-or-migrate/view
- MCP 2026-07-28 changelog: https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2026-07-28/changelog.mdx
- Memory benchmarks: https://mem0.ai/blog/ai-memory-benchmarks-in-2026 and https://mem0.ai/blog/state-of-ai-agent-memory-2026
- Local embedding models: https://www.morphllm.com/ollama-embedding-models
