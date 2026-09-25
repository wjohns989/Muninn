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
| 5. Merged survivor re-indexing | Open |
| 6. CI | **Done**: `tests.yml` (full suite on locked deps + clean-install import check, which also surfaced undeclared `sse-starlette`) |
| 7. Test hygiene | **Done** |
| 8. `/ingest` 500 on disabled flag | Open |
| Self-supervised adaptive importance | **Done**: see below |
| 9–13 | Open |

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
