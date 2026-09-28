# Local Muninn operating-model audit (2026-09-28)

Status: **architecture audit, not an implementation-complete claim**. This maps the
real Windows installation and the intended portable product. It separates live
observations, code behavior, proposed service-level targets, and unknowns. It
does not authorize a historical cloud sweep, a credential scan, or a release.

## Outcome and invariants

Muninn should capture agent history without making chat hooks slow, preserve it
in a portable encrypted archive, make all supported text searchable regardless
of source length, return bounded evidence on demand, and use a model only when
interpretation is useful. It should leave VRAM unoccupied between local calls.
Credentials discovered in source material belong only in the encrypted credential
vault; normal agent search returns metadata, never values. Remote interpretation
requires the user's durable ZDR opt-in and an enforceable spending boundary.

The invariants are:

1. A successful capture acknowledgement has a durable recovery path. Every stage
   is idempotent and can resume after crash, shutdown, file growth, or timeout.
2. Source length never changes a supported text item into a permanently
   unsearchable or uninterpretable item. Work is split into bounded windows;
   insufficient time, space, VRAM, or budget causes a **retryable** state.
   Unsupported formats and corrupt input are distinct from size.
3. No unbounded plaintext, model prompt, SQLite transaction, memory allocation,
   GPU residency, or synchronous HTTP request is required for a large source.
4. Source text, model inference, extracted claims, and verified operational facts
   have different trust levels and cannot silently overwrite one another.
5. Credential values are not in ordinary memory, searchable index records,
   logs, agent tool results, or repository files. Explicit local use/reveal is
   authenticated and audited; portable recovery works from a passphrase.
6. One local service owns each writable store. The repo, installed code, live
   process, configuration, data, and backup must be identified independently.

## What is running versus what is only code

| Area | Current evidence | Current conclusion |
| --- | --- | --- |
| Service | Local `/health` returned `ok`, 1,439 ordinary memories and 1,439 vectors during this audit. | Service is running; health does not prove strict capture or model enrichment. |
| Archive | Prior authenticated backup at generation 54 covered 3,871 snapshots and 15.16 GB plaintext; a read-only index check during this audit reported 3,871/3,871 ready, zero missing/unsearchable. | Search coverage is current for those archived snapshots, not proof of perpetual backup/capture coverage. |
| GPU | `ollama ps` showed no resident model during this audit. | Idle residency is zero at this instant; the health field `runtime_resources.embedding.loaded=true` does not itself prove VRAM residency. |
| Large source | A focused direct diagnostic fetched a redacted hit from the former 1,062,682,447-byte overflow source; combined search/fetch took about 30 seconds. | No current encrypted-index size cutoff for that source; global search and latency under multiple candidates remain unproven. |
| Code versus live process | The large-JSONL structured-fetch improvement is in the current branch but is **not loaded in the live service**. An experimental authenticated-window reader was removed after independent review. | Do not describe a branch test as a running feature; the rejected reader must not be deployed as-is. |
| Clients | Installed hook entries were observed previously for Codex, Claude Code, and Gemini CLI. | Configuration presence is not proof that real events reached and durably captured at the shared endpoint. |
| Credential vault | Portable vault and backup exist; initial backup had zero records. | No project/transcript discovery pipeline or recovered real credential record is proven. |

## End-to-end lifecycle and timing contract

Times in the “target” column are **proposed engineering budgets**, not measured
guarantees. Large-work completion must be asynchronous and proportional to bytes
or model tokens; the request/ack path must not wait for it.

| Stage | Trigger, owner, durable output | Existing behavior | Proposed timing / checkpoint |
| --- | --- | --- | --- |
| Startup | One service opens private stores, archive, and policy; publishes health. | Shared launcher avoids a duplicate healthy listener; strict archive must unlock. | Health only after stores are ready; restart/rollback proof on exact installed code. |
| Session briefing | Client `SessionStart` hook asks for project context. | Hook bridge waits up to 8 s; installed hook timeout is 10 s. | Aim under 2 s; failure returns an empty briefing without blocking chat. |
| Chat capture request | `Stop`/`AfterAgent`/compaction/end hook names a validated transcript path. | Bridge waits 0.8 s for fast events. Server schedules in-memory background capture, returns immediately. | Ack under 0.6 s **after a durable job record**; source may grow while queued. |
| Encrypted capture | Single archive writer reads source and commits a versioned manifest. | Strict-mode debounce is 600 s unless forced; no periodic strict sync. | Begin new/changed small files within 60 s; large files report byte progress; retry until archive version and source fingerprint reconcile. |
| CPU index | Archive commit wakes one background builder. | Up to 20 snapshots per pass; retry after 30 s while missing, otherwise 300 s; no model. | Indexed state follows durable archive version; report backlog, bytes, and retry cause, not just counts. |
| Search | Authenticated API/MCP query checks blind filters, then verifies candidates. | MCP HTTP timeout 40 s; up to 20 candidate full decryptions; search has no job/poll protocol. | Fast small queries in seconds; long verification returns a job id before client timeout and supports resume/cancel. |
| Fetch | Short-lived capability requests one redacted excerpt. | One slot; 0.1 s acquire; 10/minute; 4,000-character maximum; capability lives 10 minutes. | Bounded page with explicit cursor and provenance; first full verification may be long, so it too needs an async state when over the request budget. |
| Interpretation | Pertinent evidence triggers local route or approved ZDR remote. | Strict mode offers ephemeral one-hit `analyze`; it does not schedule automatic enrichment. Local Ollama sets `keep_alive=0`. | Queue only relevant windows, checkpoint each, concurrency one on local GPU, re-probe VRAM before each call; never keep a model resident while idle. |
| Remote fallback | Local route defers, or explicit remote preference; ZDR key/cap checked. | Local opt-in **and per-call** allowance required; provider cap checked before a call. | Persist consent/budget policy locally, re-check every call, record actual billed usage, expose revocation; no fallback after a failed local request without a distinct policy decision. |
| Credential discovery | Project file or transcript version is scanned locally. | Vault API/CLI exists but no scanner is wired; zero-record initial backup. | Queue by source fingerprint, isolate candidate values before ordinary indexing/model egress, deduplicate, audit false positives and explicit use. |
| Backup/recovery | Archive/vault ciphertext snapshot and passphrase recovery. | One authenticated archive backup and initial empty credential backup are known. | Automatic consistent incremental cadence and restore drill; never report a backup as current without its generation, time, and restored counts. |
| UI | Local dashboard displays state and user decisions. | UI exists, but no full consent/budget/coverage/resource workflow. | Last implementation priority; first fix the root-page token disclosure described below. |

The current hook-to-archive chain is especially important: the hook's success
means only that the server accepted an in-memory task. It is **not** proof that
the snapshot reached the encrypted manifest or CPU index.

The target durable state machine should be keyed by immutable archive version,
not only by a changing source path:

```text
source event -> capture_queued -> capturing -> archived
                                            |          |
                                            |          +-> index_queued -> indexed -> searchable
                                            |          |
                                            |          +-> relevance_queued -> window_queued
                                            |                                 -> interpreting -> provisional
                                            |                                                  -> verified only with evidence
                                            +-> retryable_failure(reason, next_attempt)
```

Indexing and interpretation are independent branches: GPU/budget failure must
not delay capture or search. A new source version gets a new identity while old
snapshots remain recoverable. `done` means the specific stage committed, never
“all history is understood.” A durable queue would coalesce repeated hook
events but must not suppress the final changed version.

Capacity accounting must be explicit. The archive encrypts 1 MiB plaintext
chunks. A 15.16 GB corpus divided into 3,000-character prompts is on the order
of **millions of model windows** if every byte is interpreted; that is why an
automatic full-corpus model sweep is neither a latency target nor a sensible
default. Historical search is CPU/index work; model work is triggered by
relevance and bounded per period. For any large operation, ETA must be derived
from measured bytes/sec or tokens/sec on this machine and shown as a range with
backlog, rather than a fixed promise. Queue limits should bound resident memory,
temporary disk, concurrent decryptions, GPU calls, and remote spend separately.

## Contradictions and blocking gaps, ordered by consequence

| Priority | Conflict / counterexample | Required design change and proof |
| --- | --- | --- |
| P0 | `GET /` replaces `{{MUNINN_TOKEN}}` with the main bearer token without an authentication dependency (`server.py`, dashboard root). Loopback and Origin checks do not authenticate another local process. | Stop embedding a bearer in anonymous HTML. Use explicit local login/session or a narrowly scoped UI token; verify anonymous GET cannot recover the main token and existing UI still works. |
| P0 | Strict capture is scheduled only in process memory, while hooks return success; a crash can discard it. The 600 s debounce plus absent strict periodic sync can leave a growing transcript uncaptured. | Durable capture journal before hook ack; coalesce by source identity; retry with backoff and a startup reconciliation scan; prove crash between ack and archive commit recovers. |
| P0 | Search/fetch are synchronous through a 40 s MCP timeout, yet one selected 1 GB direct search-and-fetch took about 30 s; multiple candidate verifications can exceed that. | Deadline-aware asynchronous job/poll or indexed authenticated chunk offsets; test slow candidate, expiry, cancellation, and no duplicate work. Raising one timeout alone is not a fix. |
| P0 | Strict mode has no automatic interpretation: `HistoryService.start()` starts only CPU indexing, and `_launch_auto_analysis()` exits in strict mode. | Query-driven priority queue plus durable per-source/window cursor; new captures and pertinent searches can enqueue work. Avoid a blind model sweep of all 15 GB. |
| P0 | Export parser uses whole-file `json.loads` and ZIP reads; a huge export can fit the encrypted index but still exhaust RAM during semantic extraction. | Streaming format-specific parser and bounded conversation/turn windows; corrupt/binary/unknown formats receive explicit retryable or unsupported states. |
| P0 | Credential discovery is not wired, yet existing source material can contain credentials. The CLI still supports a saved OpenRouter key, contrary to the requested environment-only key handling. | Credential scanner/isolation and environment-only key path; audit ordinary memory for prior leaks before asserting the invariant. Never print values during tests. |
| P1 | The rejected random-access window prototype rescanned frame headers for each page, cached full verification by forgeable file metadata, and decompressed before enforcing output size. Independent review: FLAG; prototype removed from the branch. | Design authenticated frame-offset projection or full verify per release; bounded decompression; adversarial replacement and tail-page performance tests before reimplementation. |
| P1 | `analyze` has a 180 s MCP timeout and a separate 180 s model HTTP timeout **after** fetch, route probes, and queue wait. | End-to-end deadline budget with separate asynchronous job status and idempotent result; model timeout must be less than the remaining job budget. |
| P1 | OpenRouter's known key has a provider-enforced $1/day cap while the user requested a potentially larger one-time historical allowance; policy/UI supports higher ceilings but no approved first-run amount is configured. | Keep current cap until explicit local change; use a finite dedicated key cap and cumulative usage reconciliation; present a first-run estimate before any historical remote work. |
| P1 | Hook entries exist, but event delivery, Claude retention, source-root relocation, and exact shared endpoint have not been proven end-to-end for each client. | One controlled real event per client and archive-version evidence, without synthetic history or duplicate servers. |
| P1 | The current code/repo and live process can diverge; a green unit test or pushed commit does not prove the process is running that code. | Record source commit/config fingerprint, deploy once under service-owner control, health and real workflow smoke, then verify rollback. |
| P1 | Model output is not evidence that work completed; strict analysis is ephemeral, while legacy insight persistence is provisional. | Source-linked claim IDs, evidence state, dedup/supersession, and a verification path before durable fact promotion. |
| P1 | Secure search is global (`query`, `limit`) without a project filter; a matching term can return a different project's history. Catalog dates are archive-capture dates, not necessarily conversation dates. | Bind project and source-event time as authenticated metadata, preserve both timestamps, test moved projects/duplicate exports and cross-project ranking. |
| P1 | Archive versions are keyed by absolute source path. A moved history home or copied export may create a second source identity. Capture checks size/mtime around a read, not a stable open-file identity, so an unusual same-size/same-mtime rewrite may escape the change check. | Stable content/source identity and handle-level pre/post checks; manifest records lineage. Test replace-during-capture and relocation without double-counting. |
| P1 | Index coverage, model results, ordinary memory counts, and backup generation are different denominators. An equal count in one pair does not prove source-to-claim alignment. | Per-stage versioned reconciliation ledger: discovered, captured, indexed, parsed, credential-scanned, interpreted, backed up, each with exclusions and hash/version linkage. |
| P2 | Search capabilities expire in 10 minutes; a long fetch/analysis queue may outlive its grant. | Resolve and pin an authenticated source reference when creating the job, or renew only after reauthorization. |
| P2 | API rate limits (10 fetch/min, 3 analyze/min) and single slots can reject valid batched agent work with 429 after 0.1 s. | Fair bounded queue with explicit retry-after, not a silent miss. |
| P2 | A repository-wide test run currently cannot collect an unrelated untracked MCP test without names from its untracked module; a later broad run hit a SQLite concurrency failure that was fixed and focused-tested but not reloaded into the live process. | Preserve the untracked files; verify the exact PR candidate with affected tests and an isolated broad run, reporting exclusions and failures rather than claiming full green. |

The reconciliation ledger should use one canonical source-version key (source
identity + archive blob/sha256), then join every downstream row to it. It must
show **why** a row is absent: not discovered, debounce/queue pending, capture
error, unsupported format, index pending, model deferred, credential scan
pending, or intentionally excluded. Neither `archive snapshots == indexed`
nor `ordinary memories == vectors` proves all live transcripts or credentials
are represented. User-facing history dates must not silently substitute
capture time for conversation/event time.

## Failure-scenario coverage matrix

Each row describes a **class of failures**, not a claim that every possible
combination has been exhaustively tested. A release must prove the relevant
failure injections on the exact candidate and retain the receipts.

| Class | Scenarios to exercise | Required visible state |
| --- | --- | --- |
| Capture/retention | app open while file grows; compaction; Stop spam; missing end event; machine sleep; app deletes source; relocated home; path symlink; export replacing itself | `queued/capturing/archived/indexed`, source version, retry reason, last successful event; no false ack |
| Data integrity | truncated blob; tampered later chunk/trailer/manifest; partial write; source changed mid-read; duplicate version; full disk; power loss; DB lock contention | fail closed, no partial published result, recoverable journal and exact version |
| Parsing/scale | 1 byte, empty, 15 GB, one giant line, multi-byte UTF-8 split, malformed JSONL, ZIP bomb, mixed binary, repeated terms, large tool output, duplicate exports | bounded memory/disk and explicit `processed/deferred/unsupported/corrupt`; no permanent `oversized` |
| Retrieval | common term yields many candidates; hit at tail; cross-chunk term; capability expiry; 429; timeout; concurrent clients; canceled request | stable cursor/job, bounded grant, no missing or duplicate spans, no secret-bearing fallback |
| Model/resource | no Ollama; model not installed; GPU busy; another model resident; telemetry stale; OOM; timeout; malformed/refusal reply; shutdown mid-call | deferred/retryable with cursor unchanged, no forced eviction, `keep_alive=0`, no fabricated memory |
| Remote/policy | ZDR unavailable; key missing/revoked; provider cap reached or reset; usage lag; network/proxy failure; consent revoked mid-job; daily/monthly override | no unapproved request, bounded spend, status explains denial without logging private data/key |
| Secrets/privacy | `.env` symlink; copied secret in chat; multi-line value; false positive; false negative; project rename; auth brute force; backup theft; model/tool prompt injection | encrypted value only, metadata search only, authenticated audited use, portable restore, no content in logs/PR |
| Recovery/operations | concurrent servers; wrong Python; antivirus blocks launcher; service restart; old runtime files; DPAPI account move; passphrase loss; stale backup | one owner, explicit error and rollback, no overwrite of unrelated work, recovery instructions/validated restore |
| Client/UI | Codex/Claude/Gemini schema drift; wrong endpoint; hook timeout; dashboard token leak; revoke while job runs; browser Origin/CORS mismatch | per-client delivery proof, accessible controls, no exposed bearer, authoritative policy state |

## Architecture choices to settle before another feature pass

1. **Historical processing:** choose CPU capture/index of all supported text,
   with model interpretation queued by relevance/newness, over a full historical
   LLM sweep. The latter has unbounded cost/latency for 15 GB and would occupy
   shared GPU for too long. Explicit bounded backfill remains possible.
2. **Workflow coordination:** choose one local SQLite-backed durable job journal
   with source-version and window cursors over in-memory tasks. The trade-off
   is more schema/recovery work, but it closes the hook-ack and timeout gaps.
3. **Large-source retrieval:** choose authenticated per-chunk offsets plus
   bounded content windows, with full-source integrity at the correct trust
   boundary. Never rely solely on file mtime/inode as proof of unchanged
   ciphertext. The exact offset-authentication design needs review before code.
4. **Inference:** choose a separate optional worker, not a permanently loaded
   model. CPU capture/search remains available when GPU or remote budget is
   unavailable. Local model selection uses measured quality and fresh VRAM,
   not model size alone. ZDR fallback is a policy-controlled branch.
5. **Secrets:** choose a separately encrypted portable vault with explicit
   local-use authorization. Normal transcript/agent retrieval may show
   bounded redacted context, never a credential value by default.

## Dependency-ordered acceptance gates

1. Keep the rejected large-window prototype out of the service; its independent
   P1 flags are design inputs for a replacement. Keep live data intact.
2. Close P0 token disclosure and capture durability first, then prove real
   hook events and restart recovery.
3. Close large-query/request timeout mismatch with job/poll and streaming
   format parsers; prove a tail hit in the real large source without a client
   timeout or full plaintext allocation.
4. Add relevance-driven model-window jobs with checkpoint/resume, fresh VRAM
   routing, model unload, ZDR consent/budget, and no model-produced false facts.
5. Add credential discovery/use, portable restore drill, and privacy audit.
6. Reconcile installed code, repo head, README, and live runtime; run focused
   integration and representative failure tests, then update the existing PR.
7. Finally design and implement the localhost UI around these authoritative
   states and the user's revocable choices.

Unresolved choices requiring user input only when their dependent action is
ready: the initial provider-enforced historical remote cap, permitted project
roots for a real credential scan, and whether any raw transcript value may be
shown outside explicit local authentication. Until then, safe CPU capture,
search, code review, and design work can proceed.
