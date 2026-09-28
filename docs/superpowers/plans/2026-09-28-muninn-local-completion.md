# Muninn local completion plan — 2026-09-28

Outcome: the running Windows installation durably captures real agent history,
searches/fetches bounded evidence from all supported source sizes, interprets
pertinent evidence automatically without idle GPU residency, protects credentials,
and proves each configured client. Portable behavior is pushed to PR #142.

The architecture contract, observed starting state, failure matrix, and timing
budgets are in `docs/architecture/local-operating-model-audit.md`. The audit's
timings are targets, not claims. Preserve live data, untracked files, and the
current service until a reviewed replacement and rollback are ready.

## Critical path and proof

1. **Dashboard authentication:** remove bearer injection from anonymous HTML;
   accept a user-entered token only into volatile browser state, validate it
   before opening the UI. Prove anonymous GET contains no bearer and wrong
   tokens stay locked out.
2. **Capture durability:** commit a validated, bounded job record before hook
   acknowledgement; recover pending jobs on restart; reconcile changed sources;
   archive/index idempotently. Prove crash-after-ack recovery, file growth,
   debounce expiry, and unavailable archive behavior on temporary stores.
   Design for review before code: keep one owner-only SQLite journal beside the
   encrypted archive, with DELETE journal mode and synchronous FULL. Store a
   keyed source identifier and AES-GCM-sealed canonical path, never plaintext
   path/content. Enqueue validates the provider/path and stats the source, then
   commits before returning. Repeated events increment a revision and coalesce
   by source; the worker may mark a job complete only when its claimed revision
   still matches. A crash requeues `capturing` rows. Capture uses the existing
   archive writer lock and archive_file idempotency; completion wakes CPU index.
   A bounded startup/periodic discovery pass queues changed provider sources,
   so missed hooks and growing files recover without repeatedly hashing all
   15 GB. Deleted-before-copy sources remain explicit retry failures (a queued
   acknowledgement is not a claim that bytes were archived). No unreviewed
   queue schema or live migration is installed.

   **Durable capture protocol (review revision 2).** The journal has a `jobs`
   table keyed by an archive-keyed HMAC of `(provider, stable_source_id)`. For
   Codex/Claude session files, `stable_source_id` is the UUID from the validated
   filename; for other formats it is a keyed canonical locator until a
   provider-native identifier can be safely read. The row contains only an
   AES-GCM-sealed canonical locator and bounded metadata: provider, kind,
   monotonically increasing `event_revision`, observed size/mtime (or missing),
   state, due time, attempts, last error **code**, and last committed archive
   content hash/generation. The encrypted archive manifest still uses paths
   internally; the journal's stable identity deduplicates repeated events at
   one validated locator. Relocation is **outside this first capture-journal
   slice**: a moved source can still create a second archive lineage. A later
   authenticated manifest-lineage migration and move test are required before
   the overall local installation can claim location independence.

   The hook validates provider/root/name without trusting a symlink or
   non-existent final file, seals the locator, and commits an UPSERT in SQLite
   `journal_mode=DELETE`, `synchronous=FULL`, with a short busy timeout. Only
   after commit may the endpoint return HTTP 200 (its body remains `{}` for
   hook compatibility). Invalid paths get 400; unavailable journal/failed
   commit gets 503. No source bytes are copied in the hook. The 0.8-second
   client deadline is a measured service target, not a promise: timeout or 5xx
   is printed as a bounded failure without blocking the host, and the
   independent discovery pass later retries the source. A test must pause
   enqueue before commit and prove the endpoint cannot acknowledge it, and
   measure the real local fast-path latency over repeated events.

   One worker transaction changes due `pending/retry` to `capturing` and
   records the claimed `event_revision`; startup requeues every `capturing`
   row. The worker revalidates the decrypted path and source identity, uses
   `archive_file()` under its existing writer lock, then updates the row to
   `archived` only if the claimed revision is still current and observed
   fingerprint still matches the post-capture file. If a newer event arrived,
   it stays pending. The archive manifest commit precedes the journal done
   commit. If the process dies between them, replay calls idempotent
   `archive_file()` and records `unchanged` as success after verifying the
   published version; an unreferenced staged blob is never treated as done.
   Missing/changed-in-flight/file-permission/full-disk/locked/archive-error
   become typed retry states with capped backoff, not dropped jobs.

   Startup starts the worker and a separate CPU-only scanner. The scanner
   iterates validated provider roots in batches of at most 250 paths using a
   bounded directory iterator, persists `(scan generation, per-source seen
   marker, counts, error codes)` after each batch, and restarts a partially
   completed traversal from the root after a crash while skipping rows already
   marked seen. It never globally sorts or retains all paths in memory. It compares size/mtime
   against one authenticated in-memory manifest-signature snapshot plus job
   state; no startup full-file hash of the 15 GB corpus. A later bounded
   rolling verification pass handles same-size/same-mtime rewrites. Each
   completed scan restarts from the beginning on the configured cadence so
   files added before a cursor are found. Excluded formats, missing files,
   and scan errors are counted independently. Journal backup uses SQLite's
   online backup while the archive backup writer lock is held; replaying a
   restored pending row is safe even if its snapshot already exists. No
   plaintext paths, source contents, sealed payloads, or exception strings
   may appear in logs, status, the ordinary persistent memory store, or agent
   tool results. A decrypted locator necessarily exists transiently in worker
   RAM during validation/copy; strict capture must not retain it in a cache.
3. **Asynchronous large work:** make search/fetch/interpretation resumable jobs
   with deadlines and bounded authenticated windows. Stream supported exports
   without whole-file reads. Prove tail hit, competing candidates, malformed
   input, interruption, cancellation, and no size-only exclusion.
4. **Automatic model decisions:** queue only new/pertinent windows; choose from
   installed Ollama models using fresh resource telemetry, unload after calls,
   defer under VRAM pressure, and use ZDR OpenRouter only under persistent
   consent and enforceable spend. Prove real local and remote routes, idle GPU,
   refusal/retry, and accurate route status without model-written facts promoted
   as verified evidence.
5. **Credential isolation:** stream project/transcript scans into portable
   passphrase vault; expose metadata search and authenticated audited local use,
   never normal-search values. Prove controlled marker end-to-end, restoration,
   no plaintext leak, and environment-only provider key configuration.
6. **Client and operations:** verify real Codex, Claude Code, Gemini and MCP
   delivery against the one local service; check retention, source paths,
   scheduler/backup cadence, authenticated restore and rollback. Update README
   for configurable roots/models and actual setup, not machine-specific paths.
7. **Candidate integration:** focused red-first tests, independent spec/quality
   review of consequential slices, isolated broad tests, then one controlled
   service update. Verify code/config identity, live archive/search/model flows,
   backup generation, resource baseline, and remote PR head. UI overhaul plan
   and controls come last, except for the immediate bearer leak.

Each slice must identify its changed dependency, smallest meaningful check,
retained evidence, and unresolved limitation. A passing component test is not a
claim that the running listener has loaded that code.
