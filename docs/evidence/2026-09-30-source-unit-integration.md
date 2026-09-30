# Source-unit integration: bounded evidence, not goal completion

## Verified locally on 2026-09-30

- User-approved existing service start: one expected Miniconda process and
  listener on 42069; health/authenticated checks 200, anonymous protected route
  401, anonymous dashboard did not contain the token. Strict encrypted history
  ready; 5,235 snapshots / 4,110 sources at the observation. No PID recorded here.
- Real bounded encrypted source samples: Codex 567 bytes / 1 unit; Claude Code
  13,210 bytes / 17 units; Gemini CLI 408 bytes / 1 unit. Independent JSON record
  counts matched streaming units, all sidecars verified; no inference sent.
  The Codex sample was metadata-only. These samples contained no ambiguous
  assignments; matching credential review remains unproven on real data.
- Additional real pending Claude Code snapshot (1,655,851 bytes): 32 raw
  ambiguity occurrences joined to 32 authenticated physical source units,
  all with event-time and project evidence; 11 matched pending name/reason
  metadata. Vault values were not unlocked, so exact encrypted-candidate
  agreement and classifier correctness are not claimed.
- Real archived excerpt through the live authenticated ZDR OpenRouter route:
  successful `openai/gpt-6-luna-pro` response with the four required fields.
  No transcript, generated text, credential or capability printed.
- Checked-out secure-analysis code against the real archive: Qwen 2.5 7B
  successful requested-model interpretation in 7.59 seconds; new Defiant Q8
  successful requested-model interpretation in 44.08 seconds. Native schema
  now includes the parser's existing 12-entry array bounds and prompt priorities.
  These two results are direct local component proof, not proof the already
  running HTTP process has loaded later source changes. No inference output saved.
- After both local calls, Ollama reported no loaded models; GPU utilization
  was 0% with 13,817 MiB free. This verifies post-call release for these calls,
  not every interruption/driver-failure scenario.
- Independent review found and prompted fixes for repeated metadata keys,
  UNKNOWN leaving pending, and pending rows starving later pages. Regression
  tests cover these, model digest changes, portable sidecar restoration,
  source reorder/tamper, late failures, large single records and JSON escaping.
- Final focused affected suite: 219 passed in 58.18 seconds across credential
  discovery/review/context, archive/sidecar recovery, streaming projection,
  source units, blind search and secure analysis. Independent final inspection
  cleared the reviewed pagination/schema changes. Full merge suite remains a
  separate gate; no merge claimed.
- The first Linux full CI run had 2,280 passed / 13 skipped and three new
  review-setup failures because path-only reopening required a passphrase
  without Windows DPAPI. ReviewSource now accepts an already authenticated
  archive handle and the tests use their passphrase-created handle. The
  corrected full CI run must be observed before claiming the merge gate passed.

## Remaining acceptance gaps

## Follow-up: credential context reader lock (2026-09-30)

The first monitored six-row/two-call live sample validated its pre-review
backup (10,424 credential records), then failed with `OperationalError` after
one local model call. The post-review backup was not produced. An isolated
regression reproduced `database is locked`: a DELETE-journal reader remained
open across the yield while the same database tried to commit the review cache.
No completed queue decision is claimed for that failed pass.

The reader now releases connections before external yields, reads at most
32 bounded encrypted pages at a time, and compares the consumed ciphertext
sequence with a final pinned scan before iteration completes. This does not
change the encrypted format or require a vault migration. Late mutation of an
already-yielded page, deletion and invalid length fail closed. Cache writes
can commit while the reader is suspended. A real encrypted-store integration
test proves a partial review resumes from its cache and a late integrity
failure prevents applying a queue decision (synthetic input/model only).

Focused verification: 36 tests passed in 12.89 seconds plus two added
integration cases passed in 2.86 seconds. Independent actual-diff review
cleared this lock/integrity fix for the bounded retry, not broader activation.
A fresh local interactive retry was opened with separate before/after
destinations; its completion and classification quality remain to be observed.

Read-only runtime recheck: strict archive ready, 5,248 snapshots / 4,112 sources;
health/authenticated routes 200, anonymous protected route 401, no dashboard
token disclosure, one expected interpreter/listener. Ollama had no loaded
models and GPU utilization was 0%. The earlier corrected Linux CI and the
capture-error diagnostic commit both passed all five GitHub checks. These
claims do not prove this follow-up has been loaded by the running HTTP process.

General durable claim extraction, project/type filing, temporal conflict review,
historical backfill and UI integration are still implementation work. This slice
does not promote provisional text into ordinary indexes. Credential triage needs
a local interactive unlock and representative context proof before a large pass.
The observed capture queue had two unavailable items requiring diagnosis.
Read-only journal diagnosis identifies both as missing Claude Code sources,
not model, archive-unlock, or GPU failures. They remain explicitly unavailable;
this check does not establish whether their last bytes were captured elsewhere.
New application code needs a separately authorized single-service restart before
its live HTTP behavior can be claimed. Other services have not been stopped.

Runtime and encrypted data stay local. Public source, tests and portable scripts
belong in the existing repository branch/PR; unrelated dirty files stay untouched.
