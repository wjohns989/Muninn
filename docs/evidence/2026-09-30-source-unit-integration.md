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

## Follow-up: durable candidate ledger (2026-09-30)

Implemented an archive-attached encrypted event ledger with exact source-page
citations, bounded whole-unit screening, opaque project references, separate
source-observation/excerpt/model-interpretation labels, and append-only
`needs_user` decisions. Only complete verbatim user observations with known
project/event evidence auto-file, and remain `unverified_assertion`. Unsupported
typing, quoted/negated excerpts and paraphrases remain provisional. Potential
secret contexts do not enter public record text or eligible remote input.
No automatic inference or ordinary-index publication is added by this component.

The event chain and encrypted head commit atomically; every publication verifies
its existing prefix. Deletion/ciphertext/reference/head changes fail closed, as
do unsealed/forged citations and interrupted head publication. Whole-database
rollback cannot be detected without an external freshness anchor. That is an
explicit limitation, not a recovery guarantee. Archive online backups/restores
include and verify the new ledger; isolated portable recovery passes.

Source-unit lookup uses indexed binary seeks plus only the selected unit's
contiguous fragment range. The regression proves logarithmic seek decrypts plus
the selected range, with one completion-count authentication. Whole-unit
screening keeps digest state and a bounded 128-entry cache. A 300,000-character
benign unit is accepted for provisional excerpt use without a total-size cap;
a credential outside the selected chunk denies release/remote eligibility.
No measured whole-machine peak-memory claim is made.

Focused affected suite: 61 passed in 28.76 seconds; after distinguishing actual
user excerpts from model interpretations, the ledger's 25 cases passed in
12.25 seconds. Intact source/archive/context results were reused. Independent
review cleared the changed prefix-integrity and whole-assertion gates, then
cleared the actual bounded receipt and smoke script. Earlier full Linux CI passed all
five checks for the context-lock fix; this new slice needs its own CI result.

Bounded real-source proof: one actual Claude Code 13,210-byte snapshot produced
one encrypted whole-user observation. A fresh ledger object authenticated its
event and source citation after reopening; project and provider event time were
present. State was `filed`, truth status `unverified_assertion`. No provider was
called; no transcript, opaque ID or credential value was printed. The first
small-source selection had insufficient eligible text; selection was changed
to completed Claude source units and bounded excerpts rather than weakening
the secret gate. This is local persistence proof, not model classification or
automatic agent-search proof.

## Remaining acceptance gaps

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
# Interactive review C: truthful completion and resource reporting

The user ran the visible interactive PowerShell helper with one six-row page
and a two-call limit. Both encrypted backups validated at 10,424 records. Two
local model calls completed; no queue rows resolved (24,442 pending, 155
deferred, 30 rejected). Exit code 2 correctly reports unresolved review, not an
unlock or backup failure. A subsequent read-only check found no review worker,
one healthy existing Muninn listener, and an idle GPU. No service was restarted.

The old page report could retain a transient `gpu_busy` reason after later
successful calls, without identifying subsequent per-page quota deferrals. A
red-first four-context regression demonstrated that exact misleading result.
Reporting now distinguishes quota-deferred and route-deferred contexts and
updates the route after successful admission; it does not relax GPU checks,
invent resolutions, or change cached decision/application behavior. The old
live report cannot establish how many contexts each deferral affected.

Focused validation: 43 ambiguity-triage and encrypted-memory-ledger tests
passed. The synthetic home-path privacy fixtures use the standard `user`
placeholder so CI's portable-path gate stays enforced. The preceding commit's
full locked-dependency suite passed; its only CI failure was those synthetic
paths, not a discovered real secret.

The reporting fix was pushed as `8fc88fc`; all five GitHub checks passed,
including the full locked-dependency suite and the privacy gate.

## Bounded atomic ledger batch component

Ten new cases failed first because `record_batch` was absent. The implemented
1–64 proposal API authenticates every source/quote before publication,
validates the old encrypted event chain once per batch, preserves distinct
occurrences, and skips exact retry/within-batch duplicates. Events and one
committed head update are atomic. Late invalid quote/page, corrupt existing
prefix, and head-update interruption leave prior database rows unchanged.
All 35 ledger tests passed, including portable recovery and the prior privacy
and citation tests. Independent examination of the actual diff was CLEAR;
the reviewer did not rerun tests or access live data/providers. Automatic
model-job integration remained FLAG for missing source citations, whole-input
remote credential screening, explicit model-proposal provisional status and
unreserved spend admission. These are recorded in the routing plan, not
treated as resolved by batch tests. No live activation or restart occurred.

## Model proposal origin boundary

A red-first whole-user-message test demonstrated that the original batch API
could alias an already-filed direct source observation. Batch proposals now
have fixed model origin, remain provisional, and use distinct HMAC refs. The
trusted single-source rule API retains its existing identity derivation. Neither
public component API accepts a caller-supplied origin label; independent review
FLAGged that proposed override and was CLEAR after its removal. Old encrypted
records without an origin remain explicitly `legacy_unrecorded`; retries do not
retroactively invent one. Portable restore retains both old and model-origin
labels and states. Validation: 42 ledger tests passed, followed by the one new
legacy/model portable-restore test (43 total intact cases). Reviewer examined
the actual diff without provider, live vault or process access. Automatic source
citation and complete remote-input screening remain unresolved integration
dependencies; this change did not activate inference or restart Muninn.
