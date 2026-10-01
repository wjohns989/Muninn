# Source-unit integration: bounded evidence, not goal completion

### 2026-10-01 CI contention counterexample

The full locked-dependency run for `01d6f1d` passed 2,604 tests but failed the
direct-service search polling test: a concurrent SQLite writer exceeded the
existing 100 ms read wait. The HTTP boundary already returned sanitized no-store
503; the direct test bypassed that boundary and treated transient SQLITE_BUSY
as fatal. No model failure or lost data was demonstrated.

The correction retries only primary SQLITE_BUSY in that direct test under the
unchanged five-second deadline. Both authenticated poll endpoints now advertise
`Retry-After: 1` only for SQLITE_BUSY, preserving all other errors and durable
states. Real isolated SQLite EXCLUSIVE-lock checks prove busy 503, authentication,
no-store and nonsecret output, followed by the same pending job after release;
nonbusy SQL errors do not advertise retry. Four affected suites passed 18 tests
in 12.38 seconds (including a temporary, subsequently removed within-budget
probe that also passed the preimage). The prior failing full CI run is retained
as the original counterexample, not relabeled green. Replacement full CI and
live activation remain separate checks.

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

## Cited model-input component

`CitedAnalysisSource` binds an authenticated immutable snapshot/version,
source attempt/page, selected offsets/length, parser version and domain-separated
canonical input digest. Preparation drains the complete source-page iterator;
reopen rejects changed input, unavailable/unsealed pages and unsupported
descriptors. Source location/native IDs are not included in model-facing data.
Source role, event-time basis and opaque project evidence are retained.

Independent review caught a split-query boundary bug. The fixed descriptor
includes both authenticated adjacent pages from the same unit when necessary,
within the 3,000-character total window. Explicit citation ranges retain the
query hit and prohibit invented quotes spanning those ranges. A future model
prompt must honor these ranges. All model proposals are validated before an
atomic model-only ledger batch; late invalid quotes publish nothing. Whole-unit
credential screening denies remote eligibility without altering authorized
local input. This is an input component, not complete-request remote admission.

Validation: 20 focused cited-input tests passed; combined cited-input, ledger
and archive checks passed 76 tests. Actual-diff independent review was CLEAR
after the boundary fix. A bounded real local archive preview reopened an
89-character Claude Code window from a 13,210-byte source, with query retained,
event time/project evidence present and remote eligibility true. It dispatched
no model, added zero candidates and changed no historical source. No raw text,
source path, IDs or credentials were printed. The preceding `85b96be` head had
all five CI checks passing. Automatic worker/staged replay and complete remote
request screening are subsequent integration dependencies, not verified by
this preview.
# Durable cited-result staging (component, not live activation)

The encrypted capture journal now binds an immutable cited input, stages a
validated model reply, and separates publication admission from ledger writes.
Cancellation is durable before admission; after admission recovery replays the
staged result without another inference call. Acknowledgment verifies the exact
stage-derived IDs actually exist in the authenticated ledger, outside the journal
writer transaction, then rechecks the lease before marking success.

Focused isolated evidence: 33 staging/journal/service cases passed across the
affected test runs, plus 10 capture/search/API cases. Tests cover remote/local
crashes before and after ledger commit, portable restore, tampering, cancellation,
fabricated or unpublished receipt IDs, corruption, lease expiry during proof,
and independent writes during verification. The independent actual-diff/result
review was CLEAR for this component. These are not new live inference tests.

Subsequent worker integration now connects the cited schema, immutable input
binding, whole-unit/full-envelope remote admission, encrypted staging, and
publication-only replay. The private model reply is never an ordinary tool/API
result. Actual installed Ollama weights are checked before and after a call;
remote identity binds the returned model identifier, not invented weight data.

Focused evidence: 21 transport/service cases passed in 13.73s. They cover real
isolated encrypted stores with mocked transport, staged replay before/after
ledger commit, changed weights, invalid citations and schema type overrides,
nested request credentials, mutation during remote marking, source credentials
outside the window (both production routes), and resource deferral retention.
Five resource cases failed red-first before the route/journal reason mismatch
was corrected. The old safe-input ZDR fallback regression passed separately.
One earlier red-first service test unexpectedly dispatched the old local route
on nonsecret test text; a forbidden-legacy-route guard now prevents recurrence.
That call is not claimed as representative live validation.

The independent diff review found the credential-only type-schema mismatch;
the explicit output check and malicious-response regression fixed it. No
service restart or historical model backfill was performed for these slices.

### Representative local cited inference

The bounded real-source preview initially found no eligible window under its
arbitrary 150-character minimum; no inference occurred. Using the existing safe
89-character user window from a 13,210-byte Claude snapshot, Qwen returned valid
JSON/schema but failed the citation check. A secret-free failure-category field
distinguished citation failure from malformed JSON without printing the reply.
The parser now derives a coordinate only if an unchanged exact quote occurs
uniquely in the authenticated input. It does not repair paraphrases, non-integer
offsets, ambiguous occurrences, or source-range crossings; valid repeated-quote
coordinates remain unchanged. The extraction identity versions this behavior.

Four coordinate cases passed, including old-defect sensitivity. After this
reviewed correction, the real `qwen2.5:7b` preview completed in 6.19s with one
validated proposal, no ledger publication (`candidate_delta=0`), and remote
disabled. Ollama `/api/ps` then returned zero resident models. This proves the
new parser/transport on a short actual chat source, not historical accuracy or
automatic live filing. The final affected transport/service/legacy-analysis run
passed 45 cases in 13.73s; real provider replies remain out of logs and this repo.

### Representative ZDR cited inference

The ZDR preview requires the installation's explicit existing policy root,
not the configurable archive's parent, and fails closed without enabled managed
consent. A separate regression proves revocation/re-enable during source loading
cannot confer new consent on an old request. The initial actual preview sent no
request: full-envelope screening treated the bundled 29-character fallback model
ID as an opaque credential. A red-first regression reproduced this. Screening
now recognizes only the three exact bundled public IDs in structural model
fields; source/prompt/extra fields and unknown IDs are unchanged and screened.
Four affected envelope cases passed, including final pre-POST mutation denial.

After that reviewed correction, one actual preview under the existing persisted
consent/budget and provider ZDR constraints returned `openai/gpt-6-luna-pro` in
5.66s with valid analysis and zero proposed memory claims. Publication remained
off (`candidate_delta=0`). No policy/cap changed and no private response text
was printed. The short source supports parser/route proof, not extraction
coverage or accuracy on the historical backlog. A spend-availability query is
still not a reservation; automatic paid historical backfill remains gated on
that separate dependency and ledger growth validation.

### Approved activation and actual queued publication

The user approved restarting only the existing Muninn installation. Before
termination, five SQLite online preimages passed structural integrity checks;
the final journal writer fence confirmed no in-flight or claimable capture,
search, or analysis jobs. The Windows stop was forced termination, not graceful
application shutdown. A log reservation failed after stop because the runtime
parent was not owner-only. No data was removed. The existing local launcher
then started the same checkout, Miniconda interpreter, runtime directory, and
loopback port 42069 without bypassing antivirus or execution policy.

Post-start checks: exactly one Muninn process/listener, health 200, anonymous
protected route 401, authenticated route 200, strict encrypted archive ready,
and no auth token in the anonymous dashboard. At the final observation the
archive held 5,322 snapshots / 4,123 sources; capture was 3,259 archived and two
explicitly unavailable missing Claude Code files, with no pending capture jobs.

One authenticated queued search used an actual indexed archived chat, not a
synthetic fixture. Search succeeded with one match and automatically linked an
analysis job. The live worker selected `qwen2.5:7b`, reached `succeeded`, and
published five encrypted provisional candidates with five acknowledged memory
refs. The cited window, validated extraction, and publication receipt were
persisted; publication admission was set. Fresh ledger verification authenticated
all five refs and the event chain. Queue acknowledgment to publication proof was
16.58 seconds. Source-selection preparation before queuing was much slower;
16.58 seconds is not an end-to-end search latency or optimization claim.

The diagnostic observer was stopped once and resumed using its private saved
job handles; that did not cancel the durable job or repeat search/inference.
After publication Ollama `/api/ps` returned `{"models":[]}`. No source text,
credential, bearer, capability, or process ID is recorded here.

All five GitHub checks passed for the activated `ebadb0c` code, including the
full locked suite (2m53s). This closes the bounded live local publication gate,
not historical classification accuracy, agent lookup of new ledger refs, paid
backfill budget reservation, or UI completion. Those remain explicit acceptance
gaps rather than grounds for widening this activation test.

### Cited-memory agent reads and source follow-up

Added dedicated authenticated, loopback-only API/MCP operations for safe lexical
candidate search, exact ref lookup, and source follow-up. They do not infer,
publish, promote provisional claims, reveal credentials, or build plaintext
indexes. Search authenticates the chain once and keeps at most 20 results.
Private queries are rejected without echoing input. Reads freshly screen the
complete cited unit; a stale screen from a prior read cannot hide later fragment
tampering. Unsafe units have metadata and access to the existing redacted
projection, not raw context. Source grants carry a fixed public term, never a
quote or path, and expire under the existing capability policy.

The shared single CPU-reader slot survives client cancellation until the actual
thread finishes. The 60/minute admission check runs again after slot acquisition
to avoid concurrent last-allowance overshoot. Invalid request fields, private
MCP failures, and task failures do not echo private values or exception traces.

Evidence: the initial three regression cases failed for missing search/source
operations and stale public screening. The affected ledger/API suite passed
95 cases in 40.48 seconds. Five subsequent validation/error cases passed in
6.19 seconds, and the concurrent last-allowance case passed separately. These
focused results overlap and are not summed as distinct tests. An initial test
collection import error was corrected before any tests ran.

Actual-data read proof reopened all five candidates from the preceding live
model job, preserving model-origin/provisional labels. The new local search
found a published candidate and returned 1,362 characters of safe source context.
Its grant followed the source through the existing live authenticated transcript
API: one redacted page, 1,443 characters, with no ledger change or model call.
This read proof took 1.58 seconds. New API/MCP definitions have not yet been
loaded by the running service; activation and a live MCP round trip remain a
separate gate. No PID, query, source text, grant, ref, or credential is saved here.

### Live agent-access activation and configured bridge

After a separate user-approved reload, the actual live MCP transport passed
`get_cited_memory`, `search_cited_memories`, `get_cited_memory_source`, and the
related redacted transcript-page read on a previously published real memory.
Provisional/model-origin labels survived, anonymous and wrong-token reads were
401, and Ollama reported zero resident models. The reload preserved one existing
interpreter/listener, strict ready history, and five structurally verified SQLite
preimages. Its first readiness assertion raced startup initialization; the same
new process became ready and was independently verified, without another reload.

Hook status confirms all four installed Codex and Claude Code hooks and Gemini's
SessionStart/SessionEnd/AfterAgent/PreCompress hooks. Actual acceptance receipts
exist for all four Codex and Claude events and Gemini SessionStart/SessionEnd.
Gemini AfterAgent/PreCompress are installed but lack actual receipt evidence.
The inspected existing client bindings select the same Python and stdio bridge
with the core profile; no client configuration was changed or disabled server
enabled. The bridge forces authenticated loopback and disables autostart itself.

CI caught two concrete gaps: a synthetic home-path canary did not use the accepted
portable `user` placeholder, and the expanded core profile exceeded its 20-tool
limit. Both were corrected without weakening the guards. Three mutation tools
remain available in the full profile instead of core. The existing compatibility
case passes. A real subprocess launched through the configured stdio bridge,
outside the repo working directory, listed exactly 20 core tools including the
three new reads and retrieved an actual saved provisional memory. No model call
or settings edit occurred. The running HTTP process retains its earlier core
list until a future batch reload; configured stdio clients load the corrected
20-tool definition now. A pre-activation smoke parser incorrectly assumed an
API envelope in the MCP text result; correcting the probe to the existing data-
only MCP contract required no program change or inference retry.

At `1d759ff`, privacy, clean imports, replay, and benchmark CI checks have passed.
The full locked suite remains in progress under run 36806259226; no full-suite
pass is claimed. The preceding full run had 2,431 passed / 13 skipped and only
the now-corrected core-count failure. Automatic historical enrichment, ambiguity
resolution, paid-backfill reservation/scaling, and remaining UI work are still
open; this evidence does not mark the installation goal complete.

### Follow-up: journal connection contention

Run 36806259226 later hit its 20-minute limit. The subsequent run 36807052477
completed with 2,431 passed / 13 skipped and one failure: a concurrent status
poll in `test_search_automatically_queues_and_completes_one_analysis` failed at
the routine `PRAGMA journal_mode=DELETE` assignment with `database is locked`.
This identifies the failed operation, not the cause of every previous stall.

Journal initialization now establishes DELETE mode before schema setup. Runtime
connections only read/validate that mode, rejecting drift rather than attempting
a database-wide change. FULL synchronization and the 100-ms lock timeout remain
unchanged. SQLite documents the distinction between the querying and assigning
forms: <https://www.sqlite.org/pragma.html#pragma_journal_mode>.

Two red-first checks demonstrated the old routine assignment and silent runtime
conversion of WAL. The revised checks cover a real committed search-job status
read while another connection holds a write transaction, and fail-closed mode
drift. Capture/analysis suites passed 23 checks in 8.93 seconds; the strengthened
writer regression plus affected search journal/service/API/cancellation checks
passed 14, with one platform skip, in 8.47 seconds. All fixtures use isolated
temporary archives; no credentials, models or live service were involved.

The connection fix is source-only until a separately approved reload. Existing
live cited-memory and configured-bridge proofs remain unchanged. A new exact-
candidate Linux CI result is required before calling this regression resolved
across platforms; no broader scheduling or paid backfill has been enabled.

**Result:** exact program/test candidate `e2236f3` passed all five PR checks.
Linux run 36808400643 completed with **2,434 passed / 13 skipped** in 163.47
seconds, including the previously failing automatic-analysis case. Independent
examination of the actual connection diff and held-writer/mode-drift proofs was
CLEAR. This closes the demonstrated connection-setup regression across the
tested Windows and Linux environments, not every possible contention or prior
stall. Source activation remains pending the next scoped service reload; no
new service operation or model dispatch occurred in this follow-up.

### Query-independent encrypted window plans

Program candidate `6d7bf01` adds `CitedWindowPlanStore`: descriptor-only encrypted
pages partition supported conversational bodies into fragment-bound windows of
at most 3,000 characters, without a fabricated search query or a total-message
cutoff. Generated role labels and omitted/nonconversational records are not
inference work; raw originals are retained separately. A plan is publishable
only after authenticated archive and source-evidence EOF. It does not claim
that any model ran or that ambiguity/classification is settled.

Lookup and AEAD seals bind snapshot/version, evidence attempt, parser/plan
algorithm and width. Independent review caught both evidence-regeneration and
geometry-cache invalidation gaps; both were corrected. Supported old widths
remain verifiable, while unknown algorithms require explicit migration. Copy,
backup and portable restore now include and verify the derived encrypted store.

Eleven isolated plan checks passed in 9.30 seconds, including full long-message
body reconstruction, cached reuse without raw rescan, omitted units, cancellation
after a staging commit and retry, late source/plan corruption, changed evidence
attempt, changed geometry and portable recovery. The related source/citation/
ledger suites passed 91 checks in 51.10 seconds. Initial red checks demonstrated
the missing component. Early integration failures exposed formatting-label and
empty-cache handling mistakes and were corrected; focused passes are not a
full-installation completion claim.

After independent examination cleared the corrected diff, the bounded real-
source proof used the source of an already published memory, not a fixture:
two encrypted windows, 1,422 conversational characters, nine source units;
plan build/reopen/verification took 0.12 seconds and cached reuse passed. No
source text, credential or bearer was printed. The memory ledger was unchanged;
no inference, service restart, settings edit or automatic scheduler activation
occurred. This is short-source operational proof, not large-backlog performance
or model-processing coverage. Capture outbox, window-processing checkpoints,
search priority and cadence remain the next integration dependency.

The exact-candidate Linux run 36810376224 passed **2,445 tests / 13 skips** in
155.51 seconds. All five PR checks passed. The running service remains healthy
(HTTP 200); it has not been reloaded to use an automatic capture lane.

Independent actual-proof examination retained the real source/plan/ledger
result but identified one unmeasured telemetry field: the probe's literal
`automatic_scheduler_active=false` was not a live scheduler-status check. That
field was removed; only the probe's own absence of inference/restart paths is
claimed. The scheduler is not implemented by this component or this proof.

The same examination flagged the next architecture's archive/outbox crash gap:
if source content changes between archive commit and retry, processing only the
latest version can omit an earlier committed snapshot. The next integration
must reconcile every post-enable commit absent from its durable outbox, exclude
pre-enable historical snapshots, and obtain exact commit identities under the
archive write lock. This is a required next design correction, not an additional
pass claimed for the proposed scheduler. This documentation-only receipt can
accompany the next code batch rather than force another identical full CI suite
solely for a status update.

### Durable capture outbox (source-only; not model coverage)

The existing cited-tool activation was rechecked instead of redundantly restarting
the shared service. One authenticated installation had idle durable queues.
Actual MCP search, lookup, source-follow and a 1,443-character redacted transcript
page passed; anonymous/wrong-token HTTP returned 401 and Ollama had zero resident
models. No inference, service operation or client settings edit was performed.

New source code adds an opt-in `MUNINN_CAPTURE_ENRICHMENT` outbox, default off.
Archive commits return optional internal path-free exact-version receipts under
the writer lock. The service strips these from ordinary capture results, seals
an immutable starting watermark under the archive lock before capture/start/scan,
and enqueues receipts after commit. A transient journal lock does not discard a
successful raw capture; subsequent reconciliation recovers every eligible version,
including one followed by a newer source version before retry. Existing archive
return contracts and CPU-only capture remain intact when the flag is off. Empty
schema tables are additive even while the flag is off, not a schema-inert claim.

Independent design review caught a concrete efficiency defect in the first
reconciler: bounded inserts still repeatedly enumerated the completed prefix and
loaded every known outbox ID. The corrected reconciler persists an encrypted
checkpoint tied to an immutable authenticated manifest generation/source/version
position, examines at most 128 entries (including ineligible entries), and commits
receipt inserts plus checkpoint advance in one transaction with a checkpoint
comparison against concurrent advancement. New earlier-sort paths arriving during
a pinned pass are found on the next generation pass. A zero-insert batch is not
EOF. Manifest metadata loading/key-list construction remains O(catalog size);
large-backlog scaling and recovery latency remain pending before activation.

Four service admission checks failed before wiring. The integrated outbox,
strict-service, archive and capture-journal suites passed **65 checks** in 41.41
seconds. Failure/recovery cases include queue lock plus source growth, watermark
lock ownership, unchanged/new and legacy receipts, batch commit epochs, bounded
traversal, resumed checkpoint, concurrent checkpoint race, atomic rollback,
encrypted-data/cursor/pinned-manifest tamper rejection and portable restore.
An added partial-checkpoint portable recovery check plus affected window-plan
and secure API suites passed **51 checks** in 16.98 seconds. Each set reported one
unrelated installed-library deprecation warning. No synthetic model response is
represented as a live model or installation proof.

This stage does not enqueue typed inference windows or acknowledge processing
coverage. Worker cadence, search priority, classification/ambiguity review and
paid-backfill cost admission are still open. The live service was not reloaded
to activate this source-only stage, and the installation goal remains active.
Independent examination of the actual corrected source diff and focused results
was CLEAR for this default-off outbox stage only; it did not clear scheduler or
backfill activation.

**Post-push result (2026-10-01):** candidate `428474f` passed all five PR checks:
locked full suite, clean imports, privacy, replay and benchmark dry run. Exact
full-suite run: `36813299319`, job `110212844556`. This is source verification,
not live activation or completion of the installation goal. This status-only
receipt can accompany the next code batch instead of triggering identical CI.

### Typed capture-window jobs and acknowledged coverage (2026-10-01)

The trusted outbox path now derives query-independent capture-window targets;
ordinary search completion cannot manufacture this lane. Authenticated targets
bind source/version, plan attempt, ordinal and canonical descriptor hash. Lane,
target, outbox mapping and local-only policy are checked on claim and through
window bind/stage/publication. An altered routing lane or a different window of
the same snapshot cannot be admitted. Capture jobs cannot set the remote-dispatch
marker or use the legacy result-only completion path. The worker reopens the
exact plan window without search terms and forces local-only routing without
consulting general remote policy. Existing background consumers do not include
the capture lane unless explicitly requested internally; no new timer or live
activation occurred.

Source plan/cursor state is encrypted. Jobs, authenticated ordinal mappings and
cursor advance commit together, compare the expected sealed plan on races, and
consume no ordinal on saturation. Automatic jobs occupy at most 24 of 32 active
slots; foreground search analysis has claim priority. Pending/due/running source
searches suppress automatic claims. The indexed source planner rotates fairly
across large sources instead of always restarting at the first receipt.
Coverage advances atomically only after the existing stage/actual-ledger receipt
validation and publication ACK. Valid zero-proposal analysis still acknowledges
its analyzed window. A zero-window plan is `no_context`, not completed analysis.
Failure, cancellation and resource deferral are not successful coverage.

Review identified two status/index contradictions before activation: retained
completed receipts were still counted as pending; later, a corrupted exclusion
hint could hide unfinished work before selected-row authentication. Both were
corrected. Pending/plannable totals are sealed and checked against index counts
before selectors/status; all relevant source insert/plan/ACK mutations update
them transactionally. Full verification also authenticates every mapping and
checks exact sealed plan EOF/ordinal coverage. Two exclusion checks failed before
the aggregate correction, then passed. Extra cases cover missing/corrupted
totals, source deletion and a balanced planning-hint swap.

Eight initial admission checks failed before the capability existed. Core queue
checks passed, and an integrated suite later passed 79 checks in 54.33 seconds.
After the index correction, the expanded queue, outbox, strict service, analysis,
publication and recovery suites passed **103 checks** in **71.64 seconds**, with
one unrelated installed-library deprecation warning. The worker fixture reopens
all actual fixture windows, verifies local-only/no-search behavior, persists ACKs
and restores completed coverage. It mocks model inference: it is plumbing proof,
not a new live model-quality or complete-installation claim. Earlier integration
failures included an unnamed INSERT invalidated by a new column and a lazy-import
test patch leaking into a later test; both were fixed without relaxing guards.

Remaining activation gates are unchanged: cross-version/partial-growth occurrence
deduplication, measured catalog/ledger cost, responsive bounded cadence and scoped
live activation. Paid automatic fallback also still needs durable reservations
and output-cost/reconciliation bounds. No original history, credential value,
service process, model or client settings were changed by these isolated checks.
After moving aggregate validation out of the per-receipt loop into the admission
batch boundary and leaving legacy publication independent of capture counters,
the final affected queue/outbox/publication/archive/recovery suites passed
**95 checks in 67.53 seconds**. Independent examination cleared the actual final
source diff for integration/push only; it did not clear live scheduler activation.

**Exact-candidate CI result:** `2c9d642` passed all five checks. Full locked suite
run `36816691630`, job `110223191280`: **2,497 passed / 13 skipped**, two warnings,
171.58 seconds. Privacy, clean imports, replay and benchmark dry run also passed.
This closes this source-only window-job integration check, not activation or the
full local-installation goal. This receipt accompanies the next code batch rather
than rerunning identical CI solely for documentation.

### 2026-10-01 growth preservation prerequisites (source-only)

Changed dependencies: secure archive prefix creation/authentication, private
same-occurrence window matching, and shared cited interpretation-contract hash.
No shared-service reload, model call, policy change or automatic activation.

- Red-first prefix tests: 15 failed / 8 passed against the preimplementation.
  Focused prefix plus original archive: 36 passed in 24.67 seconds. Added checks
  prove one source read pass, no old blob decryption and open-inode protection
  without an explicit expected path. Prefix/outbox/source-evidence/plans: 72
  passed in 44.87 seconds. Portable recovery checks valid, malformed and false
  byte-prefix relations; legacy absence is preserved.
- Recomputable interpretation identity: 13 tests failed before helper extraction;
  helper plus existing cited transport/recovery: 40 passed in 16.34 seconds.
  Existing staged hash formula is unchanged. Schema, prompt, classifier version,
  model/weights, text, role, timestamps and range mutations invalidate identity.
  Classifier-semantic changes still require a contract-version bump.
- Same-occurrence matcher: 10 tests failed before implementation, then 10 passed
  in 8.33 seconds. Tests use actual isolated encrypted archive/parser/window
  plans. Earlier windows match; new occurrences, rewrites, geometry changes,
  missing prior plans and differing native/physical coordinates do not. Corrupt
  prior plans fail closed. Matching creates no job/ACK and dispatches no model.
- Affected matcher/plan/identity/transport/capture-job/publication integration:
  **108 passed in 73.30 seconds**. Parent diff/whitespace checks passed;
  independent examination cleared the actual prefix, matcher and identity helper
  for source integration, with coverage/cadence/runtime claims explicitly excluded.

These are component proofs, not extraction quality or live growth processing.
Durable encrypted reuse coverage referencing a prior publication ACK, current
local model identity, bounded automatic cadence and measured scaling remain the
next activation dependencies. No old quote is re-published as a new citation by
this stage; cloud alias reuse is not admitted.

**Exact-candidate CI:** pushed `8de07e4aa048c7973ea2380c5ecc7cc6beb77e28` passed
all five checks. Full suite run `36819338643`, job `110231294669`: **2,546 passed /
13 skipped**, two warnings, 203.90 seconds. Clean imports, privacy, replay and
benchmark dry run also passed. This status receipt is held for the next code
batch; no identical test rerun or status-only CI push is necessary.

### 2026-10-01 planner failure fairness (source-only)

Changed dependencies: encrypted outbox planning state, authenticated scheduler
schema migration, source selection/deferral and worker cancellation/failure path.
No inference, live schema migration, service reload or scheduler activation.

- Nine planner-deferral tests failed against the preimplementation. Initial
  deferral/window/outbox checks: 57 passed in 39.91 seconds.
- Migration rollback (also `recover=False`), portable recovery, stale CAS,
  missing/tampered established state and actual worker A-fails/B-proceeds checks
  plus affected service/publication: 85 passed in 62.81 seconds.
- Added malformed authenticated state and off-thread cancellation/drain checks:
  targeted 19 passed in 10.36 seconds. Source-level failures do not advance
  coverage; unknown preparation errors are sanitized, unresolved and blocked.
- Independent examination cleared the actual persistence/fairness design and
  integration. Large-source durable parser resume remains specifically unproved;
  queueing four windows does not bound source preparation, and restart can still
  discard incomplete staging. This is not a live readiness/activation claim.
- Expanded migration/worker/original capture/archive checks: **114 passed in
  80.97 seconds**. A concurrent real-SQLite outbox writer then exposed a mixed
  root-seal/count read (`VaultIntegrityError`) in a red-first regression. Status
  and scheduling reads now pin one transaction; authentication is unchanged.
  Final concurrency/deferral/window/outbox checks: **68 passed in 42.47 seconds**.
  Independent review cleared the actual snapshot correction. The earlier broad
  proof remains scoped before this final reader change; exact-candidate CI is
  required for the integrated batch.

**Exact-candidate CI:** `777661db5c212db4e80a816fe65f8e7243a59661` passed all
five checks. Full suite run `36822223918`, job `110240063322`: **2,566 passed /
13 skipped**, two warnings, 148.73 seconds. Clean imports, privacy, replay and
benchmark dry run also passed. This status-only receipt is held for the next
code batch; no extra CI push is needed for an unchanged implementation.

### 2026-10-01 sealed-plan derivation and worker screening reuse

Changed dependencies: private source-unit-to-window derivation and encrypted
whole-unit worker-screening attestations. Automatic capture inference/backfill
and paid dispatch are not activated by these changes. The running HTTP service
was not reloaded.

- Five derivation tests: initially 2 failed / 3 passed; corrected derivation plus
  original plan/preservation checks: 26 passed in 21.11 seconds. A completed
  source-unit attempt retains its full raw EOF proof; the new plan drains the
  pinned authenticated fragment stream and checks parent/count/stat bindings,
  without a third raw decryption. General raw projection construction is unchanged.
- Three initial cache checks failed before implementation, then passed. The
  cache stores only encrypted hashes, lengths and exact source/attempt/unit/policy
  bindings. Publication follows normal whole-unit screening EOF. Tampering,
  transplant, new source/attempt/policy, late failure, backup and selected-page
  authentication are tested. Outgoing requests are always screened fresh.
- Expanded affected checks found the existing public-read late-tamper regression;
  persistent reuse was narrowed to worker preparation, preserving fresh whole-unit
  public reads. The regression assertions were not relaxed. A mocked private
  method was adapted for the added keyword argument. Final combined ledger,
  cache, derivation, plan and preservation suite: **95 passed in 63.43 seconds**.
  Earlier source/analysis/cache/ledger run had 100 passed before that correction.
- Independent actual-diff review was CLEAR for derivation, cache storage/privacy
  and the preserved public-read boundary. Cache verification includes obsolete
  policy entries in portable recovery. It attests originally authenticated
  immutable units, not fresh on-disk authentication of every unselected frame
  for each worker lookup. Full-store verification retains that corruption check.
- Real local archived-unit proof: initial preparation drained one whole unit;
  a fresh worker reused the exact screening result with zero whole-unit scans.
  Observed timings were 32.16 ms and 1.42 ms for this small unit only; no general
  machine/backlog speedup is claimed. The host-only proof initially selected no
  sample because roles are capitalized; its selector now normalizes case. No
  transcript, credential, ID or model output was printed; zero model calls.
- The existing live service independently passed actual MCP cited lookup/search,
  source-follow and one 1,443-character redacted transcript page. Anonymous and
  wrong-token reads returned 401; Ollama had zero resident models. This validates
  existing live agent retrieval, not activation of the new preparation code.

The one derived screening record was written encrypted to the existing local
archive. No original transcript, model setting, credential vault, service process
or other application was changed. Durable parser restart/resume, growth-coverage
reuse, automatic cadence, paid reservations and remaining UI work stay open.

**Exact-candidate CI:** preparation/cache commit `1ed4279` passed all five checks.
Full run `36830522432`, job `110265830334`: **2,583 passed / 13 skipped**, two
warnings, 195.63 seconds. Imports, privacy, replay and benchmark dry run passed
too. This evidence covers that pushed candidate, not the subsequent reuse slice.

### 2026-10-01 direct-parent analysis reuse (source integration)

Changed dependencies: local generation-contract identity, additive encrypted
reuse receipts, capture coverage/status and the optional private worker callback.
No capture timer, live journal migration, service reload or paid/model dispatch.

Review identified omitted generation options in the earlier identity helper.
Two new checks failed first. New local identities now include effective options;
actual outcomes use the options from the body sent, not defaults reread afterward.
Existing staged identities/receipts remain historical and are not rewritten;
they do not acquire new reuse authority. Remote identities remain unchanged.
Identity plus prior transport/publication checks: 64 passed in 32.83 seconds.

Eleven reuse cases failed before the private admission method existed, then
passed in 18.80 seconds. Reuse has a distinct terminal state and required AEAD
receipt bound to its current immutable target/window and a direct original local
parent extraction/ACK. Expected original ledger refs, including explicit empty
extractions, authenticate before admission. Source ACK and reuse receipt commit
together after lease/cancellation and exact original/current binding rechecks.
Reuse publishes no new event or quote and retains original citations. Restore
does not compare historical coverage against today's installed models/options.

Expanded storage/queue/publication/identity/transport checks: 105 passed in
78.49 seconds. Added final-weights-guard and worker-no-POST cases both failed
before wiring. The callback receives the actually selected local model, freshly
read digest, exact body options and validated loopback base. It runs only for
capture jobs with a nonzero source version, rechecks weights outside the writer
and returns `reused` before HTTP client construction. A miss retains normal
inference admission and rechecks weights before POST. Search, remote and original
version-zero paths do not use it. Combined affected suite: **108 passed in
88.77 seconds**.

Additional checks prove cancelled workers drain the blocked reuse writer before
shutdown and changed weights after a reuse miss prevent any POST. Final affected
reuse/transport checks: **45 passed in 50.58 seconds**. Independent actual-diff
and final-results review was CLEAR for source integration/push, not activation.
These are isolated encrypted-store plumbing tests, not synthetic responses
presented as live local model quality or installed scheduler behavior.

Direct-parent-only admission intentionally misses when the parent itself reused
analysis. Multi-growth amortization remains open. Resource telemetry/GPU-busy
rules still govern route admission; this slice does not bypass them to load a
model. Automatic capture cadence, durable parser restart/resume, paid reservations,
full backlog scaling, credential review and UI completion remain open goal work.

Actual configured Codex stdio launch recheck: 20 core tools, three cited-memory
tools present, a real saved provisional memory retrieved, zero model calls and
no settings changed. Four saved client profiles matched the authenticated bridge.
The generic profile smoke initially launched without their explicit `core`
environment and listed the default 58-tool set; its verified-profile mode now
reproduces that setting instead of implying the generic launch was a configured
client test. The corrected live smoke reports 20 tools and successful context
retrieval. Existing running clients may retain an earlier tool catalog until
reconnection; these checks do not assert a client app was restarted.
