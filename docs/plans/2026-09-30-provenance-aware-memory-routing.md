# Provenance-aware memory routing (source-unit foundation implemented; filing pending)

## Decision and observable outcome

Muninn must not treat a model's interpretation as a verified memory merely
because it is fluent. It must capture original material durably, classify
bounded candidate knowledge, file only evidence-supported high-confidence
items in the right project/type/time scope, and preserve unresolved cases for
agent or user review. Agents must be able to follow an item back to its
authenticated source and surrounding transcript. Credential *values* remain
in the separate portable credential vault and never enter ordinary search.

The user chose evidence-gated automatic filing on 2026-09-30. A later
contradictory statement is not by itself proof that the earlier one was false.
The local model may reject clear noncredentials; it cannot silently promote
uncertain credential candidates. User consultation is required when the
evidence cannot settle a consequential classification or conflict.

## Current behavior that this design must not mistake for completion

- The strict archive encrypts full source snapshots and retains source path,
  provider, capture time, source mtime, and version. Capture time is **not**
  the event time of a statement. Provider-level message timestamps exist in
  some parsers, but the present secure-hit model window passes message text
  alone. A 256 KiB physical-line limit in its fast path can also lose
  structured context even though the streaming projection supports large
  values.
  **Update:** the model-window fast path now uses authenticated streaming source
  units, eliminating that physical-line cap. Encrypted sidecars preserve unit
  provenance for review; ordinary analysis still lacks durable claim filing.
- Strict secure analysis interprets a pertinent search hit in a bounded
  window. It does not provide full historical-archive enrichment coverage.
  The encrypted analysis result in the search journal expires after 24 hours;
  it is not durable knowledge with a source citation.
- Legacy history insight storage is disabled in strict mode. Even there,
  output is marked model-inferred/unverified, not claim-to-source verified.
  Optional ordinary-memory conflict detection and semantic deduplication
  default off. Enabling the legacy conflict resolver unchanged would allow
  age/importance heuristics to supersede or merge claims without sufficient
  temporal evidence.
- Historical ordinary import code can store source paths in searchable
  metadata and can collapse divergent equal-key sessions by keeping only the
  fullest one. Existing derived records need a metadata-only privacy/coverage
  audit and recoverable quarantine or reindex plan. The authenticated raw
  archive remains the source of truth; migration must not delete it.
- Credential ambiguity has its own encrypted queue and explicit local review
  path. It is a special lane of this workflow, not a substitute for review of
  project identity, duplicate versus distinct, facts, decisions, tasks, and
  contradictory or time-scoped claims.
- Current transcript credential scanning records provider as the queue's
  `project` and does not carry original source path or message timestamp into
  the classifier input. Its value-based review groups can span distinct
  source snapshots. A model judgment about one representative must not reject
  an entire cross-source group until all relevant contexts are evaluated.
  Existing transcript rows can be joined back to an authenticated archive
  blob by recomputing the recorded SHA-256 fingerprint of
  `vault_id:blob_id:content_sha256`. The manifest then supplies the original
  path/version, mtime, and capture time; none is a substitute for an absent
  provider event timestamp. A missing or nonunique join is `source_unverified`.
  Keep full paths inside encrypted state or a local ephemeral lookup; models
  receive only the bounded project/source-type context they need.
  For Codex specifically, session file location identifies a session/date,
  not reliably its project. Parse `session_meta` and each `turn_context.cwd`
  as bounded provider metadata, attaching the applicable cwd to each turn;
  never apply the session's final cwd retroactively to earlier turns.

**Implemented foundation:** immutable encrypted source-unit pages, per-occurrence
credential context replay, model-weight-bound review caches, keyset review
pagination that retains UNKNOWN as pending, and portable sidecar recovery.
Both streaming source passes authenticate their complete size/hash before
publication. Gemini supports pretty-printed container JSON; other providers
retain strict JSONL. See the source-unit integration evidence receipt for the
bounded real-provider/model checks and exact remaining gaps.

The 2026-09-30 read-only local join audit found 24,627 ambiguity rows across
1,074 transcript snapshots, with 0 missing and 0 nonunique archive joins.
This proves source lookup coverage for those rows, **not** that their values
are credentials, that message event times are available, or that group-wide
model decisions are safe. Do not run a mass classifier pass on the current
name/reason/value-only input.

## Options considered

1. **Chosen: encrypted evidence ledger attached to immutable archive
   snapshots.** Small, durable claim/review records reference authenticated
   source units. It reuses local encryption and backup boundaries while
   keeping provisional content out of ordinary vector/BM25 indexes.
2. **Rejected: add review flags directly to current memory records.** Less
   code, but uncertain text could enter searchable indexes, source/turn
   evidence would be weak, and rollback of bad model assertions would be hard.
3. **Deferred: separate temporal knowledge-graph service.** Useful for much
   larger multi-user reasoning, but unnecessary operational and recovery
   complexity for this local installation.

## Data and state boundaries

Each source unit carries an opaque source ID, immutable snapshot blob/hash
and version, provider, authenticated record/turn ordinal (or byte-range where
safe), role/content type, source path held only inside encrypted state,
project evidence, and separate timestamps: provider event time, source mtime,
archive capture time. Timestamp value and *basis* travel together; absent or
unparseable event time is `unknown`, never backfilled from file mtime as if it
were the conversation time. Native IDs and content digests deduplicate units
across append-only snapshots without conflating distinct occurrences.

Candidates have type (`fact`, `preference`, `decision`, `task`, `procedure`,
`project_attribution`, `duplicate`, `conflict`, or `possible_credential`),
proposed destination, scope, temporal qualifier, confidence, source-unit
reference(s), model/rule version, and state. States are `pending`,
`provisional`, `filed`, `rejected`, `superseded_with_evidence`, and
`needs_user`. All changes append an audit decision; no model can erase the
source or prior interpretation. A candidate is filed only if its source is
authenticated, the type/project/temporal scope is supported, no unresolved
contradiction or secret risk exists, and the policy threshold is met.
Potential secrets divert to the local credential lane before any ZDR request.
No candidate enters an ordinary content, metadata, vector, or BM25 index
until a mandatory secret-diversion/redaction gate has passed. Existing legacy
indexed records need a verified-backup migration; fixing future writes alone
does not close prior exposure.

The ledger stores sensitive text only encrypted. Ordinary agent search may
return bounded redacted text, provenance metadata, status, and an opaque
source capability. A local authorized interpreter may inspect the original
window; ZDR may do so only under the user's persisted consent, provider ZDR
constraints, and spending thresholds. Credential values never appear in
ordinary agent results. Agent/user review can ask for surrounding source
pages before deciding; explicit local credential use remains separate.

## Processing order and resource policy

1. Capture and authenticate immutable raw snapshots immediately; do not wait
   for a model, GPU, remote provider, or review. CPU-only indexing and bounded
   redacted projection continue independently.
2. Stream source units into resumable, coverage-tracked work keyed by snapshot
   and parser version. Long JSON strings and multi-GB transcripts are chunked
   with bounded buffers; completion requires end-to-end authentication and
   contiguous unit coverage. A retry skips verified units, not whole files by
   name alone.
3. Prioritize pertinent active-conversation units, then new history, then the
   historical backlog. Batch candidate extraction under measured resource
   limits; reuse an eligible resident local model only while GPU idle and
   headroom remains, use only short bounded model residency during a batch and
   return to idle afterward, and use the approved ZDR
   path only where privacy/budget gates permit. Defer without losing work if
   no safe route fits. Capture/search remain available throughout.
4. Apply deterministic rules first, then bounded model classification with
   cited evidence, then user review only for unresolved consequential cases.
   Search must distinguish verified/filed, provisional, and source-only hits.
   Agents can retrieve related transcript pages if a summary is insufficient.
   Divergent sessions with one native ID remain distinct observations until
   their relationship is resolved.

## Failures, recovery, and tests

The implementation must handle source growth during a pass, malformed or
unknown provider records, missing event time, conflicting project evidence,
large single messages, cross-snapshot duplicates, late archive corruption,
model refusal/malformed output, GPU contention/OOM, provider outage, budget
exhaustion, user consent revocation, crash during ledger publication, and
backup/restore on another machine. Failures retain source and checkpoint or
defer; they cannot turn an incomplete pass into a `filed` claim.

Before live activation: red-first unit tests for provenance/time precedence,
dedup and temporal conflicts; integrity/rollback and secret-exclusion tests;
bounded-memory long-source tests; local and ZDR routing tests; restore drill
including the ledger; representative real-history end-to-end checks for
agent source-following and correct project/time filing; independent review of
design, persistence/security diff, and actual live result. Record observed
durations and resource peaks rather than promising a fixed time. Keep the
current live listener running until migration is backed up and separately
authorized for restart.

The privacy gate must assert that a synthetic credential-like value and
private path are absent from ordinary content, metadata, vector payload,
BM25, logs, and agent responses. A separate nonsecret source-reference test
must prove authorized local interpretation can recover the original location
and provider event time. Verify model residency returns to idle within its
configured short duration even on early termination.

## Implementation slices

1. Fix proven resident-model route regression and verify live triage can
   continue without leaving a model resident indefinitely.
2. Add source-unit provenance extraction and immutable encrypted ledger with
   crash-safe coverage/checkpoints and portable backup.
3. Add evidence-gated classifier/review decisions, credential diversion,
   temporal conflict rules, and agent-facing metadata/source navigation.
4. Backfill historical snapshots under resource/budget policy; validate real
   local/ZDR and restore paths; then add UI controls after the core is proven.

## Next implementation boundary: durable candidate ledger

The next slice is an isolated encrypted ledger and evidence gate, not immediate
activation of an archive-wide inference job. Its observable outcome is that a
bounded source window produces durable typed candidates with exact citations;
failed, missing or contradictory evidence cannot become a filed item. This
closes the current 24-hour-analysis-expiry gap before scheduling more inference.

- Store under the archive recovery envelope in `memory-ledger/ledger.sqlite3`,
  with an independent HKDF/AAD domain. Only opaque HMAC identifiers, sequence
  numbers and necessary structural states are plaintext. Text, project labels,
  source references, model proposals and review decisions are encrypted.
  Events form an AEAD-authenticated sequence with an encrypted committed head
  and previous-ciphertext digest. Reads/restore validate the full chain before
  returning a public result. This detects corruption, missing events and changed
  references, but cannot detect replacement of the entire database by a valid
  older copy without an external freshness anchor; rollback resistance is not
  claimed. Writes commit event and head in one FULL-synchronous transaction.
  The first publication API validates the whole chain in its write transaction;
  an authenticated tail is insufficient proof that an earlier prefix is sound.
  Measure/amortize that validation for background batches before large backfill.
- An immutable citation identifies snapshot hash/blob/version, source-unit
  ordinal, parser version and bounded window offsets/digest. Quote checks use
  the original authenticated source, not model paraphrases. Context must retain
  the source role, cwd evidence and event-time basis. A source observation is
  not a world-fact verification. Assistant output remains model-inferred.
- Candidate extraction can propose type/scope/text plus an exact supporting
  quote. Candidate IDs bind source occurrence, schema/policy version and model
  weight identity. Repeating a job is idempotent without collapsing distinct
  occurrences or conflicting snapshot versions. Decisions append; they do not
  destroy earlier candidates or sources. No automatic newest-wins rule.
- `filed` requires exact quote, known project/event basis, a passed secret gate
  and a narrowly supported source assertion. Unsupported paraphrases, missing
  scope/time, assistant assertions and possible contradictions are provisional.
  Any possible credential is diverted to credential review and cannot be an
  ordinary filing. Model confidence by itself never authorizes promotion.
- Until conflict identity is evidence-supported, automatic filing is restricted
  to source observations, not consolidated truth. Typed summaries and temporal
  relationships remain provisional. User-review decisions must name evidence
  or explicit user authority and preserve contrary observations.
- Persist `epistemic_kind=source_observation` and `truth_status=unverified_assertion`
  for a filed verbatim user observation; assistant/model interpretations are
  always explicitly labeled and provisional. The first gate only auto-files
  type `observation` only for whole-user-message equality, never a stripped
  quotation, negation or reported speech; proposed facts/preferences/decisions/tasks need further
  type/conflict evidence. Public read/search contracts retain these labels.
- Screen the entire serialized remote model input, including the evidence
  window and supporting quote, before dispatch. Unknown screening outcome
  blocks ZDR. Safe claim text cannot exempt a credential-bearing context. No
  private cwd/path/native ID enters that input; project references are opaque.
  Local credential interpretation remains in its separate approved lane.
- The initial citation targets an independently authenticated bounded encrypted
  source-fragment page from a fully sealed SourceEvidenceStore attempt. This
  avoids rereading a multi-GB source for each candidate. Fragment-boundary
  claims remain provisional or await a contiguous-window citation; the source
  is not discarded or declared fully interpreted because one page was handled.
  The gate streams the complete authenticated source unit through redaction
  before exposing/admitting any selected window, so labels/open quotes cannot
  hide in earlier chunks. Indexed binary seeks locate the unit's encrypted
  fragment range, without scanning preceding conversations. Digest state and
  a 128-entry immutable-unit screening cache keep memory bounded. There is no
  whole-unit size cap. Cross-fragment claim citations and automatic enrichment
  scheduling remain subsequent dependencies, not proven by this gate alone.
- Publication is transactional after complete source verification. A crash or
  late corruption leaves no published page/checkpoint. Bounded windows and
  continuation offsets support arbitrarily long records; no whole-source size
  ceiling and no claim of completion on a partial iterator.
- Backups take a consistent SQLite ciphertext snapshot and verify ledger
  integrity before declaring restore success. No new key or passphrase prompt
  is necessary beyond the existing archive recovery credential.

Cheapest proof: isolated red-first cases for quote mismatch, unknown event/cwd,
assistant-vs-user role, secret diversion, distinct same-text occurrences,
idempotent retry, crash-before-publication, temporal disagreement retention,
ciphertext/AAD tamper and portable restore. Then one bounded real source window,
local and approved ZDR route, source-following search, and idle-model release.
Only after those pass does the scheduling slice prioritize active/new/backlog
work. UI controls and historical backfill are subsequent dependencies, not
implied by the ledger's unit tests.

## Automatic integration admission requirements

Independent review of the actual ledger/analysis boundary identified these
remaining blockers before connecting model results to automatic filing:

1. Bind a bounded source-fragment window and its exact offsets to the existing
   encrypted analysis job **before** inference. The current summary window
   supplies no authenticated claim citation. Persist parser/schema/model
   identity and input digest; retry must reuse that window, not silently select
   a different excerpt. Cross-fragment coverage is separate resumable work.
   **Component implemented:** `CitedAnalysisSource` produces/reopens a private
   descriptor bound to snapshot, attempt, page coordinates, parser and canonical
   input digest. Boundary hits retain authenticated adjacent-page context in a
   bounded window; explicit single-page citation ranges reject cross-page
   fabricated quotes. Queue binding and prompting remain to be connected.
2. Add exact-quote proposals to the internal extraction contract. Check each
   against the persisted source window. Model-origin proposals stay explicitly
   provisional even when they echo a whole user message; quote equality is
   source support, not approval of the proposed type, scope or truth. Preserve
   the public four-field analysis response until agent contracts are updated.
   **Component implemented:** the model-only batch API fixes proposal origin
   to `model` and cannot accept a caller's source-rule override. Those candidates
   remain provisional even for whole-message exact echoes. Trusted direct-source
   rule callers retain the single-record API and historical ref derivation.
   Model refs bind their origin separately; older records report
   `legacy_unrecorded`, not an invented origin. Portable recovery retains this
   distinction. The automatic worker has not yet been connected to the API.
3. Screen the complete actual serialized remote request, not merely the claim
   or response. Credential-bearing or unknown inputs stay in the local lane;
   consent, budget, lease and ZDR settings do not override that boundary.
4. Reuse existing lease, remote-dispatch/unsent markers and consent-generation
   checks. An interrupted possibly sent request is outcome-unknown, not an
   automatic duplicate paid request. Persist accepted proposal refs atomically
   before marking extraction complete; recover ledger-commit/job-ack crashes
   by deterministic idempotency. Do not confuse a quota availability query
   with a spend reservation.
   **Reviewed queue design:** short transactions bind the window, stage validated
   encrypted extraction, and fence publication admission/acknowledgment.
   Source authentication and ledger append run outside the journal writer lock,
   so CPU capture and heartbeats are not blocked by a growing ledger scan.
   A durable publication-only state replays the immutable stage without another
   model call. Deterministic candidate refs recover a ledger-commit/job-ack crash.
   Cancellation and publication admission are mutually exclusive; after
   publication admission cancellation must report that it is too late, not
   pretend already committed claims can be rolled back. Expired staged local
   publication is retryable; possibly dispatched remote work with no durable
   response remains outcome-unknown. Stage AAD binds job, target/window and
   result identity. Prove cancel/admission races, stale-lease acknowledgment,
   no duplicate inference after publication crash, and tampered stage rejection.
5. The bounded `record_batch` component validates all citations before a single
   transaction and authenticates the prior ledger chain once for up to 64
   proposals. Single-record callers retain the same semantics. This amortizes
   validation but does **not** remove quadratic cumulative full-chain cost.
   Measure representative growth before enabling a historical backfill; add
   an authenticated scalable validation boundary if needed, without accepting
   an unverified old prefix or holding a source reader through inference.

The component is tested independently of the live service. It does not by
itself activate historical inference, resolve conflicts, establish extraction
coverage, or federate ledger results into agent search.
