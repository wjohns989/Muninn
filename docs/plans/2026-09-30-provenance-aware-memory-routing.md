# Provenance-aware memory routing (proposed architecture; implementation pending)

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
