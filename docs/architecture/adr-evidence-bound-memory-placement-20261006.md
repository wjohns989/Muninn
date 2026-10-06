# Evidence-bound automatic memory placement

Status: accepted design; implementation and live semantic-quality proof pending.

## Concrete missing dependency

The authenticated local ledger baseline has 5,950 candidates and no review
decisions. Of these, 4,063 have `credential_risk=false`: one legacy filed source
observation and 4,062 provisional model candidates. All 4,063 have known project
attribution and provider-record timestamps. Among provisional candidates are
3,699 tasks, nine conflicts, one duplicate and other typed interpretations.
These are source/candidate counts, not distinct real-world tasks or verified
facts. The other 1,887 candidates are outside this noncredential baseline.

Production publication calls `record_batch`, which intentionally leaves model
interpretations provisional. The direct-source `record` helper has no production
caller; wiring only its exact whole-user observation rule would not resolve
general semantic ambiguity and could create duplicate candidate identities.

## Decision and alternatives

Use a separate, encrypted placement decision attached to the EXISTING candidate
ID. Accepted project/type placement is not verified truth, completed work,
execution authority, or proof that no conflicting evidence exists elsewhere.
Retain the original citation, extraction type, project/time provenance,
`proposal_origin`, epistemic label and truth status unchanged. Public retrieval
must expose placement status/type separately; human decisions remain authoritative.

Do not (a) relax `record_batch` to let model output claim source-rule authority,
(b) republish a second direct-source candidate, (c) turn a confidence score into
truth, or (d) ask the user to manually file every otherwise clear interpretation.
The trade-off is one explicit classification lifecycle instead of conflating
extraction, placement and truth. Reuse the existing journal, encrypted ledger,
single inference consumer, privacy policy and cost accounting; no new service.

## End-to-end lifecycle

1. Discover classification work from authenticated durable publication ACKs,
   including old backlog ACKs, with a resumable idempotent checkpoint. An
   after-publication callback alone is insufficient: a crash could miss work.
   Reused extraction ACKs retain occurrences but cannot multiply classification
   bills for the same authenticated classification input.
2. Prepare an immutable bounded context from freshly authenticated, privacy-safe
   candidate/source views and explicitly selected peer evidence. Bind candidate
   and citation digests, current human/decision revision, source project/time
   provenance, peer identities and comparison coverage. Preserve event time and
   time basis; never substitute capture or publication time. Unknown source/time
   attribution and partial comparison coverage remain explicit.
3. Use a SEPARATE classification contract/purpose/input identity. The classifier
   may propose a bucket, contextual contradiction/duplicate relationships or a
   clarification. Exact citation checks and provenance gates, not confidence
   alone, control accepted placement. Comparisons spanning pages/sources must
   retain their authenticated evidence; one page is not the complete conflict set.
4. Stage and recover classification results without another dispatch. Existing
   Luna remote consent, screening, daily/monthly admission accounting, foreground
   priority and the serial paid-owner fence apply. No local-model fallback, paid
   checkpoint repacking/resend, or retry of uncertain sent work is introduced.
   Credential values remain in their separate vault/local-use boundary. No
   temporary-retention route receives an unscreened private classification input.
5. Reauthenticate the input/privacy/decision revision before commit. Append one
   evidence-bound classification event with CAS; do not mutate candidates or
   override intervening or prior human decisions. Unknown attribution,
   unsupported classification or unresolved conflict requires consultation, not
   fabricated confidence. Contradictory observations may be historically valid;
   ordering by timestamp alone does not grant supersession authority.
6. Read-back placement, original truth labels and citations. Update retrieval,
   grouped consultation and the local UI to show accepted placement separately
   from unresolved review/truth status. A hidden event is not delivered filing.
   Human answers invalidate dependent stale classifications and are retained.

## Persistence and integration gates

Extend authenticated event readers, chain/snapshot checks, public reads,
preimages and portable backup/restore together. Old readers reject new event
kinds, so no classification event may be written to the shared ledger before
the reviewed compatible service is installed. Keep existing extraction job and
reuse identities unchanged; classification gets its own stage/receipt proof.

The legacy `mark_needs_user` method is NOT the commit boundary: it has no CAS
and could overwrite a human decision. This new workflow must not use it.
Source-rule filing is likewise not borrowed by classifier output.

## Smallest discriminating acceptance

- Existing backlog and new ACK discovery; duplicate and reused publications.
- Crash after ACK and after classification stage, without redispatch.
- Human-decision/revision race, revocation, unavailable budget and owned batch.
- Contradictions across pages/sources; known ordering and unknown time without
  invented recency, automatic supersession or cross-project contamination.
- Exact citation and comparison binding; partial context and projected-range
  denial; unsafe source/context and credential exclusion.
- Durable placement visible in real agent retrieval and consultation, with
  original truth/provenance unchanged; rejection never deletes original data.
- Portable recovery of classifications, queued work and human decisions.
- Representative live Luna results reviewed for semantic placement quality,
  latency, retries and total billed cost before expanding classification runs.

Independent read-only design review cleared this boundary and rejected the
direct-source-only shortcut as a completion claim. This document and the
metadata baseline do not claim an implemented classifier, model success,
resolved ambiguity queue or completed local installation.
