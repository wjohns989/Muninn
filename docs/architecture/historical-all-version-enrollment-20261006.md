# Historical snapshot-version coverage

Status: accepted after independent design/diff review and isolated validation.
Live installation/enrollment evidence is recorded separately; acceptance of the
design alone is not interpreted backlog completion.

## Concrete gap

A query-only census against authenticated manifest generation 3214 found 6183
eligible transcript snapshot versions. 2125 had no enrichment index record:
2124 pre-watermark older versions and one later version. No latest version was
missing. Index presence is not interpreted coverage or publication proof.

The existing completed latest-v1 enrollment is pinned to generation 2786. An
authenticated census of that exact pin found all 2124 missing pre-watermark
versions within its 6603 total snapshots (5755 eligible transcript versions).
The gap therefore can be closed without changing the original pin or old grants.

## Decision

Add a separate sealed all-versions cursor in the existing capture journal. Require
the original latest enrollment to be complete first; a fresh operator workflow
runs latest enrollment followed by the all-versions pass. Both selections share
the existing immutable manifest identity. Existing grant wire format, AAD, latest
cursor, paid requests and batch/publication identities remain unchanged.

Each call examines at most 128 snapshot/empty-source positions. Preserve a
canonical source/version cursor, exact visited/queued/existing/excluded counts,
authenticated pin/count checks, stale-writer CAS and atomic receipts+cursor commit.
Existing receipt IDs must authenticate and are not duplicated. Verify the union
of processed latest and processed all-version selections, rejecting orphan grants,
invalid progress and incompatible pins. Query-only preview must not create schema.
Normal post-watermark reconciliation remains responsible for later captures.

Expose an explicit CLI all-versions preview/apply option and separate status. Its
complete flag means enrollment, never completed interpretation. Maintenance writers
must use recover=False and never reset another live worker's claim.

Use one authenticated selection-union helper for both journal verification and
batch activation. Batch activation currently reconstructs latest-only IDs itself;
leaving that consumer unchanged would strand newly enrolled older versions outside
the batch lane. Its owner/policy/privacy/budget/ordering gates stay unchanged.

| Consumer | Required behavior |
| --- | --- |
| Grant reader | Preserve original pin, grant format and authenticated identity |
| Verification/restore | Verify the exact union, not merely the old latest prefix |
| Batch activation | Select only IDs in that same authenticated union |
| Status/preview | Separate latest and all-version enrollment from interpretation |
| Later capture reconciler | Preserve its existing post-watermark cursor |
| Existing paid checkpoint | Preserve all stored requests, bills and ACK identities |

## Alternatives and trade-offs

- Keep latest-only: smallest change, but demonstrably leaves older versions out;
  rejected as completion behavior.
- Rewrite the original cursor/grants to a new pin: creates unnecessary compatibility
  and recovery risks for existing paid work; rejected.
- Add one same-pin cursor: preserves existing identities and limits new persistence
  to one sealed progress row; selected for implementation review.

The new pass visits excluded and already-enrolled versions too; its bounded metadata
work is not a model request. It may make genuinely uncovered windows eligible for
future paid processing, still subject to unchanged privacy, serial checkpoint,
deduplication and $5/day/$50/month controls. No monetary savings or total new-window
count is inferred from snapshot counts. Existing reuse has been checked: it requires
exact authenticated evidence, and does not provide a general reverse-version cache.
Treat uncovered older windows as potentially new paid work under W's existing
entire-backlog authorization and unchanged limits. Do not claim deduplication savings
or spend on a new cache without measured benefit. Mere later-version existence cannot
stand in for exact evidence reuse.

Source order with ascending versions is a stable resumable traversal, not proof of
global conversation-time ordering. Existing ordering of planned windows is preserved;
unplanned coverage must not be advertised as globally chronological.

## Smallest acceptance proof

Use isolated encrypted archives with distinct content retained only in an older
version. Prove all eligible versions become independently readable receipts, old
latest seals/grants stay unchanged, bounded progress resumes across reopen/portable
restore, concurrent writers cannot regress it, failure rolls cursor+inserts back,
and tampered positions/counts/pins/grants fail closed. Exercise mixed provider/kind,
empty source and exhausted cursor cases plus query-only no-schema/no-write preview.
Reuse focused unchanged batch identity/publication tests. Before live enrollment,
independently review actual results and the exact installation/backup plan; preserve
the current owned paid checkpoint and all encrypted inputs/results.
