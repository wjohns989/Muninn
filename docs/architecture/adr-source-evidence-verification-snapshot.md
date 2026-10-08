# Source-evidence recovery verification: reuse seals within one read snapshot

## Status

Implemented; focused validation recorded in the local recovery receipt.

## Context

Recovery must authenticate every complete source-evidence attempt, page sequence,
unit count and cached screening binding. The old verifier streamed each attempt,
then recounted its entire page index for every screening row. For P pages and R
screen rows in an attempt, that introduced an avoidable O(P * R) count scan.
It also opened independent readers between coverage and screening validation.

## Decision

Use one read-only SQLite connection and transaction for a verification call.
Authenticate each attempt's seal/count and stream every page once, retaining only
identity/count/statistics per attempt. Authenticate every screening record, check
its exact source/attempt identity, and use the existing bounded indexed unit seek
against that same snapshot. Ordinary unit readers still independently authenticate
their seal/count. No persistent verification cache, schema change, plaintext body
cache, proof skipping, model dispatch or backup deletion is introduced.

## Alternatives and trade-offs

- Repeating the checks preserves the old behavior but makes large backups
  unnecessarily expensive and does not provide one coherent database snapshot.
- Persisting a verified cache introduces invalidation and authority problems.
- Retaining all plaintext units would increase memory use and sensitive exposure.
- A pinned reader removes repeated count scans, but remaining unit seeks still
  cost O(R * log(P)); this is not a claim of wholly linear verification.

The reader never changes journal mode. On rollback-journal databases a long read
transaction can delay writers; full recovery verification belongs on the closed
backup/staging copy. WAL concurrency tests prove snapshot consistency without
changing the live database's journal policy. Metadata memory remains proportional
to attempt count, as the existing archive/attempt inventory already was.

## Proof and operational boundary

Regression tests reproduce repeated seal/count authentication in the old code.
Tests cover multiple attempts, no screens through existing fixtures, missing and
extra pages, altered completion, page/screen AEAD failure, correctly encrypted but
false unit/source/attempt bindings, portable backup/restore and concurrent WAL
mutation. A later verification must reject the mutation: reuse lasts only for
the current read transaction.

An already-running backup retains its loaded implementation. This change does
not justify interrupting it, announcing its completion, or retiring older full
recovery copies. Revisit remaining seek costs only with representative completed
verification timing; do not launch another large backup solely as a benchmark.
