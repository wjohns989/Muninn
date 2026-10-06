# Evidence-bound placement implementation checkpoint

The known missing production dependency is filing noncredential model memories
without confusing extraction, classification, verified truth or human authority.
The architecture decision is in
`docs/architecture/adr-evidence-bound-memory-placement-20261006.md`.

This checkpoint adds the input contract, encrypted cohort placement replay and
durable publication-ACK discovery/staging. It is not a live classifier completion
receipt. The existing paid batch and shared service have not been modified.

## Observed proof

- A real existing local candidate prepared successfully through the read-only
  ledger: original provider time and project known, source-fragment coordinates,
  2,230 serialized payload bytes, opaque local references excluded from the model
  payload. No text/value was printed; zero model calls and ledger writes.
- Placement regression first failed because `commit_classification` did not
  exist. Subsequent isolated tests prove durable placement visible through get,
  source following, search and grouped consultation, while original candidate,
  type, review state, citation and truth remain unchanged.
- Effective replay revisions include all members of a cohort even though one
  existing candidate owns its physical event. Human decisions invalidate
  dependent placements transitively. A peer invalidated during the new event
  is not silently rebound to a revision the model never observed.
- Exact staged replay after a later human decision is a no-op, not restoration
  of an old acceptance. Correctly sealed but invalid classification events stop
  reference verification and subsequent publication as well as public reads.
- Discovery registers an ACK and work atomically, catches late ACKs of older
  jobs and does not multiply work for duplicate candidate publication results.
- Staging and publication remain separate. Charged malformed replies and stale
  evidence require consultation; sent expiry remains unknown and cannot dispatch
  again. Backups reject orphan ledger stages, while allowing the genuine
  ledger-commit-before-journal-ACK recovery window.
- Classification admissions bind job and input before sending. Extraction and
  classification cannot use each other's charges. Portable encrypted accounting
  retains that binding with remote permissions disabled on restore.

Independent source review identified and drove corrections to replay, semantic
verification and admission ownership. Valid operator reconciliation authenticates
bookkeeping only: it preserves uncertain work and permits recovery, but does not
prove a provider response, allow staging, or permit redispatch.

Validation completed locally on October 6, 2026:

- The affected ledger, classification, publication, accounting and agent-facing
  read suite passed 235 tests in 104.83 seconds before the final admission/recovery
  changes. This is retained evidence for the unchanged read/replay dependencies,
  not a claim that the earlier run covered subsequent changes.
- After the final admission invariants, exact pre-ACK stage comparison and
  operator reconciliation corrections, the classification jobs, placement and
  accounting suites passed 51 tests in 23.24 seconds. These include real local
  admission bookkeeping and encrypted portable recovery; provider replies are
  synthetic and no provider request was made.
- Independent source re-review cleared the changed recovery issue. This is
  source clearance, not live activation or full-installation approval.
- No full repository suite or merge was performed for this checkpoint.

## Required next integration

Connect the existing single inference consumer to these durable jobs, with paid
checkpoint, foreground, consent and spending gates. Select bounded related
cohorts/peers instead of thousands of single-candidate calls. Install compatible
readers before any live classification event, retain encrypted preimages, and
review representative Luna results for semantic quality and total billed cost.
The current input/staging checks do not prove an active classifier, a resolved
historical ambiguity queue, complete hooks, or the full local installation.
