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

## Consumer integration checkpoint

The source now connects the existing single inference consumer to the durable
classification jobs. Enrollment groups candidates from the same project and
original publication ACK, at most 12 per cohort, with an encrypted membership
index. Existing jobs are authenticated and seeded without rewriting them;
missing or foreign members fail verification instead of silently rebilling.
The 25-ref splitter test proves the helper's 12/12/1 bound, not a fabricated
25-candidate publication ACK (the actual publication contract is bounded).

The consumer recovers staged publication with inference disabled. New dispatch
requires the same passed paid owner at selection and immediately before POST,
fresh source/human revisions, current consent, foreground priority and managed
spending. Native writes are drained on cancellation. Proven-unsent release is
recoverable; sent uncertainty is never automatically resent. Tail scheduling
does not require an additional clean batch or unused batch quota.

Live expansion is deliberately not implemented: the initial worker pins Luna
without fallback and permits at most ONE classification admission in total,
counting released-unsent admissions too, under the accounting writer transaction.
That bound survives restart and concurrent reservation. Reload checks refuse
running or staged classification writers; legacy journals without the table
remain compatible. These are source/test properties, not live Luna quality.

Additional completed isolated validation on October 6, 2026:

- The affected enrollment, worker, ledger/retrieval, publication, accounting
  and service suite passed 302 tests in 141.90 seconds before the final pilot
  admission limit, Luna-only request and reload-guard changes.
- After those changes, the worker, accounting, reload and paid-stop-fence suites
  passed 147 tests in 32.45 seconds. Provider responses remain synthetic; zero
  provider POSTs or shared schema mutations were made during this validation.
- Independent source review cleared the pilot-only changes, including the
  transport-edge revocation/revision gate and cancellation drain. Its scope
  does not validate a real Luna result or authorize full classification expansion.
- No full repository suite or merge was performed. The existing paid batch and
  raw data were retained; no batch was deleted, cancelled, repacked or resent.

## Scoped installed-runtime proof

Candidate `80fef98` was installed through the existing reload procedure on
October 6. The secret hook passed before commit. Reload validated seven encrypted
database preimages with zero in-flight jobs, then verified one authenticated
strict-archive service on port 42069 with the existing automatic capture and
remote flags preserved. Read-back observed PID 90700 as the sole listener,
authenticated history HTTP 200, and the new empty classification jobs/membership
schema. Empty work at this checkpoint is expected: discovery/inference is gated
behind the currently owned, unpassed paid batch.

The retained owner `e126f20c9fba4b47b1eee34402ce0b59` stayed sent/submitted with
25 windows in 12 logical requests and unchanged input/wire hashes before and
after reload. Its provider GET was in-progress, zero completed/failed, bill
pending. No result or pilot success is inferred from acceptance. Existing policy
remained generation 2, $5/day and $50/month; anonymous accounting returned 401,
authenticated accounting 200 with no-store, and the public page contained no
token. Local settled run cost remained $0.859029 with one unresolved admission;
this is not a reconciled provider total. Independent results review cleared this
scoped installation, not full-goal completion.

At read-back enrollment was complete (2,457 queued, 924 existing, 848 excluded
out of 4,229 source versions). Enrollment is NOT interpreted coverage; remaining
windows/overall percentage are not established by those counts. Backlog drain
was inactive, so ordinary cadence applies. The existing credential worker was
alive but awaiting a local passphrase; no duplicate worker was started.

Required next proof: inspect the bounded real Luna pilot after the exact paid
owner passes, including semantic quality and total billed cost before expansion.
Neither these tests, this installation nor the pilot cap prove a resolved
historical ambiguity queue, complete hooks or full local installation.
