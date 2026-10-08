# Non-streaming backlog resumption — 2026-10-08

## Exact authority and recovery

W explicitly approved the proposed one-time operator adjustment at the retained
$0.01 ceiling for failed streaming diagnostic
`8b3e49b954354b6f878a9a5593e26fe9`, then resuming the existing non-streaming
backlog worker under unchanged $5/day and $50/month limits. This is not a
confirmed provider charge, another streaming attempt, or publication authority.

Before the action, the encrypted diagnostic binding authenticated generation 2,
parent `c31739ec3e5c447ea09d61a94b967ee4`, streaming kind, HTTP 404, no content,
no usage, and a $0.01 ceiling. It was the sole unresolved admission. The existing
SQLite backup helper retained an owner-only preimage under the writer fence;
its integrity check, encrypted diagnostic row and full logical SQL digest
matched the source. Independent review cleared the exact action.

The existing accounting CLI was invoked once for that exact admission at
$0.01 with explicit outcome confirmation. Read-back authenticated settled
admission, resolution `operator`, and operator cost $0.01. The encrypted failed
response remains intact: provider billing unknown, output invalid, zero backlog
publications. No queue reset, manual batch POST, policy change, service restart,
cancellation or deletion was performed.

## Live resumption, not inferred completion

The same healthy service process subsequently submitted its already-owned
checkpoint `87ef9ec0a9a44363b8ca1e5a0fc1fb60`: **9 windows, 9 provider requests**.
At the 18:43 UTC observation it was `awaiting_provider`, provider `in_progress`,
completed 0/9, failed 0/9. Last successful GET poll was approximately 18:42:33
UTC, age 153 seconds, deadline October 9 at 18:40:28 UTC, zero consecutive poll
errors. Progress basis was provider creation with zero reported outcomes;
internal provider activity remained unknown. Acceptance and successful polling
prove operational resumption, not inference, completion or new publication.

The previous 60-window/60-request checkpoint passed: 60 validated citation/schema
results matched 60 authenticated durable publication receipts, no unresolved
items, actual aggregate provider cost $0.03602725. At the resumed-run count
sample there were 2,435 successful authorized-run windows, including 23 older
explicit readmissions, plus 10 reused results. Other states were 128 pending,
6,668 privacy-parked retries, 3 uncertain outcomes, 111 no-context and zero
terminal failures. Claims totaled 3,692 with 76 repeat claims, not API calls.
Accepted backlog batch submissions were 41, with 851 logical provider requests:
842 terminal HTTP-successful requests and 9 pending. Synthetic diagnostics
contribute no backlog success. These figures do not establish total remaining
windows or overall coverage; unplanned eligible source versions remain.

## Accounting and visibility

Fixed run cutoff remains `1791154542.000905`. Settled managed run accounting
was $1.218825, up $0.01: $1.208825 response-settled micro-dollar ledger costs
and $0.01 operator adjustment. Diagnostic bookkeeping was a $0.010482 subset,
included once. The new pending batch had no actual bill yet; its managed unknown
admission is the normal pre-send fence, not a terminal billing failure.

Dedicated-key actual usage remained daily $0.0004819 and monthly $1.210987947.
After monthly provider baseline $0.00252758, provider run usage was
$1.208460367. The response-ledger difference was $0.000364633, unresolved;
including the operator adjustment the total difference was $0.010364633.
Rounding and delayed billing are limitations, not proven explanations or
measured savings. Daily usage cannot reconstruct earlier run days; monthly
rollover cannot reconstruct the entire fixed-cutoff run from one current bucket.

The existing ten-minute heartbeat was found paused. Its exact configuration
preimage was preserved and verified, then the same monitor was resumed through
the application tool, retaining its cadence, cutoff, privacy and read-only
constraints. Its updated prompt distinguishes the operator adjustment from
actual provider billing and records the synthetic two-request batch as already
terminal, rather than repeatedly expecting its historical watcher handle.
Read-back confirmed ACTIVE, ten-minute cadence and no-repeat-settlement wording.

## Read-only reporting proof

Retained streaming status now reads the encrypted binding and admission in one
pinned snapshot, checks provider identity/BYOK/cost before reporting an actual
bill, and reports operator settlement separately from unknown provider billing.
Run accounting separates integer-micro-dollar response and operator totals.
This reporting change itself never settles an admission or invokes inference.

The five affected accounting/streaming/dashboard test files plus the existing
remote-accounting tests passed **73 tests in 15.44 seconds** on the canonical
Windows interpreter. Earlier preimage checks demonstrated missing status and
cost-split expectations. Independent review cleared the actual reporting diff.
This is focused proof, not full-suite success, service reload, merge or overall
Muninn installation completion. Credential triage is separate: no live worker
or active password request was observed, and no credential retry was launched.
