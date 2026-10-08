# Credential-review visibility — October 8, 2026

## Outcome and scope

The checked-out dashboard now has a conspicuous cross-tab alert for a verified
local hidden-input wait, terminal failure, exited worker, or unverified status.
It also reports waiting for admission, backup validation and review counters.
Refresh is read-only; it never starts a worker, collects a passphrase, opens a
credential vault, retries inference, or changes policy.

This is component and isolated rendered proof, **not installed/live-worker
notification proof**. The existing service PID 18936 was preserved without a
restart or reload. Its authenticated new-status endpoint is not yet available.
No credential-triage retry was performed. The old private progress record's
failure cannot retrospectively establish the failure stage or a wrong password.

## Identity and privacy boundary

The new main-token, loopback-only `/credentials/triage/status` GET endpoint uses
process identity and owner-only operational progress, not `CredentialStore`.
Generic API and dedicated reveal tokens cannot read it. Responses, including
rejections, are no-store.

Format-2 progress adds the worker PID, process creation time, record timestamp
and versioned SHA-256 workspace correlation. The fixed-order path tuple is
vault/archive/policy/repository/interpreter, with Windows case normalization.
This is correlation, not cryptographic authentication; owner-only permissions
remain mandatory. Neither paths nor process arguments leave the reader.

A live prompt requires exactly one repository/interpreter-matched worker,
unambiguous exact vault/archive/policy arguments, a progress path within the
configured namespace, matching PID/creation time/workspace correlation, and a
second process-identity check. Missing, inaccessible or contradictory evidence
returns unknown with input-needed false. An exited worker cannot produce an
active prompt, and a legacy waiting record with no worker is idle.

Reads reject linked/reparse ancestors, nonregular/hardlinked or nonprivate
files, duplicate JSON fields, incomplete latest lines and oversized records.
Inventory is bounded to 64 entries, each tail to 64 KiB and latest record to
16 KiB. File identity/size/time are checked before/after reading. The reader
does not fall back to an older password wait after a malformed newer record.

## Validation

- 130 focused Windows tests passed in 25.89 seconds across the paid-fence,
  operational reader, authenticated endpoint, existing triage workflow,
  dashboard session/control-center and new alert tests.
- Replacing only the numeric predicate in an isolated test process with its
  pre-fix form reproduced two oversized-integer failures (six other selected
  checks passed). The fixed predicate passed both regressions in the normal
  suite. This mutation did not edit disk source or access runtime stores.
- A real append during a bounded read exercised file-race rejection; actual
  hardlinks, PID reuse, foreign interpreter, wrong/missing/duplicate runtime
  arguments, malformed/truncated tails and terminal correlation are covered.
- Independent native review found one oversized-JSON-integer overflow; it was
  corrected and re-reviewed CLEAR. Authorization, privacy, late-response lock
  handling and final actual diff were independently examined.
- An isolated Edge browser rendered the exact checked-out HTML/CSS at 1280 and
  390 pixels. Both displayed input/failure alerts and cleared the alert on
  locking, with no horizontal document overflow, JavaScript error, mutating
  request or external request. All responses were intercepted synthetic
  fixtures, not a real worker/service. Parent inspected rendered screenshots.
  The first screenshot caught the existing modal fade; the final probe waits
  for the modal to become hidden, without an arbitrary sleep.

## Current operational observation

Read-only runtime inspection at epoch 1791487485.6880736 found one canonical
service, healthy/authenticated, and no credential-review worker. The new direct
reader returned previous-run-failed, input-needed false, worker-count zero.

Backlog owner `87ef9ec0a9a44363b8ca1e5a0fc1fb60` remains accepted/in-progress:
9 windows, 9 provider requests, 0 completed/0 failed. Batch age was 2657.54
seconds, last successful poll 1791487454.859564, deadline 1791571228, polling
errors zero. Health was awaiting-progress, based on provider creation and zero
reported outcomes; internal provider activity remains unknown. No replacement,
parallel POST, cancellation or observer-side settlement was performed.

Previous parent `c31739ec3e5c447ea09d61a94b967ee4` was rechecked: 60 requests,
60 citation/schema-resolved windows, 60 authenticated publication receipts,
60 matching published stages, zero unresolved, aggregate actual cost
$0.03602725. Whole authorized-run successes remain 2435, including 23 explicitly
readmitted older lane-1 jobs; reuse 10; pending 128; privacy parked 6668;
uncertain 3; no-context 111. Claims 3692, repeat claims 76 are not API calls.
Accepted backlog submissions 41; logical requests 851; terminal HTTP successes
842 and nine pending. These HTTP results are not publication counts.

All capture-lane jobs separately total 11377: 3478 succeeded, 501 reused, 128
pending, 7050 retry/privacy parked, 8 uncertain and 212 no-context. They must not
inflate the authorized-run totals. Capture continued to 7410 retained snapshots;
6197 source versions remained pending planning. Source-version coverage and
windows are different denominators; total remaining windows stays unknown.

Fixed run cutoff remains 1791154542.000905. Settled managed total is $1.218825,
unchanged from the previous observation: $1.208825 response ledger plus the
already approved one-time $0.01 operator adjustment. That adjustment is NOT
actual provider billing, and its grant has been consumed. Diagnostic subset
$0.010482 is included once and contributes zero backlog publications.
Provider dedicated-key usage was $0.00533505 today/$1.215841097 this month;
minus the original $0.00252758 monthly baseline gives $1.213313517. The response
ledger is $0.004488517 below that provider delta; including the operator amount
makes the ledger $0.005511483 above it. Billing attribution/delay remains
unreconciled; no new current-batch aggregate bill is known. Daily usage cannot
reconstruct earlier days, and month rollovers limit baseline reconciliation.

The existing ten-minute read-only monitor is active. Delivery of an actual
future input alert remains unverified until the reviewed endpoint is installed
and an independently authorized worker actually reaches that stage.
