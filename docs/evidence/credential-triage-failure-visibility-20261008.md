# Credential triage failure visibility — 2026-10-08

## Actual operational evidence

The canonical localhost service remains healthy and authenticated. Encrypted
capture continues. Its retained previous checkpoint passed all 60 window jobs;
the next checkpoint is prepared with nine windows and nine provider requests.
Managed accounting has one unresolved streaming diagnostic admission, independent
of the credential worker. This blocks paid admission, not capture, and is not a
spending-limit exhaustion. No unknown charge was assumed to be zero or released.

The credential worker is terminal, not waiting for input. Its owner-private
progress record contains an input-needed stage followed by a failed state with
`VaultIntegrityError`, input-needed false, and pre-triage backup not validated.
There are zero matching live workers. The retained record does not distinguish
vault opening, unlock, encrypted-record authentication or backup failure. It does
not establish that the supplied passphrase was wrong. No password, credential
value, console buffer or private exception text was inspected.

Read-only structural checks of the current vault verified owner-private ACLs,
supported header structure, SQLite quick-check `ok`, DELETE journal mode, schema
version 2, exact credential columns and recovery table set, receipt schema,
discovery index, one sentinel row and absence of WAL. These checks do not unlock
the vault or authenticate the sentinel or encrypted records. The constructor was
not invoked on the live vault because it can perform compatibility migrations.

## Changed dependency and focused proof

The portable triage entry point now immediately reports input-needed false after
the hidden prompt returns, before opening the vault or validating a backup.
Fixed allowlisted stages distinguish pre/post-backup vault opening, backup
validation, entering review and final queue-status reading. Terminal output
retains `failure_stage`, meaning the last operation entered, not a diagnosis.
No free-text error detail, paths, passphrase material or credential values are
added to the owner-private progress record. Cryptography, credential storage,
backup validation, admission policy and dispatch behavior are unchanged.

Independent design review required a review-start stage before source-reader
construction, so an early review failure cannot inherit the previous backup
stage. The implementation includes that correction. Reporter failure still stops
the worker and never retries, recreates or repermissions a failed destination.

Nine changed/new expectations failed against the previous entry point. All 42
triage-wait tests then passed in 1.41 seconds, using isolated synthetic input and
stores. The combined diagnostic and triage tests passed 52 cases in 4.98 seconds.
The latter also exercises the expired-watcher fixture with its known synthetic
archive passphrase explicitly, avoiding a Windows-only DPAPI assumption on Linux
without changing the production unlock path or skipping its no-GET assertion.

## Remaining acceptance boundaries

No failed worker was relaunched, no password was recovered or reused, no model
was called, and no service restart, queue change or billing settlement occurred.
The new stages cannot reconstruct detail absent from the old terminal record.
Future operator-authorized execution is needed to observe real unlock/backup
success or a more specific failure stage. User-visible authenticated notification
of a live hidden-input requirement is still a separate UI acceptance gap; adding
these breadcrumbs alone does not close it. The full locked PR suite remains the
broader integration gate, not a claim made from these focused tests.
