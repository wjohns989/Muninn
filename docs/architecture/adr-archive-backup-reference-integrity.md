# Archive backup publication and reference integrity

## Status

Accepted for the archive command; not a full-installation backup design.

## Context

Archive, source-evidence, window, ledger and journal databases have independent
writers. The archive lock freezes transcript publication but does not fence
ledger publication and its later journal ACK. Sequential online snapshots can
therefore contain an authentic ACK whose memory was not copied. Independent
database authentication alone accepts that torn set. Failed copies previously
also occupied a normal destination before validation finished.

## Decision

Copy ciphertext to a new owner-private sibling named `.incomplete-<random-id>`.
Use SQLite's online backup API for the journal as well as encrypted sidecars.
Authenticate raw snapshots and component structures, then recompute every
ordinary publication receipt's references from its sealed extraction and cited
window. Require exact equality and durable membership in the copied ledger.
Authenticate one pinned ledger chain, including candidate citations, before
indexed bounded reference lookups in that same snapshot. Create dependent
stores before opening the pinned reader; do not hold a journal writer lock
through citation/ledger validation. Missing ordinary ACK receipts also fail.
Unacknowledged ledger candidates are allowed for publication recovery.

Verify private staging access before final publication, then atomically rename
without replacing any destination. Windows uses `os.rename`; Linux uses libc
`renameat2` with `RENAME_NOREPLACE`. Unsupported operating systems, libc symbols,
kernels and filesystems fail closed. Restore rebases its already-authenticated
object before publication, so no second unlock or filesystem validation can
fail after the final name appears. Failed stages remain private for recovery.

## Alternatives and limits

A global barrier across every writer could establish a single cross-store cut,
but adds new lock ordering to ingestion, credential scans, planning, publication
and accounting. This bounded change instead refuses a torn reference graph.
It does not claim an atomic cross-store instant, anti-rollback protection,
capture of separate credential-vault or policy/accounting directories, or a
portable complete-machine backup. Existing historical archives remain readable;
the command does not change encryption formats or erase source data.

Revisit for a complete-installation coordinator and measured live backup cost.
Do not activate or overwrite a live installation as part of a restore.

## Proof

Isolated encrypted stores reproduce the old defect: a valid empty ledger and a
valid acknowledged journal both pass component checks, yet restore must reject
them together. Tests also exercise publication between ledger and journal
snapshots, exact-reference binding, deleted receipts, recoverable unACKed
candidates, committed copying during an uncommitted SQLite writer, one ledger
walk, copy/ACL failure staging, and a destination created after the absence
check. None of these tests runs inference or accesses the real vault.

## Primary API references

- [Python rename semantics](https://docs.python.org/3/library/os.html#os.rename)
- [Linux no-replace rename semantics](https://man7.org/linux/man-pages/man2/rename.2.html)
