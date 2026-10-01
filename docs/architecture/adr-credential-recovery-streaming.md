# Streaming credential recovery and safe publication

## Status

Accepted for the credential component only. Full-installation recovery remains
unproven; this change does not access or migrate a live vault.

## Context and decision

The old backup materialized credentials and ambiguity rows in lists. Restore
had a 1 GiB physical database limit, used a raw database copy, and reopened the
store after publishing it. A raw copy of a changing SQLite database is not a
consistent snapshot; publication followed by a failed reopen is an ambiguous
operator result. Linux rename could replace a raced empty destination.

Pin a read transaction under the existing thread/process locks. Authenticate
source records one at a time; use SQLite online backup, then compare streamed
schema and every persistent table against that pinned source. The fixed catalog
includes sentinel, credentials, receipts, both audits, ambiguity queue and
SQLite sequence state. Unknown persistent tables fail closed until supported.
Authenticate copied records as well; no table-sized record collection is kept.

Restore retains header/link/WAL/passphrase checks without a database size cap.
Use online SQLite backup into private staging, close and flush it, then validate
the staged store. Rebase the already validated object before atomic no-replace
publication, with no fallible reopen after success. Failure preserves private
incomplete staging and never replaces another destination.

## Alternatives and limits

Keeping the byte cap limits some resource exposure but rejects legitimate large
vaults and does not establish database consistency. Streaming plus SQLite
snapshots addresses those defects; available disk and completion time remain
real constraints. This does not claim constant runtime or measured GiB speed.
The reused publication primitive supports Windows and Linux, failing closed
where atomic no-replace publication is unavailable.

A future combined recovery must preserve spending and unresolved admissions
and restore remote consent disabled. Copying an enabled policy can revive
consent; two independently active restored accounting roots cannot enforce one
shared local spend floor. Neither concern is solved by this vault component.

## Proof

Isolated encrypted fixtures cover bounded iteration, size-gate sensitivity,
full-table omissions and their targeted old-defect mutation, unknown schemas,
SQLite rather than raw restore copy, destination races, and no post-publication
reopen. Existing portable recovery, corruption, ambiguity and cross-process
writer-lock tests remain required. Tests do not stand in for W's live recovery
drill or a full-installation backup.
