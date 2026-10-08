# Legacy credential-context backup verification with an original witness

## Decision

Accept an explicit, original-witness verification path for matching legacy
credential-context contents. Keep generic missing-table verification fail-closed.
Never alter the original database or the retained backup to make it pass.

## Context

The preserved installation backup and its live original both lack the optional
`context_remote_calls` table. The normal writable context constructor adds that
table on open; the read-only section verifier deliberately bypasses writer
initialization and therefore failed when the current verifier queried it.
Code age alone does not prove legacy provenance: the backup's source candidate
already supported the optional table.

## Alternatives and scope

Silently skipping a missing table could conceal lost recovery evidence.
Migrating the retained backup would discard its byte-preservation property.
Repeating the full archive copy or unrelated successful checks would not prove
the absent table's provenance.

Instead, `--legacy-context-source` requires a distinct, private original database
with the same authenticated vault identity. Pinned read snapshots must match the
complete SQLite schema and type-bound logical rows, and both must contain exactly
the known legacy tables. A checked in-memory SQLite copy then receives only the
empty optional table and runs the existing context/page/review authentication.
Equal-but-corrupted copies still fail cryptographic validation.

The report retains `remote_receipts= schema_absent_unknown` and
`source_witness= matching_original_contents`. This proves recovery of the
matching original contents, not absence of historical remote dispatch, complete
provider billing, or a newly inferred paid receipt. Any publication or restore
evidence must carry this scope forward. A source witness is not a generic bypass
or authority to alter the original, settle accounting, or send a model request.

## Evidence

Independent native design review cleared these requirements before implementation.
Six correctly constructed red-first tests failed at the missing witness method.
Initial green checks passed 29 section/context tests in 19.68 seconds. Expanded
section checks passed 13 tests in 16.09 seconds, covering original/backup byte
preservation, writer-constructor prohibition, same-file and other-vault rejection,
schema/row mismatch, equal context/review corruption, and modern-schema rejection.
Independent actual-diff review subsequently found two gaps: inherited readers
left an explicit transaction open on the reused memory connection, and equal
vault IDs did not by themselves bind equal keys. A two-attempt fixture reproduced
the first failure; a wrong-key fixture then reproduced the second. Ending only
memory read transactions and comparing keys with `hmac.compare_digest` corrected
both. Focused final checks passed 32 tests in 23.49 seconds, and independent
changed-issue review cleared the real read-only witness run.

That run finished successfully against the preserved installation copy:
11 authenticated context snapshots and 500 contexts, with the explicit original
witness and `schema_absent_unknown` receipt limitation. The original/backup
encrypted logical inventory also matched 76 cached reviews. Backup publication,
incremental enrollment, exact restore, and off-device recovery remain separately
verified requirements; this result alone claims none of them.
