# Incremental storage for verified history backup bundles

## Status

Accepted for an isolated pilot. Real-bundle migration, retirement and scheduled
rotation are not enabled by this change.

## Decision

Use checksum-pinned Restic 0.19.1 around Muninn's existing consistent application
backup protocol. Do not extend the one-database restart pool into a custom
whole-tree backup/garbage-collection engine.

The public write API requires the same authenticated archive identity AND key as
its anchor, then calls `SecureHistoryArchive.backup_to` before enrolling the new
closed bundle. Restic never receives live SQLite stores. Forced byte reads and
exit-zero-only admission refuse incomplete file snapshots. Restore selects an
exact snapshot ID and writes to a fresh local destination outside its repository,
with byte verification. Muninn's cold restore remains the application proof.

## Encryption and bootstrap

Derive the password from the archive key through a distinct HMAC domain and send
it only through stdin, never argv, environment, a password file or logs. The exact
binary's init/reopen/check/restore behavior passed an isolated test. Retain the
original passphrase-wrapped header in a private key anchor outside Restic's store.
Preserve BOTH anchor and repository for recovery: a repository cannot bootstrap
its own derived password. No new passphrase is needed.

UNC/device and mapped-network paths are rejected before filesystem/child effects.
No cloud backend is exposed. Output uses bounded 64 KiB reads, at most two summary
records and discarded raw diagnostics, not source-path logs.

## Trade-offs and unfinished work

Restic supplies established whole-tree deduplication/indexing/restore but adds a
pinned local binary and bootstrap anchor. Initial enrollment still reads data;
file-tool success alone is not application consistency. This pilot implements
no snapshot expiry, `forget`, `prune`, temporary-bundle retirement or schedule.
Original backups, credential vaults, batches and the running backup remain intact.
Local deduplication is not offline/off-device protection.

## Evidence

The native 128 KiB test reopened via the pipe/portable anchor, added zero data on
an unchanged second snapshot, checked repository data and restored exact bytes.
A complete synthetic managed-runtime bundle retained its batch DB SHA-256, then
passed Muninn cold restore, two publication receipts and disabled remote/batch
permissions. Fixture provider calls were fakes. Identity, local-drive, incomplete
exit and bounded-output tests supplement this proof, not live migration evidence.

Primary sources: [repository/password](https://restic.readthedocs.io/en/stable/030_preparing_a_new_repo.html),
[backup](https://restic.readthedocs.io/en/stable/040_backup.html),
[restore](https://restic.readthedocs.io/en/stable/050_restore.html).
