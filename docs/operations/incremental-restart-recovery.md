# Incremental encrypted restart recovery

One canonical live runtime remains at `.muninn_runtime`; worktrees contain code
only. The source-evidence database dominates accumulated restart preimages.

`scripts.compact_restart_recovery` packs only the closed
`source-evidence/projections.sqlite3` copy in older restart recovery folders.
The newest four actual full database copies stay full, plus the exact new
restart preimage when it is explicitly excluded. Already-compacted folders do
not consume that floor, even if their names sort newest after clock changes.
Full slots must pass the existing private, regular-file and independent/unlinked
path checks; an invalid slot refuses selection before any retirement.
Every other file, batch database, full
backup, archive blob and unfinished forensic copy is excluded. No chunk garbage
collection or snapshot expiration is implemented.

After a verified owned reload, the reload helper now launches one bounded
compaction in a separate hidden process. It excludes that exact new restart
snapshot (including clock rollback), inherits no service/provider secrets, and
records private progress beside the new preimage. Maintenance failure cannot
turn a verified service restart into a failure. This keeps routine restart-copy
growth bounded after the initial backlog is compacted; full runtime backups are
still standalone and are not automatically thinned or expired.

The pool stores independently authenticated encrypted 1 MiB chunks once.
Each original restart folder retains its own authenticated encrypted manifest;
the pool retains an identical central manifest as well, so it can recover the
packed database without the original runtime folder.
Original database bytes are retired only after reconstructing the complete
database, checking its SHA-256, SQLite integrity, and the unchanged original.
Interrupted work leaves an original or a complete recoverable pool reference.

Compacted restart folders are **not self-contained backups**. Preserve the pool,
including its key anchor, chunks and central snapshot manifests. Four recent full preimages and separately
validated full backups remain independent recovery paths. This is byte-preserving
database recovery, not proof that historical application versions can run today.

Run bounded compaction from the canonical repository:

```powershell
& 'C:\Users\user\miniconda3\python.exe' -B -m scripts.compact_restart_recovery compact `
  --archive-root 'C:\Users\user\muninn_mcp\.muninn_runtime\history_secure_archive' `
  --pool-root 'C:\Users\user\muninn_backups\restart-recovery-pool-v1' `
  --keep-full 4 --limit 1 --retire
```

Restore a selected compacted DB to a new directory; the recovery passphrase is
prompted locally, never sent to an agent:

```powershell
& 'C:\Users\user\miniconda3\python.exe' -B -m scripts.compact_restart_recovery restore `
  --pool-root 'C:\Users\user\muninn_backups\restart-recovery-pool-v1' `
  --snapshot-id '<original restart folder name>' `
  --destination '<new empty recovery destination>'
```

Do not copy a recovered DB into a running service. Restore other retained
preimage files using the normal operator recovery procedure. Do not remove
original full history backups merely because newer archives exist: they may
contain unique legacy memory stores or policy recovery files.

The initial migration predates central manifests. After that process finishes,
`backfill --archive-root <root> --pool-root <pool>` authenticates each retained
unique chunk and durably copies the existing encrypted manifests centrally.
It rewrites neither database bytes nor chunks. Do not overlap backfill with an
older compactor that does not yet know the central-manifest protocol.
