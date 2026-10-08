# Preserved full-copy section verification

The original copied stage remains the same 10,711 files / 40,296,599,306 bytes.
Its source-evidence database retains device 8075710914016319318, file identity
1125899909008291, size 3266338816 and mtime_ns 1791418063639603600. No second
full archive copy was made, and no retained file was deleted or migrated.

## Reused and completed proof

- Earlier archive/accounting/batch section results preceded the read-only
  journal-adapter failure; their unchanged frozen inputs retain those proofs.
- Complete source-evidence verification retained in
  `local-resume-20261007.md`: 3,228 snapshots, 6,169,223 authenticated units and
  6,742,249 physical fragments. The exact database identity still matches.
- Corrected journal verification completed: 3,553 captures, 3,421 publication
  receipts and 45 classifications, in 725.1 seconds. Its process then failed
  at the independently diagnosed legacy credential-context table gap; it is
  terminal, not a continuing wait.
- A separate selected ledger/window verifier exited zero: 6,526 events and
  candidates, zero decisions; 3,143 window snapshots and 283,817 windows.
  Ledger verification took 10.6 seconds and window verification 125.8 seconds.
- The reviewed original-witness credential-context verifier exited zero:
  11 snapshots / 500 contexts. Its original and frozen ciphertext inventories
  matched, including 76 cached reviews. Neither disk copy was altered.

## Exact limitation and remaining gates

Credential-context proof is recovery of `matching_original_contents`.
`remote_receipts=schema_absent_unknown` remains explicit: an absent optional
table on both copies does not prove no historical remote dispatch or complete
provider billing. The generic missing-table verifier still rejects unexplained
absence. This scoped proof does not release the unrelated streaming billing hold.

These checks do not themselves publish the stage or create/validate a Restic
snapshot. Those require the separate no-replace publication, repository data
check, exact-ID restore, byte-identical tree checks and read-only restored-store
proof. A same-disk restore is not off-device recovery. Full installation
acceptance, live interpretation completion and upstream promotion remain open.

## Reviewed publication and recovery drill

Independent integration review cleared no-replace publication of the exact
unchanged stage. Publication succeeded to
`C:/Users/wjohn/muninn_backups/history-runtime-full-20261007-classification-10e0ae5`:
10,711 files, 40,296,599,306 bytes, 7,256 snapshot versions, zero deleted files.
The scope and unknown historical remote-receipt limitation were preserved.

The parent-owned migration helper now gates on these combined proofs rather
than the old failed verifier's exit code and carries the credential limitation
in its final receipt. The actual encrypted repository/exact-restore drill was
started in new `history-recovery-v1` and
`history-recovery-v1-restore-drill-20261008` destinations. At this checkpoint the
owned process was alive and hashing the validated bundle; no snapshot, repository
data-check, exact-restore completion, or off-device proof is claimed yet.
The full published original remains intact, and this drill has no prune/delete
operation or provider transport.
