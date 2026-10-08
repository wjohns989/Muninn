# Recovery platform proof — 2026-10-08

## Changed dependency and preserved contract

PR candidate `31d79ea` passed privacy, clean imports, benchmark dry-run and replay,
but locked Linux pytest stopped at 757 passes and eight skips. The classification
recovery test invoked `SecureHistoryArchive.backup_to`, which intentionally
requires Windows user protection. Further direct callers had the same mismatch.
The authoritative failure was run `37826831586`, job `113481724491`.

Production backup, restore, encryption, admission policy and service code are
unchanged by this correction. Tests now distinguish two explicit backends:

- `windows_unattended` invokes the real existing backup method on Windows. Its
  original publication/accounting/ownership assertions and staging callback
  checks remain. Only this parameter is skipped on Linux.
- `portable_snapshot` constructs a temporary encrypted recovery input from real
  archive/store and journal copies, using the fixture's explicit synthetic
  passphrase. Managed fixtures use real encrypted `snapshot_into` and
  `verify_snapshot`, then actual `restore_from_backup`. It is not a production
  unattended Linux backup implementation or a DPAPI simulation.

The test builder rejects destinations outside its temporary boundary, linked or
existing paths, and destinations inside the source archive. It authenticates
the real phrase before copying. Accounting ciphertext is regenerated after
journal copying; stale encrypted accounting is not reused. Report counts come
from actual archive, journal/publication and batch verifiers, not invented
fixtures. Classification identity, unknown/operator admissions, repair ownership,
paid receipts, disabled restored policies and tamper failures remain asserted.

Portable backup-section tests give only the standalone verifier's constructor
alias their known synthetic phrase. The real archive decryptor and all actual
read-only verification, corruption rejection, writer prohibition and byte
fingerprints remain active. The Windows parameter uses the unmodified local
unattended constructor. These portable cases do not prove unattended Linux CLI
unlock. Production backup callback/unsupported-backup tests remain Windows-only;
unsupported *restore* also has an explicit all-platform rejection check.

## Retained focused results

On the canonical Windows interpreter:

- Classification jobs, memory placement, historical batch jobs/repairs and
  retained replies: 89 passed in 111.50 seconds.
- Paid recovery and credential ZDR receipts: 59 passed in 69.06 seconds.
- Backup-section verification: 28 passed in 42.22 seconds.
- Fixture boundary, actual wrong-phrase rejection, encrypted copy/source
  preservation and no-DPAPI portable unlock: five passed in 3.12 seconds.
- New unsupported managed-runtime restore cases: two passed in 5.22 seconds.
- Actual archive/publication/capture/search backup/restore subset: 17 passed,
  one expected platform skip, 54 deselected in 13.78 seconds. The skipped test
  rejects unattended backup specifically on non-Windows hosts; it is not a
  skipped Windows restore proof.

The builder boundary implementation was tightened during the first group run;
its copy, encrypted accounting and verifier behavior was unchanged. The final
builder-specific checks independently cover the tightened boundary. These are
focused results, not a whole-suite or cross-platform completion claim.

## Linux and broader gate

The existing locked Linux CI job now runs the nine affected files together
without fail-fast before its unchanged full-suite step. The existing dummy tray
backend, stack diagnostic timeout and 20-minute job bound remain. This collects
the known portability class in one candidate instead of a separate first failure
per push. It adds no provider, credential, service or test-security exception.

Independent review cleared the test architecture and bounded validation plan.
Actual Linux terminal results and full-suite acceptance remain separate gates;
timeouts or partial passes are incomplete proof. No merge, service reload,
credential retry, broader backup claim or overall installation completion is
authorized or established by this test-only integration.
