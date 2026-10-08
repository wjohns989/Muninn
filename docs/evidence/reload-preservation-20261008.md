# Reload preservation and exact candidate scope — October 8, 2026

## Changed dependency and observable property

An ordinary explicitly authorized service reload now preserves recovery
preimages and launches no maintenance worker. The separate destructive choice
`--retire-one-old-recovery-preimage` requires a restart with the tested revision
and is considered only after replacement ownership and health are verified.
The existing opt-in compactor bounds remain four full copies, at most one old
preimage, and exclusion of the newly created preimage. No retained batch is
deleted. No live reload or compaction was performed for this change.

The candidate check now covers the exact reload helper, its compactor and the
hot-served dashboard HTML/CSS, both in tracked differences and untracked source
inventory. Generated-cache exclusions remain intact; unrelated scripts are not
newly included. Dirty controls/UI are rejected before preimage preparation,
settings access, process stop or launch.

## Smallest checks and retained evidence

- Before the fix, two retirement regressions reproduced an absent explicit
  choice and default successful reload scheduling maintenance.
- Before the candidate-scope fix, seven checks reproduced missing paths,
  exempt untracked HTML/CSS, and dirty controls reaching a forbidden effect.
- The focused reload/paid-stop group passed 140 tests in 37.51 seconds.
- After the test-only recovery-fixture portability correction below, the
  combined recovery-pool/reload/paid-stop group passed 160 tests in 44.02
  seconds on Windows, including actual user-protected unattended unlock.
- Independent native examination of the actual source/test diff returned
  CLEAR for commit/push after the terminal focused result. Full Linux,
  installation, retirement, merge and deployment acceptance remain separate.
- All lifecycle matrix effects and retirement happen only in isolated fake
  installations and synthetic temporary encrypted recovery fixtures. These
  tests do not constitute live installation, retirement or worker authority.

## Linux gate correction, not a production unlock change

Exact-head CI run 37832318272 passed clean imports and the affected portable
group, then stopped with 2688 passed, 38 skipped and one failure in 700.90
seconds. The recovery-pool fixture omitted its existing synthetic passphrase
when reopening the copied anchor. Windows automatic unlock had hidden that
fixture assumption; Linux correctly refused it.

Only that fixture and its foreign-identity case now supply the existing
synthetic phrase. A red regression first observed the omitted phrase on
Windows. Added checks use real encrypted constructors, reject a wrong phrase
without changing the header, and separately prove actual Windows unattended
unlock and foreign-archive rejection. Production recovery/unlock code is
unchanged. This file joins the affected portable CI group; the full locked
suite remains required. At that checkpoint, a new exact-head Linux result was
still required; the terminal result follows below.

### Terminal exact-head gate and conditional reload review

Tests run 37834757988 completed successfully for
`20d630d436aca8dc53f6e15c07f2c0811f4cadde`. The affected portable group passed
179 tests, with 43 skipped, in 92.93 seconds. The locked full Linux suite
passed 3997 tests, with 101 skipped and two warnings, in 766.93 seconds.
Clean-install imports, privacy, incident replay and benchmark checks also
passed for that head. Skips do not constitute platform behavior proof; the
160 focused Windows tests remain the separate Windows evidence.

Independent examination of the unchanged candidate and conditional reload
proof plan returned CLEAR. This is not human execution permission. The
prepared action preserves capture/remote settings and permits replay of at
most one interrupted CPU capture; it contains no retirement, new activation,
new diagnostic or credential-worker flag. Fresh uncertainty in paid binding,
candidate, queue/lease, ownership or validated preimages must refuse stop.

Read-only preimage inventory found seven private, unlinked databases totaling
3,501,576,192 bytes and 417,836,204,032 bytes free on the destination volume.
No new preimage was copied and copy duration remains unknown. After permission
and execution, actual acceptance still requires one replacement listener,
strict/authenticated readiness, an authenticated no-store triage-status GET
that is no longer 404, no password request when no worker exists, unchanged
paid owner/request identities/input hashes and preserved queues/settings/copies.
Existing authorized worker progress may legitimately advance checkpoint state.
No such live reload or acceptance is claimed here.

## Read-only live observation

At epoch 1791488705.9375699, the sole canonical service remained PID 18936,
healthy and authenticated; no credential-review worker existed. Capture
continued to 7418 snapshots, with 6205 source versions pending planning.
Dashboard files are hot-read and match the checkout, while the new Python
credential-status endpoint is not installed (authenticated HTTP 404).

Owner 87ef9ec0a9a44363b8ca1e5a0fc1fb60 retained nine windows/nine provider
requests, provider in-progress, zero completed/failed. At age 3877.77 seconds
the service reported degraded-unknown based on provider creation and zero
outcomes, not confirmed provider failure. Last successful poll was
1791488638.284744, polling errors zero, deadline 1791571228. Internal provider
activity remains unknown. This warning never authorizes replacement or bypass.

Fixed run cutoff 1791154542.000905 is unchanged. Settled managed total remains
$1.218825: $1.208825 response ledger plus the consumed $0.01 operator adjustment,
which is not actual provider billing. Diagnostic subset $0.010482 is included
once and contributes zero backlog publications. Dedicated-key monthly usage
$1.215841097 less baseline $0.00252758 is $1.213313517; response ledger is
$0.004488517 below that delta, or $0.005511483 above it including the operator
adjustment. The current aggregate bill is unknown and attribution/delay is
unresolved. Daily totals cannot reconstruct earlier days; month rollovers
limit baseline comparisons. No actual savings claim is supported.
