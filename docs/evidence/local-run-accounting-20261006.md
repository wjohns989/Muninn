# Local run-charge accounting, 2026-10-06

## Delivered boundary

Installed candidate `739e23cc74f0165514115434a86c4f1535d7c6f0` adds a
main-authenticated, actual-loopback, no-store GET interval reader and explicit
Run charges controls in Backlog & costs. No schema, permission, threshold,
provider-dispatch or queue policy changes were made. Reading the local ledger
does not query a provider. Comparing provider usage is a separate explicit GET.

The interval includes **all managed admissions started since the supplied UTC
cutoff**, not exclusively backlog work. Settled response and operator charges
are distinct. Each batch aggregate admission is included once, including repair
admissions; it is never multiplied by window/request count. Reserved and unknown
charges remain unresolved, not zero. Older unresolved admissions are also shown.
Uninitialized stores and legacy batch attribution remain explicitly unknown.
Existing accounting rounds response charges upward per admission to micro-USD.

Start timestamps retain up to six fractional digits. Provider monthly subtraction
requires an explicit user-supplied baseline and the same UTC month across both
server sample bounds. Month resets invalidate a full-run comparison. Same-key
identity and baseline timing are not independently proved by the form. Numerical
agreement is not invoice reconciliation; delayed and earlier activity can affect
provider usage. Input edits, new reads and session locks invalidate stale output;
lock clears the form. No browser persistence, automatic provider polling, model
launch, cancel, delete or resend is added.

## Focused proof and review

- Initial frontend tests failed because the actual consumer was absent.
- A stale rejected request overwrote locked status before the catch guard;
  the targeted regression failed before the fix and passed afterward.
- Independent actual-diff review identified invocation-before-snapshot timing
  and unsanitized framework validation errors. Both targeted tests failed
  before fixes. Snapshot time is now sampled after the read transaction is
  established; missing/nonnumeric cutoffs receive a static no-store 422.
- After those fixes, the changed core/API/provider-envelope/actual-JS tests
  passed: **38 tests in 5.58 seconds**, including whole inline JS parsing without
  execution. Before those two fixes, the broader focused accounting/control-center
  group passed 58 tests in 7.03 seconds; intact remote-accounting/control-center
  evidence is reused, not counted again as new tests.
- Other affected dashboard files passed **37 tests in 4.42 seconds**.
- Tests use isolated temporary stores and synthetic tokens/responses; no live
  model calls or real credential values are fixtures. Read-only file-byte checks,
  rollover, invalid/corrupt rows, legacy attribution and session/form races are
  covered. `git diff --check` passed. Independent review is CLEAR after focused
  proof; no whole-suite or whole-installation claim is made.

## Installed runtime proof

The existing owned reload procedure validated seven private database preimages,
preserved settings, and recovered one permitted in-flight CPU capture. The
retired PID was 83388; the sole post-reload listener/process was 51624 on 42069.
Health/authenticated protected GETs returned 200, anonymous protected GETs 401,
and strict encrypted history was ready. Automatic capture, remote analysis and
remote-only routing remained enabled. No user setting was persisted during reload.

The new live GET returned 200 with no-store; anonymous access returned 401.
Served HTML matched the candidate source exactly and contained no main bearer.
The actual browser remained locked; the new read control was disabled. Positive
signed-in manual browser interaction remains unrun, not substituted by API proof.

Live managed permission remained generation 2, enabled, $5/day and $50/month.
Retained-batch consent remained generation 2, enabled, quota 10,000 with 9,979
remaining at the sample. This quota is not permission to skip serial checkpoints.

The already-owned retained checkpoint was inspected before and after reload:
33 windows, 18 provider requests, provider identity
`batch-1791278985-LekmzqQB5r5zaQ9NPMuw`. Both samples were submitted/sent. Exact
items digest `1c4abfa251a56cd9ad3f3fbd3228cac2fa31435fbc50956cf68dd3f1f135a5dc`
and wire digest `3d4d57db0ddc6f0901212c32c10d7cb155ffb423f1dbe272bef818097e31bb8a`
were unchanged. No cancellation, deletion, repacking or resend was performed.

## Cost sample, not a reconciled invoice

Cutoff: epoch **1791154542.000905**, UTC **2026-10-04T22:55:42.000905Z**.
Local sample epoch 1791279614.8507946: **$0.840392** settled across 729 admissions,
all response-settled; one unsent released, zero reserved and one unknown.
The batch-owned subset was 21 settled aggregate admissions, **$0.223675** already
included in that total. One unresolved admission existed across the ledger.

Explicit provider GET sample epochs 1791279684.4312208–1791279684.6287782:
daily usage $0.20105323; monthly usage $0.860159037. Subtracting the retained
monthly baseline $0.00252758 gives **$0.857631457**, **$0.017239457** above the
local settled sample. Both samples are in October but are not simultaneous.
The difference is unassigned; unresolved/delayed billing is not claimed settled.
These admission counts are not successful job/window counts.

## Full-goal gaps remain

Fresh post-reload all-capture-lane counts: 2,703 succeeded windows, 501 reused,
128 pending, 212 failed, six outcome-unknown and 4,740 privacy-parked retries,
total 8,290. Source-version coverage is separate: 6,088 pending snapshot versions
against archive generation 3,239 / 7,056 snapshots. Enrollment completion refers
to an older queued generation, not current interpretation completion; no overall
remaining-window percentage is inferred.

Exactly one existing triage process was verified live, with its private progress
record still at `awaiting_passphrase` and `passphrase_needed=true`. No duplicate
worker or passphrase transport was created. Historical/generic ambiguity,
private-window routing, local credential-use/audit proof, natural client cycles,
current portable recovery, and signed-in manual UI QA remain required. Local
model tuning stays deferred under W's remote-first decision. The full goal stays
active. No merge or full pytest was performed.
