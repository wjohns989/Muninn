# Selective capture recovery and installed accounting repair

## Actual local change

The reviewed accounting repair in `f36a585` was installed via the existing
single-owner reload helper with candidate `a63745d`. The helper validated six
private database preimages while fencing the idle journal, then confirmed one
listener on the existing port, strict archive readiness, authentication, and
preserved automatic local/remote capture settings. No catch-up deadline,
provider, consent or budget was extended or replaced by that reload.

The affected accounting/transport/catch-up suite passed 53 tests locally. The
existing PR candidate's full locked-dependency CI and privacy checks were green;
these are not evidence that the historical backlog is complete.

## Recovery dependency

Read-only inspection found failed unsent local windows preventing whole-source
completion. The live preview selected four from 618 eligible windows: 60 invalid
local outputs, 387 quote-validation failures and 171 unavailable-model failures.
This is the entire current eligible queue, not just jobs created after the paid
run cutoff. Counts are a dated observation, not a permanent installation fact.

An explicit recovery transaction authenticates the existing target, mapping,
window and exact plan ordinal before re-admitting it under current consent. It
preserves the job ID, attempt count, creation time, source plan and ACK count,
and reseals the target/mapping/window together. It checks bounded runnable
capacity and stale attempt/preview fences. A source is never marked complete
merely because failed windows were requeued.

Cancelled, dispatched/uncertain, completed/reused, staged/publication, unknown,
insufficient-context and integrity-failed jobs remain untouched. No provider
request or credential read occurs in the recovery helper; existing automatic
workers may subsequently interpret explicitly recovered work under their
privacy, consent-generation and spending controls.

## Proof and remaining dependency

The first new test failed because recovery did not exist. Implementation tests
caught a nested reader observing a pre-transaction binding during window
resealing; passing the same writer transaction to the purpose validator repaired
that defect. The affected recovery/window/journal suite then passed 89 tests.
Additional review added existing-store preflight and an exact preview-target
fence. Results of those additional checks and live recovery are recorded below.

This recovery facility does not resolve a genuinely unknown remote reply, fix
unsupported context by declaring it empty, enable temporary-retention batch
transport, or prove comprehensive source coverage. Batch submission remains a
separate next dependency, not a feature activated by this change.

Final focused proof: 42 updated recovery tests passed, the portable prompted CLI
path passed its targeted check, and both final operator regression checks passed.
The independent review cleared the exact journal transaction and post-preimage
preview comparison for a bounded application. It did not perform the operation
or claim backlog completion.

The first bounded live recovery command reported `queue_full` and changed no
jobs or source counters; no unnecessary preimage was created. This exposed the
next scheduling dependency, rather than proving recovery completion. The
separate remote scheduling revision permits currently opted-in remote network
work during chat activity, preserving quiet GPU work, foreground priority and
all dispatch controls. Its affected suite passed 53 tests and independent
design/diff inspection cleared it. Installed verification follows separately.
