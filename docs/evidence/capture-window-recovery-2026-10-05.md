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

## Installed revision and real outcomes

Candidate `8ecc280` was loaded through the same verified-owner procedure with
six validated database preimages and preserved local/remote flags. Authenticated
strict history returned 200, anonymous protected access remained 401, anonymous
HTML contained no token, and one listener remained. The temporary drain stayed
expired; no deadline or budget change was made.

While the quiet timer still had approximately 253 seconds remaining, metadata
inspection found three Luna jobs succeeded since that reload. This demonstrates
real remote processing during ongoing chat, not a synthetic model stub or an
assertion based only on enabled flags. Source coverage remained incomplete:
103 resolved and 3,622 pending versions at that observation, including subsequent
new captures. Enrollment completion is still not interpretation completion.

The bounded capacity watcher verified the actual service process, observed one
free slot, and invoked the reviewed recovery command. A fresh encrypted journal
preimage passed integrity checking. Exactly one former local failure was queued;
the remaining selected items stayed untouched at the capacity limit. Comparing
the preimage with the live journal found that item subsequently parked with
`source_not_remote_safe`, `remote_dispatched=0`. That is a real privacy refusal,
not a successful remote extraction or a reason to weaken the guard. The three
previous uncertain remote outcomes remained unchanged.

The fixed-run ledger held 591 settled response admissions totaling $0.528908,
including the separate GPT-OSS probe. Excluding that probe, backlog ledger cost
was $0.528786. Dedicated-key monthly usage minus the run baseline and the probe
was $0.52851704; the difference was $0.00026896. These are two measured accounting
views, not an assertion of exact reconciliation. Daily key usage was $0.479699357
under the unchanged $5/day limit. No batch submission occurred.

All GitHub workflows for `8ecc280` passed, including the full locked-dependency
suite, privacy check, clean imports, benchmark dry run and transport replay.
Existing untracked user files were preserved. These scoped proofs do not assert
that every historical ambiguity or the complete installation goal is finished.
