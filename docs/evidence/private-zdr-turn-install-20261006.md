# Bounded private ZDR turn: local installation evidence

Date: 2026-10-06. Installed source revision:
`481aa28760ea8048f3351fdf18a7444333cf8b87`.

## Changed dependency and proof

The existing capture consumer now offers at most four private ZDR claim
opportunities after a retained checkpoint passes, then returns to clean batching.
Cooldown polls do not consume opportunities or prematurely prepare clean work.
This is a fairness change, not larger-context packing or a new egress permission.

Independent design and actual-diff reviews were CLEAR. Old-source regression
proof: three failed and seven passed. Focused service, claim, projection and
gathering validation: 38 passed in 14.15 seconds. `git diff --check` passed.
No full-suite, completed-backlog, model-quality or measured speedup claim follows.

## Installation and current read-back

The existing scoped reload completed with seven validated private database
preimages, zero active inference jobs, and preserved automatic capture settings.
Its verification reported one service on port 42069 and strict archive ready.
Post-reload read-only preflight confirmed:

- One Muninn process/listener, PID 77140, using the existing Miniconda interpreter.
- Health 200; protected endpoint 401 anonymously and 200 authenticated.
- Public root did not contain the main authentication token.
- Strict archive ready; automatic analysis, remote and remote-only flags enabled.
- One existing credential-triage worker; no second worker was launched.

Authenticated policy read-back confirmed unchanged $5/day and $50/month managed
limits and retention generation 2 with 9,979 checkpoints remaining. Run-accounting
GET returned 200 with no-store; anonymous access returned 401. At sample epoch
1791280724.8238654, settled managed cost since the fixed run start was $0.840392
including $0.223675 in 21 settled batch admissions, with one unresolved admission.
No fresh provider-key bill was sampled in this installation check; this is not
a reconciled total or a backlog cost estimate.

The pending paid checkpoint `4b84b05c96044cea9a40bf120061f007` retained its exact
provider identity `batch-1791278985-LekmzqQB5r5zaQ9NPMuw`, 33 windows and 18
provider requests. Its state remained submitted/awaiting-provider. Both encrypted
item and reproduced wire-input digests matched the pre-reload baseline:

```
items: 1c4abfa251a56cd9ad3f3fbd3228cac2fa31435fbc50956cf68dd3f1f135a5dc
wire:  3d4d57db0ddc6f0901212c32c10d7cb155ffb423f1dbe272bef818097e31bb8a
```

No batch cancellation, deletion, repacking or resubmission occurred. No manual
model dispatch, local inference, new budget, privacy policy or drain activation
was performed. The four-opportunity turn cannot begin while this checkpoint is
owned/sent; it waits for existing identity, billing, citations and publication
checks to pass.

## Remaining interpretation work

The preflight sampled 8,290 capture-lane window jobs: 2,703 succeeded, 501 reused,
128 pending, 4,740 retry/privacy-parked, 212 failed and six outcome-unknown.
These are windows, not provider requests or source-version counts. The separate
pending source-version count was 6,090; total unplanned windows remain unknown.

A metadata-only read of failed/uncertain capture jobs found all 212 failed jobs
had `insufficient_context`, no remote dispatch and no publication started. All
six outcome-unknown jobs had remote dispatch recorded. Neither class was reset
or resent. Insufficient evidence needs context recovery, not repeated unchanged
model calls; unknown dispatched outcomes need existing paid recovery proof.

The backlog drain was inactive; capture cadence and provider batch waiting remain
in force. This installation is operational component proof, not full local-goal
completion. Larger request packs need separate citation/schema/quality and
cost-per-published-window evidence; existing packing is capped at ten adjacent
windows from the same authenticated source, plan and scope.
