# Automatic new-capture processing cadence

Status: original local-only cadence implemented and activated; remote scheduling
revision below is implemented and focused-tested. Installed proof is tracked in
`docs/evidence/capture-window-recovery-2026-10-05.md`.

## October 5 remote scheduling revision

The original local-only design below predates managed remote capture consent and
accounting. Preserve one consumer and existing capacity/foreground controls.
When persistent remote capture opt-in and managed consent are active, permit
remote-only attempts during ongoing chat after the existing cooldown; permit
CPU planning through the same opt-in. Quiet-ready processing retains the normal
local/resource-gated lane. Remote generation, source privacy, accounting and
budget checks remain immediately before dispatch. Do not extend or reset the
separate temporary catch-up deadline.

This removes an observed contradiction: accepted chats should postpone GPU use,
not block already approved remote network work. No second inference consumer,
unbounded queue, faster attempt interval or temporary-retention batch policy is
introduced. Five added cases cover busy planning, foreground priority, remote
opt-in, cooldown and quiet gating; two behavior cases failed against the prior
source. The affected scheduler/automatic-service/transport suite passed 53 tests.

The historical original decision is retained below, not asserted as the current
remote policy or an unresolved activation gate.

## October 5 local-opportunity fairness revision

Remote attempts update the shared minimum-attempt cooldown, but no longer reset
the local maximum-wait timer. Otherwise continuous remote traffic during active
chat can indefinitely postpone privacy-parked/local-only windows. A local-primary
path or permitted pre-send local fallback resets local max-wait. This records an
opportunity, including resource deferral, not a guarantee of inference or permission
to occupy a busy GPU. Remote-only refusals do not become local fallbacks.

No queue, lease, publication, spending, provider, opt-in or temporary-drain
authority changes. Startup still starts the full grace and local max-wait period;
foreground requests remain independent. The new behavioral regression failed
against the prior service timer reset. The affected cadence/automatic-service/
backlog-drain/window-queue suite passed 111 checks, including continuous remote
activity, local-only jobs with remote opt-in, private fallback, GPU deferral and
unchanged cooldown. Installation and actual runtime evidence follow separately.

## Decision

Use the existing encrypted post-enable-watermark outbox and one shared analysis
consumer, with a separate CPU-only planner. Require both capture opt-in flags;
the default installation keeps them off. Accepted archived activity postpones
capture processing for five minutes. Every restart starts a new quiet grace.
Capture attempts have a configurable thirty-second minimum interval. Foreground
analysis is independent of that interval and prioritized at the atomic claim.

Planning first checks due foreground searches and reserved queue capacity. A
foreground search arriving during a long plan prevents subsequent capture-model
admission, rather than discarding and restarting completed CPU work. Supported
sources finish authenticated EOF with bounded memory; there is no source-size
cutoff or cadence timeout. Shutdown cancels and drains the preparation thread.

Capture remains local-only and never reads remote consent/credentials. Existing
resource-aware model selection, resident-model avoidance, output validation,
provisional cited publication, and keep-alive-zero behavior remain authoritative.
The planner does not invoke a model. One model consumer prevents competing active
job/cancellation trackers. Capture-only mode cannot dispatch search-analysis jobs.

## Alternatives and trade-offs

Running a second inference worker would race the existing active-job tracker and
compete for GPU resources. Running cold preparation inside the inference loop
would unnecessarily delay foreground interpretation. The selected split avoids
both without adding a second service or database schema.

Automatic capture fallback to ZDR is deferred until durable spending reservations
and accounting prevent concurrency overspend. Existing authorized on-demand and
search-analysis ZDR routes are separate. Historical backfill is also separate:
enabling capture never resets its immutable watermark or silently processes old
snapshots.

Cold preparation of very large sources can still be slow, and interrupted cold
parsing is not crash-resumable. Append-aware parser reuse and publication-chain
amortization remain scaling work; they are not prerequisites to safely enabling
the new-capture consumer. Continuous chat activity can postpone enrichment, while
CPU capture and authenticated transcript retrieval remain available. Attempts
that defer for GPU contention and those that reuse an old ACK also consume the
cadence interval, a conservative initial policy rather than a GPU-use claim.

## Evidence scope

Regression tests cover startup/activity timing, queue saturation, foreground
priority before and after preparation, search-lane exclusion, single-consumer
lifecycle, invalid enabled configuration leaving no tasks, revocation, and timer
loops reaching an authenticated publication ACK with an isolated model stub.
That stub proves wiring, not local model quality or a live installed deployment.
Real local activation must verify actual hook capture, eventual publication,
authenticated agent reads, and model release without affecting other services.
