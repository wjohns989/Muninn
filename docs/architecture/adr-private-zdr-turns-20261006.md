# Bounded private ZDR turns between retained checkpoints

Status: accepted after independent design and actual-diff review, 2026-10-06.
Focused service/claim/projection/gathering proof: 38 passed in 14.15 seconds.
Existing-service installation still requires its paid, idle and preimage fences.

## Context

The local backlog contains 4,740 privacy-parked windows at the October 6 sample.
The installed consumer gives one private ZDR opportunity after each exact
retained checkpoint passes. Clean batches may contain up to 128 windows, whereas
private interpretation remains a one-window, screened original-coordinate ZDR
projection. W authorized that route within $5/day and $50/month, with credential
values local, and prohibited local-model calls for current catch-up.

## Options and decision

Keep one opportunity: minimal code, but makes private work disproportionately
dependent on the clean-checkpoint cadence. Unbounded private drain: would risk
starving clean batching and leave no bounded turn. Larger private request packs:
would require a new projection/prompt/identity/recovery contract and quality
proof, unnecessary to close this scheduling gap.

Choose a static bounded **four-opportunity private turn** after a passed
checkpoint, one existing private claim per consumer iteration. Keep the existing
attempt cooldown; wait during it rather than starting the next unrelated clean
checkpoint prematurely. Stop the turn immediately when no eligible private job
exists. Then return to clean gathering. Reset the in-memory turn only when the
passed checkpoint identity changes; it is a fairness hint, not paid authority.

## Trade-offs and authority boundaries

This increases potential private work before the next clean batch, and may delay
that clean batch by the existing call time and cooldowns. Four opportunities are
not four successes or a measured fourfold improvement. It does not reduce each
window's token bill. Successful remote dispatch uses the existing five-second
cooldown; the ordinary 30-second interval and failure backoff remain unchanged.
Neither is a new guaranteed completion time.

No new schema/configuration/permission/provider/model/packing path is introduced.
Keep exact consent-generation, spending-admission, body-screening, ZDR provider,
original-range citation, publication and no-redispatch guards. Owned/sent batches
block all unrelated private inference. Revocation and auto-capture gates still
apply. Restart may reset the in-memory hint but never readmits dispatched or
uncertain jobs: the encrypted journal and ledger remain authoritative. Existing
no-owner/private-only fallback after gathering remains unchanged. Retained
batch-quota exhaustion is not private ZDR revocation.

## Acceptance and revisit

Focused synthetic/isolated tests must prove four bounded opportunities then clean
work, cooldown waiting without spending or early preparation, empty-lane
fallthrough, same/new checkpoint behavior, disabled automatic gates, owned/sent
denial, actual managed revocation, and encrypted projected-stage recovery without
redispatch. Reuse intact projection/privacy/paid-recovery evidence; independent
design and actual-diff reviews precede existing-service installation. Retain paid
identities/input hashes and preimages on reload. Do not infer full interpretation
or ambiguity resolution from scheduler tests.

Revisit only on observed private/clean progress, failed validations or admission
pressure. Large-context packing needs its own measured cost/quality case; this
decision does not authorize it or a retention downgrade.
