# Historical Luna batch: separate retention and recovery boundary

Status: implementation contract; not activated or represented as installed.

## Context and decision

W approved temporary, non-training storage for historical batch work. Normal
live memory inference must retain its ZDR controls. Public endpoint metadata on
October 5 lists Luna Pro batch through **OpenAI**, not the current synchronous
Azure route, at $0.05/M input and $0.25/M output. This is a per-token discount,
not measured total-backlog savings or provider quality equivalence.

Use the existing authenticated, privacy-screened, source-linked windows and
publication path. Implement a separate encrypted batch outbox and consent state;
do not strip or reinterpret `:batch` as a synchronous request. Avoid a parallel
memory engine, whole-transcript prompts, or inferred completion from enrollment.

## Required lifecycle

Prepare bounded immutable items with opaque request IDs, exact source/window
binding and explicit pinned provider. Verify account/provider eligibility and
non-training policy before sending any private content. Reserve a verified
upper charge against the existing daily/monthly budget before submission; an
unresolved batch reservation must survive UTC rollover and service restart.

Durably mark submission before HTTP. Persist returned batch identity and poll
with backoff. An uncertain submission is never blindly resubmitted. Reconcile
using trusted provider identity and the exact item binding, not approximate
timestamps or result array order. Do not allow the synchronous consumer to
interpret the same batch-owned jobs.

Authenticate and encrypt complete returned results and aggregate billing before
publication or deletion. Bind every item by its unique request ID; reject missing,
extra or duplicate IDs. Apply the existing schema and exact citation checks,
then idempotently publish only accepted item stages. Account for failures and
retries, with aggregate cost counted once. Preserve failed/uncertain items without
marking their sources complete or retrying successful items.

Request deletion only after durable verified local storage, retain the returned
deletion outcome, and retry cleanup independently from inference/publication.
Do not claim upstream deletion or anonymity beyond provider evidence. Preserve
ordinary live ZDR requests and local/private windows throughout.

## Smallest proof before scaling

Frozen isolated checks: pre/post-submit crash, reordered/duplicate/missing results,
expired/cancelled/failed batches, partial failures, lost cost, revoked consent,
budget rollover, interrupted publication, deletion failure and restored outbox.
Then a small actual approved-window pilot: accepted cited results, provider cost,
latency, encrypted recovery and deletion receipt. Scale only from that measured
cost per accepted result; do not credit theoretical reuse or a model-price table
as observed savings.

## Alternatives

Synchronous Luna remains available while this lifecycle is built; selective
local-failure recovery and remote scheduling separation address its current
completion blockers. A model dropdown change alone lacks asynchronous ownership,
accounting and recovery. Reusing synchronous unknown-cost admission as a 24-hour
batch semaphore would unnecessarily block all normal remote use, so batch budget
escrow must be designed explicitly rather than weakening that safety boundary.

Primary contract: [OpenRouter batch documentation](https://openrouter.ai/docs/batch-quickstart).
Endpoint inventory: [Luna Pro batch endpoints](https://openrouter.ai/api/v1/models/openai/gpt-6-luna-pro:batch/endpoints).
