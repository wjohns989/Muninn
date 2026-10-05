# Historical Luna batch: separate retention and recovery boundary

Status: encrypted recovery/transport contract implemented and tested; dispatch,
budget escrow and capture ownership not connected or activated.

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

## Account eligibility observation (October 5)

Read-only API checks with the existing dedicated key returned 466 public models,
including 73 `:batch` variants, but only 283 models and no batch variants from
the authenticated `/models/user` catalog. Luna Pro appeared in both catalogs;
Luna Pro batch appeared only in the public one. The authenticated ZDR endpoint
inventory listed Luna Pro on Azure and no batch endpoints. A direct authenticated
GET for the Luna Pro batch model returned 404; that alone is not a submission
eligibility test. No batch was submitted, no private payload was sent, and no
account or managed consent setting was changed.

OpenRouter documents `/models/user` as filtered by provider preferences, privacy
settings and guardrails. This is evidence of a catalog-level eligibility concern,
not proof of which account setting caused it. `variant` and `include_variants`
are not documented query parameters for that endpoint; their identical responses
do not provide an independent eligibility test. Confirm batch eligibility and
the pinned provider's non-training terms through a separate scoped retention
policy before implementing or enabling private-data dispatch. Do not disable
normal ZDR protection globally to make this historical exception work.

At this observation the key's daily usage was $0.486495557, monthly usage was
$0.537961537, and its enforced limit was $5/day. These totals include other
dedicated-key probes and are not a reconciled historical-run-only cost or a
forecast for the full backlog. A 50% token-price discount does not prove faster
completion or the cost per accepted, cited result.

Catalog contract: [user-filtered models](https://openrouter.ai/docs/api/api-reference/models/list-models-filtered-by-user-provider-preferences-privacy-settings-and-guardrails).

## Eligibility correction and implementation checkpoint (October 5)

After W fixed the account guardrail, a fresh authenticated `/models/user` check
returned 453 models, including 73 batch variants and Luna Pro batch. The pinned
batch endpoint reported OpenAI (`openai`) and $0.05/M input, $0.25/M output. This
supersedes the catalog-level concern above; it does not prove a private batch
submission or accepted historical results. No batch POST was made.

`historical_batch.py` now supplies the separately encrypted archive-key outbox,
FULL-synchronous compare-and-swap submission marker, immutable source/item
bindings, ordered provider-pinned payload, exact unordered result reconciliation,
aggregate cost validation, exact cited-output validation, and durable terminal /
cleanup receipts. Unknown submissions cannot be retried through this API. Invalid
results and missing costs stay stored for recovery; HTTP success is not accepted
memory and never acknowledges capture coverage. The module exposes no HTTP or
service activation, so it cannot bypass the remaining gates.

Focused isolated tests cover restart, concurrent submit markers, tampering,
missing storage, failed/expired/cancelled batches, duplicate/missing/extra IDs,
wrong model, malformed schema/quotes/truncation, missing/BYOK cost, and cleanup
ordering. An in-memory mutation that allows unknown resubmission fails the
restart test. Existing synchronous code and live policy are unchanged.

Remaining before private dispatch: atomic exclusive capture ownership, durable
aggregate escrow shared with normal budgets (without reusing the synchronous
unknown semaphore), separately revocable retention consent, HTTP polling and
publication integration, automatic backup/restore inclusion, then the small
real-window pilot. The outbox's local preparation metadata is not itself consent.

OpenAI's [API privacy commitments](https://openai.com/enterprise-privacy/) exclude
training by default, unless sharing is explicitly opted into. Batch is still
temporary-storage processing, not ZDR or guaranteed anonymity. OpenRouter keeps
batch artifacts up to 30 days unless deleted sooner. Its deletion response can
report the OpenAI batch record as unsupported for deletion while removing input
and output files; retain and report that distinction, not a blanket upstream
deletion claim. Normal live requests must keep their explicit ZDR controls.
