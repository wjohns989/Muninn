# Local Luna streaming diagnostic, 2026-10-08

User requested a streaming comparison. Exactly one synthetic model POST was
sent; no transcript, vault credential or passphrase was in the request. Existing
batches were neither cancelled, deleted nor resent. The shared service PID
18936 was not restarted.

## Observed result, not a streaming success

Local diagnostic `8b3e49b954354b6f878a9a5593e26fe9` received HTTP 404 in
0.069 seconds, zero content chunks, no final usage. The receipt is encrypted in
the existing managed accounting database. Its charge remains unknown, not zero;
ordinary admissions continue to block on unresolved rows. No automatic retry.
The non-200 error body was not retained by the first implementation, so the
specific provider rejection cause cannot be proven retrospectively.

The attempted body mistakenly pinned OpenAI (the temporary-retention batch
host) while requiring ZDR. Existing local Luna settings identify Azure as its
ZDR host; OpenRouter's current provider table also distinguishes OpenAI retention
from Azure ZDR. This is a plausible rejection cause, not proof from the 404.
The future fixed probe body now pins Azure with ZDR/data_collection=deny and no
fallback. Its price guard selects Azure catalog prices. Non-200 bodies are now
bounded before decoding and encrypted, never echoed to the operator. The already
consumed diagnostic cannot be resent by the CLI.

During preparation, main outbox `c31739ec3e5c447ea09d61a94b967ee4` became
terminal_saved: provider completed 60/60, failed 0, reported aggregate cost
$0.03602725, 220580 prompt and 99993 completion tokens. Its managed admission
was settled. This is provider completion/billing proof, not proof that every
reply passed local citations/schema checks and publication.

Tiny batch diagnostic `ae796bc8563044e7b3a26e60fa649965` remained in_progress
0/2 at its last local observation (1630.4 seconds). GET-only observer tool
session 41533 remains responsible for it; monitoring must not POST or resubmit.

## Scope and focused evidence

Operator-only diagnostics admit at most one batch and one stream per parent.
Migration takes a verified private policy database preimage before changing
fences, then binds encrypted ID/parent/owner/generation/kind to each admission.
Portable recovery validates old and new fence definitions and disables consent.
Normal reserve still checks ALL unresolved holds. A completed-parent exception
requires its exact settled generation and the already accepted same-parent tiny
batch diagnostic; unrelated holds remain blocking. There is no backlog bypass.

Streaming collector distinguishes headers, comments, content, usage and DONE,
pins SSE model/generation identity, bounds actual response bytes at 16 KiB and
retains partial timeout/cancellation receipts. Hard process death can still leave
an unknown admission without partial bytes; it never triggers automatic retry.
Only valid numeric usage with is_byok=false can settle billing, even when output
is invalid. Diagnostic totals are subsets of overall managed cost, not additions.

Focused validation: 19 streaming/batch tests passed in 8.29 seconds after final
route changes. Prior affected accounting interval checks passed (63 passes and
one corrected test-exception expectation); unchanged evidence is reused.
Independent source/design review resolved kind bindings, narrow admission,
per-kind costs, BYOK, separate header/SSE IDs and model consistency before live
migration/POST. Independent final review cleared future route/error retention
but explicitly did not certify the failed attempt or backlog publication.

The initial preflight sent no model request because the main batch had just
finished; the subsequently reviewed completed-parent predicate enabled the one
actual diagnostic. No provider policy, model fallback, queue, backup tree or
service lifecycle change was performed. Full local setup remains unfinished.
