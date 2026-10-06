# Luna historical-batch window packing

## Decision

Use Luna's context for up to ten consecutive screened windows per provider
request, while retaining the existing 3,000-character windows as the unit of
source proof, publication, coverage and repair. A batch still contains at most
128 windows and its actual POST remains bounded to 8 MiB. Do not enlarge the
capture window or rewrite an existing batch.

Group only adjacent selected items whose authenticated plan ordinals are
consecutive and whose snapshot and authenticated project scope match. Unknown
project scope, gaps, changes of source and singletons keep the existing wire
format. This supports source-page boundaries without mixing projects or
pretending already planned windows establish globally chronological coverage.

Each item keeps its original random window ID, job, descriptor and single-window
body. Additive encrypted `pack` metadata records version, root request ID, slot,
size, source-plan attempt and ordinal, scope reference and exact packed-body
digest. Packed wire input contains only public model-window fields and slot
aliases. Scope, job IDs and local plan metadata never enter that wire input.
Consent binds all stored items, including pack metadata. Reservation, dispatch
and result recovery reprove plan membership and source scope. No schema migration
or new credential, consent, provider or spending policy is needed.

Provider counts and result ownership refer to root requests. Each exact owned
reply is expanded back to per-window IDs before citation validation and durable
publication. Packed inference has its own domain-separated model identity,
including body digest, slot and original descriptor; it is not presented as the
old single-window interpretation contract. A reply frame is parsed once per
cohort within a resolution call, not persisted as a plaintext cache.

Missing/duplicate slots and bad citations leave only affected windows unresolved.
Unknown slots, excess rows or malformed outer framing conservatively invalidate
that cohort's outputs. Successful siblings remain published. Existing linked,
failed-only repairs send unpacked original window bodies, with the existing two
round limit. One aggregate bill is settled once for each original or repair batch.
The next unrelated checkpoint remains blocked until exact ownership, known bill,
citations and durable publication acknowledgments pass. Retained batches are not
deleted or cancelled. If the packed envelope exceeds its byte bound, keep the
same selected items in the original per-window shape.

## Alternatives and limits

Larger individual windows would change citation geometry, stored stages and
recovery identities. Parallel checkpoints would weaken the requested serial
boundary. Both are unnecessary for this optimization.

The completion allowance remains 2,048 tokens per window, capped at 20,480 for
ten-window requests. Packing can reduce request/prompt overhead, not the source
text that must be read. It does not guarantee a faster provider batch turnaround,
which still has a 24-hour completion window, or a measured reduction in charges.

Frozen fixtures prove protocol, privacy boundaries, exact citations, root/window
counts, portable recovery, sibling retention and aggregate accounting. Actual
provider acceptance, inference quality, latency and cost of the new contract
require the next authorized serial checkpoint. Do not claim those properties from
synthetic tests. Existing submitted batches and repairs keep their original
request IDs, bodies and accounting bindings.
