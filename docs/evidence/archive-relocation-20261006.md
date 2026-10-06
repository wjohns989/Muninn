# Native transcript relocation closure

## Changed dependency and observable outcome

Codex/Claude native UUID identity already survived moves in the capture journal,
but archive version lists were keyed solely by physical path. A move produced a
second snapshot lineage, losing append-parent reuse and duplicating interpretation.
The isolated preimage test reproduced three failures and one pass before edits.

The candidate retains immutable canonical anchors and old entries/receipts. New
physical locators become direct aliases in an encrypted format-2 manifest only
after full digest or exact latest-prefix proof. New appended entries record
`observed_source` as capture-location metadata, not project or event-time authority.
Legacy generations stay readable; format-1 writers reject new generations rather
than silently ignoring aliases. The encryption envelope, blob AAD, old citation
identities and paid request bindings are unchanged.

## Consuming-path proof

Independent review found that retained originals also need latest continuity:
otherwise stale A can become another snapshot after B grows. A regression now
checks no generation/blob/enrollment change after rejection, and valid return to
A with current bytes still succeeds.

The real scanner regression then reproduced stale-A starvation across a durable
250-item checkpoint. Discovery now groups all physical candidates before native
identity checkpointing. Multi-locator selection uses one authenticated manifest,
bounded streaming digest/prefix verification, and no size/mtime-only ranking.
Historical authenticated rewrites are not new viable branches. Divergent viable
branches require review and enqueue nothing. The writer repeats current-latest
and path/open-handle checks; selection is not capture authority. Equal-fingerprint
moves also refresh the journal's encrypted locator instead of coalescing with a
missing or stale physical path. No persistence schema or model dispatch was added.

Proof cases include copy/move plus append, stale aliases/originals, conflicting
legacy UUID lineages, malformed aliases, pinned old generations, alias-only
idempotence and no new enrollment, actual agent fetch capabilities, multi-MiB
prefixes, handle replacement, portable restore with stable receipts/project/time,
real scan interruption/resume, divergent-group no enqueue, and a corrupt final
encrypted trailer after an unknown short-prefix hit.

## Retained checks and limits

Initial archive/prefix/base checks: 43 passed. Expanded archive/capture/enrollment/
blind-index/provenance checks before the scanner delta: 149 passed in 72.77 seconds.
After the retained-anchor guard: 61 archive tests passed in 25.51 seconds.
Source-evidence/append/cited-window/memory-ledger checks: 107 passed in 49.30 seconds;
their project/time/citation dependencies are unchanged by scanner locator selection.

The first scanner/journal run passed 63 checks and exposed an environmental test
fixture issue: the strict-default test used an incomplete `__new__` object while
inheriting W's explicit auto-capture flags. That fixture now explicitly disables
those flags for its no-opt-in claim. No installed or user environment was changed.
Final combined affected validation passed 182 tests (one warning) in 86.79 seconds.
It includes all the discriminating review cases above. Results review is pending
at this record; no full repository suite or merge is claimed.

No live format-2 generation, service reload, provider POST, local inference, batch
deletion/cancellation/resubmission or credential value access was performed for
this engineering proof. The existing paid owner remains sent with 25 windows in
12 requests and unchanged input/wire hashes. Settled local run cost was $0.859029,
with one unresolved admission; this is not an exact reconciled provider total.

Live activation requires compatible readers. The existing credential worker
81412 was independently verified alive/awaiting passphrase, before either backup
started, but imported the old archive reader. Its exact owned-handle replacement
plan is independently cleared; preserve logs, terminate only while that stage is
still true, confirm terminal state before one replacement, use fresh destinations,
and prompt only in a local interactive terminal. Shared service reload must retain
its exact ownership, encrypted preimages, capture flags and paid checkpoint fence.
No historical duplicate lineages are silently merged or removed by this change.
