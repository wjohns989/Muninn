# Capture recovery fairness

## Outcome and observed defects

Older, intact, proved-unsent local failures must remain reachable by the
authorized historical interpreter. New CPU planning currently fills every freed
slot before batch recovery can take it. Private failed windows are skipped by
batch screening but cannot be claimed by the existing parked-private ZDR lane.
Both defects have isolated red tests; neither requires a new provider or daemon.

## Decision

Reserve up to 32 slots from **new planning only**, capped by structurally eligible
unresolved failures in the authenticated historical selection, and only while
the existing managed 128-slot batch opt-in is live. Check at preparation and
transaction commit. Recovery retains the original queue capacity; foreground
search keeps its eight reserved slots. No queue size or schema changes.

Extend the existing fenced retry transition for private-only recovery. Inside
that transition, authenticate the exact target, ordinal and descriptor, then
freshly reopen/screen the actual source and serialized request. A safe request
must remain unchanged. Recheck current remote consent generation and historical
selection; an unfinished owned batch prevents this transition. Atomically move
only eligible unsent failures to the existing nonrunnable
`retry/source_not_remote_safe` state, rebinding the existing consent seals.
Reuse an authenticated same-archive source reader, not a cached refusal boolean.
Actual ZDR dispatch retains its separate privacy projection, consent and budget
guards. Credential values never enter either provider route.

## Alternatives and scope

- Manual retry polling cannot solve a full-queue race reliably.
- Raising queue limits does not establish fairness and consumes more capacity.
- Moving private data into retained batches violates the existing boundary.
- Bigger model packs do not repair unreachable jobs or provider turnaround.

Do not alter, delete, cancel, repack or resend existing owned paid batches.
Blank-context failures and dispatched unknown outcomes remain excluded.
No source acknowledgement or model success is fabricated by queue transitions.

## Independent review and proof plan

Native review cleared the planning reservation design and flagged an unchecked
private-only switch. The revised transition verifies refusal itself and checks
current consent after screening. Independent actual-diff review cleared both
fixes and the integration proof scope. Review also required that reload preserve
the exact existing `historical-batches.db` store, previously omitted by the
`.sqlite3` selection. That bounded selection fix keeps the existing paid stop
locks and private-copy/integrity validation; its omission has a red regression.

Proof: red-first starvation/private-handoff tests; safe-source counterexample;
revocation during screening; full-queue private handoff; immutable/sent-row,
descriptor and rollback guards; affected batch/claim/foreground tests. Local
reload requires reviewed exact source, preimages and retained batch identity
verification. Passing component tests is not historical completion.
