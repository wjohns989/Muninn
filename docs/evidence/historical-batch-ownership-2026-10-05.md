# Historical batch ownership and recovery boundary

## Implemented, not activated

`CaptureJournal` now binds a prepared encrypted batch to exact authenticated
capture jobs, source descriptors, prompt/custom-ID digest, and consent generation.
At most one unresolved batch owns work. Owned members remain excluded from the
ordinary consumer across restarts and timeouts, without holding two-minute
worker leases for a provider batch's 24-hour lifetime. Normal cancellation cannot
detach a member from that ownership.

The encrypted head and predecessor chain authenticate phases and sequences.
Missing records, changed bindings, and replaying an older passed head while a
newer owner remains fail closed. A shared accounting reservation binds its batch
owner at creation; an unrelated same-generation unknown admission cannot be
borrowed. Only that owner's settled response permits fresh local publication
leases. Successful members cannot acquire another result claim.

Passing requires every exact member's durable publication receipt, settled
admission, validated stored cited reply, and extraction equality. Transport
success alone cannot pass. Backup verification rejects individually authentic
but mismatched outbox/publication snapshots. Inputs, replies and ownership
history are retained; this implementation adds no deletion operation.

Archive backup/restore includes both outbox marker and SQLite-online ciphertext
snapshot. A portable passphrase restore of an unsent owned batch preserves its
ownership and excludes ordinary redispatch. This is not proof of a complete
portable installation restore: the separately stored remote accounting policy,
credential vault, hooks and settings still require the integrated recovery flow.

## Focused retained proof

- Red-first ownership test failed because durable batch ownership was absent.
- Seven affected suites passed: 203 tests in 116.85 seconds (before the subsequent
  scoped-verifier/sequence and admission-association corrections).
- Fresh ownership plus secure-analysis check: 18 tests in 16.47 seconds.
- Fresh ownership plus accounting checks after association correction:
  42 tests in 20.48 seconds.
- Latest ownership and archive tests, including authentic earlier-outbox and
  changed-reply restore counterexamples: 33 tests in 26.48 seconds.
- Automatic capture/backlog-drain/cadence compatibility: 80 tests in 20.70 seconds.
- Independent actual-source review cleared the changed association and restore
  consistency issues after the previously flagged cancellation, sealed-phase,
  and verification checks were implemented. No live provider dispatch was run.

## Exact next dependencies

Connect the provider submit/poll loop to this boundary and the encrypted outbox;
persist separately revocable temporary-nontraining consent; install chronological
source/window selection with a checkpoint barrier for the complete authorized
backlog (not just whichever windows happen to be queued); include shared
accounting in integrated recovery proof; then run the bounded actual-source pilot
and retain provider cost/validated-publication evidence. Private credential
contexts need the separately authorized sanitized ZDR lane, not batch egress.

Current API semantics were checked against the official
[OpenRouter batch documentation](https://openrouter.ai/docs/batch-quickstart).
No batch request, credential value egress, local inference, service restart,
budget increase or deadline extension was performed for this source-only slice.
