# Private cited-window ZDR route

This closes the synchronous admission/scheduling gap for mixed transcript units
without relaxing retained-batch admission or ordinary public-read policy.

- Explicit private selection requires capture-only, remote-only processing.
  The journal authenticates the batch head and refuses owned/sent checkpoints.
- One private opportunity occurs after a passed checkpoint; when no clean batch
  exists, private-only work can proceed after the anchored gathering deadline.
  The in-memory fairness hint is not an idempotency/authorization record.
- Durable journal leases, dispatch state, encrypted stages and the existing
  singleton managed admission prevent duplicate paid requests across restart.
  Current consent/generation is required; revocation does not invoke local models.
- Whole-unit authenticated redaction selects unchanged original-coordinate
  ranges. Excluded text is whitespace, never quoted evidence. Unsafe lines and
  dangling labels are excluded. Project/time provenance remains bound locally.
- The final exact request body still passes the existing privacy/ZDR/budget
  gates. Projection policy/ranges participate in model identity; raw-window reuse
  cannot silently become projected coverage.
- Results disclose partial visible-range interpretation. Original bounded text
  is used only locally to scrub output, never included in the provider view.

## Retained focused proof

- Red-first failures: missing adapter, unsupported private transport/claim flags,
  missing checkpoint fairness, and an unsuppressed guessed source value.
- 60 transport/projection/model-identity checks passed during integration.
- 87 affected queue/service/accounting/batch checks passed.
- After the output-scrub change: 15 projection checks passed.
- Final claim/recovery suite: 8 passed. The service/revocation checks also passed.
- The recovery fixture uses an actual isolated settled admission and encrypted
  publication stage. After restart, it recovers the identical stage, publishes
  exact original quotes and validates the durable ACK without redispatch.
- One real parked 3,000-character window yielded 2,733 visible characters in 41
  exact original ranges. Its projected final envelope passed screening; original
  batch admission still refused it. No provider request was made by this proof.
- Independent scoped security/integration review: CLEAR. New-file lint and
  diff whitespace checks passed. Existing secure_analysis lint findings were
  reproduced unchanged against HEAD; no unrelated cleanup or full-suite claim.

## Limits

This does not prove paid private-route model quality, resolve the whole backlog,
or admit mixed-unit candidates into ordinary public filing/search. Their existing
whole-unit withholding remains in force. Full-suite merge validation is unrun.
