# Authenticated no-content completion for admitted transcript windows

2026-10-06. Separate deterministic completion from model interpretation.

## Evidence and decision

A read-only local audit authenticated all 212 failed capture-lane
`insufficient_context` inputs. Every original window contained exactly one
whitespace character; total substantive characters were zero. Six separate
`outcome_unknown` jobs had remote dispatch recorded and are not repair candidates.

Retaining blank fragments as permanent failures prevents their admitted source
plans from completing. Resending them wastes inference; inventing an Ollama or
OpenRouter result would falsely claim model processing. Repartitioning future
plans alone would not repair already-admitted windows and would invalidate
retained identities. Keep the original plans and use a distinct `no_context`
terminal, with an encrypted versioned proof in the existing result field.

The proof binds the authenticated source receipt, admitted target, exact original
sealed plan attempt/ordinal, sealed EOF count and descriptor. Reopening the
**original** window must prove whitespace-only text. A blank private projection
is not evidence that its underlying original is blank. The proof is nonexpiring;
provider and model remain null, with no proposals or publication/model result.

## Persistence and operational boundaries

The proof and inherited source completion counter commit in one SQLite
transaction. Running jobs require their unexpired lease; old failed
`insufficient_context` jobs require exact attempt and encrypted-target preimage.
Reject dispatched, uncertain, cancelled, batch-blocked, staged, published or
retained-result work. Repeated completion validates the old proof but cannot ACK
twice. Reads, backup verification and source status authenticate the separate
proof rather than interpreting it as a model result or reuse receipt.

Automatic CPU-only repair is bounded to eight old failures per planner turn.
Its gate preserves foreground search priority but does not require a free model
queue slot. Cancellation drains the writer thread. Ordinary nonempty source
claims follow unchanged model, budget, screening, consent and paid-checkpoint
rules. Read-only plan access neither builds a replacement nor performs recovery.
Dashboard counts explicitly distinguish no-content windows from model successes.

## Proof scope

Independent design review chose this separate terminal. Actual-diff review
identified the missing sealed-plan check; the corrected path and regression test
now reject a damaged original plan without changing job/source counters.
Initial regression checks failed 17 times on the source lacking this lifecycle.
The intermediate affected run passed 128 tests in 79.81 seconds. After adding the
plan safeguard, read-only plan reader and shutdown drain, focused no-context,
actual dashboard JavaScript and cited-plan tests passed 45 checks in 24.23
seconds; automatic service and batch activation passed 49 checks in 41.47 seconds.
The scopes overlap and are not an additive unique-test count.
Final cancellation-drain correction: the 28 no-context tests passed in 18.96
seconds, including a controlled writer that must finish before the cancelled
consumer can terminate. Independent final source review found no remaining
blocker; installation still requires the existing runtime/preimage fences.

Coverage includes empty/nonempty refusal, stale inputs/leases, paid-item denial,
immutable/sent preservation, rollback, corrupted proof/plan, actual portable
backup and restore, capacity-independent automatic repair and foreground yielding.
Full pytest, whole backlog, full ambiguity resolution and signed-in manual UI QA
are not established by these checks. Local installation and actual legacy repair
still require independent final clearance, existing service/preimage fences and
current runtime read-back.
