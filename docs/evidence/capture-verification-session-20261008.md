# Call-scoped capture recovery verification

## Decision and scope

Use one read-only cited-window/source store per `_verify_capture_window_jobs`
invocation. Authenticate the entire memory ledger once in a pinned SQLite read
snapshot and check each reused window's freshly computed expected references
against that snapshot. Close the reader at return, including error return.

This removes the repeated full-ledger replay for each reuse without changing
the encrypted format, publication counters, model authority or accounting.
Every source receipt, descriptor, parent, extraction, ACK and reuse seal is
still checked. Ordinary status and live reuse admission retain fresh standalone
verification. No proof is cached across invocations or archive roots.

Persistent caching and skipped source/citation checks were rejected. Read-only
constructors precede the pinned reader to avoid schema/writer initialization
inside it. Page-count and no-context validation remain unchanged in this fix.
This is not a claim that concurrent live-store snapshots are atomically aligned;
frozen backup verification remains the intended reusable proof boundary.

## Retained evidence

The new synthetic two-reuse test failed against the unchanged source:
`assert 2 == 1` full `MemoryLedger._snapshot` replays (5.52 seconds).
The regression requires one replay per invocation, fresh read-only stores on
the next invocation, rejection of a later reuse's damaged seal and rejection
of ledger ciphertext changed between invocations. Existing reuse, window
scheduling and recovery/tamper tests supplement that proof.

The running full frozen-backup verifier was not stopped, restarted or declared
complete. No private runtime data or provider inference is used by these tests.
Focused call-count improvement does not establish a wall-time speedup for the
full production-sized backup, or resume the billing-blocked paid backlog.

Focused verification: reuse, scheduling, recovery and the new session tests
passed together (114 tests, 154.81 seconds). The three session tests passed
again after adding explicit fresh/read-only-store assertions (7.46 seconds).
Independent review inspected the actual diff and tests and returned CLEAR for
this scope; it did not claim live full-backup completion. `git diff --check`
passed. No service restart, provider dispatch or live policy edit was performed.
