# Source-screen cache contention repair

Outcome: an optional encrypted screen-cache write cannot turn a fully authenticated
privacy check into an unsent model failure merely because a source reader holds a
SQLite DELETE-journal lock. This does not change model routes or privacy policy.

Current local evidence: a no-model replay of an existing failed capture window
failed with `OperationalError` at `_store_screen_info` connection commit, before
remote admission. No source text, candidate, credential, or provider request was
printed or sent by the diagnostic. This identifies the observed storage failure;
it does not establish that every historical `model_unavailable` has that cause.

Decision: after the entire immutable source unit has been authenticated and
screened, cache-write SQLITE_BUSY/SQLITE_LOCKED errors preserve that computed
in-memory proof. The optional write has a 100 ms busy timeout. A fresh process
without a committed attestation must rescan. Source authentication, screening,
existing cache validation, non-contention storage errors, and durable memory/job
publication remain strict. General store timeouts and schemas are unchanged.

Regression evidence: two real DELETE-journal reader-lock cases failed before the
repair (safe and private units). Both now pass, including uncached revalidation
and successful encrypted cache persistence after the reader closes. Corrupt,
full, and I/O error cases still fail. The focused ledger, source-evidence, and
cited-transport suites passed: 105 tests. These are isolated regression checks,
not a claim that the historical backlog or ambiguity review is complete.

Independent design review found this bounded fallback preserves the privacy
decision; integration still requires review of the actual diff and an owned
service reload with existing encrypted preimages. Rollback is the previous code
candidate; no data or schema rollback is required.
