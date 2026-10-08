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

Independent design and integration reviews returned CLEAR. Candidate `6db0e40`
was loaded into the single existing service on port 42069 after an idle window
and six validated encrypted database preimages; strict archive readiness and
automatic remote mode were verified. No active model/publication was terminated.
Rollback is the previous code candidate; no data or schema rollback is required.

A subsequent no-model replay of a real unsent failure passed the contested
cache-write path and returned `source_not_remote_safe`, preserving the privacy
denial rather than misreporting a model/storage failure. The installed bridge
also passed a real read-only project-context call and matched Codex, Claude
Code/Desktop, and Gemini client profiles (20 core tools). This is bridge/config
proof, not proof of every host hook event, full backlog success, or resolved
credential ambiguity. Existing OpenRouter budgets and retention policy were not
changed. Automatic backlog work remains remote-only.
