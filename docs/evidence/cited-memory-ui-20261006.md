# Cited-memory local interface: bounded consumer proof

Observed 2026-10-06, final live reader check at approximately 08:09 UTC.

## Delivered scope

The existing Encrypted History tab now has explicit cited-memory search,
record lookup, source following and eligible noncredential review browsing.
It reuses installed authenticated read-only API routes, without a backend
schema change, inference dispatch, filing, consent edit or new service.
Search returns at most ten records; review pages six. Review output is the
safe provisional/needs-user subset, not the total ambiguity queue.

Opaque memory/source/project references and recorded event-time basis remain
visible. Model interpretation/provisional status is not certified truth. The
reader checks exact source-memory identity and shows source context only when
explicitly available and within the requested 3,000-Unicode-character bound.
Withheld/invalid context is never replaced by the record's quote. Existing
redacted transcript paging follows the source without exposing raw originals.
Capabilities/cursors remain in session memory, never displayed or persisted.

Lock and new reads clear results, details, review cursors and transcript state.
Sequence/session/transcript guards reject old completions and old button clicks.
Busy controls bound UI reads; an observation timeout is not backend cancellation
or evidence that its CPU slot was released. No automatic retries were added.
Missing search-completeness fields display unknown, not a completion claim.

## Focused proof

- Four initial meaningful tests failed because the actual UI consumer was
  absent. A test import correction preceded this red baseline.
- All **40 affected UI tests passed in 4.95 seconds** across cited memory,
  control center, authentication, operating status and window status.
  The existing extracted-history harness was extended with the new globals;
  an asynchronous transcript fixture was corrected to match the actual helper.
  Tests cover literal DOM rendering, bounded text, withheld context, invalid
  identity, same-token session replacement, stale source completions/clicks,
  busy reads, private errors, review-cursor progression and missing completeness.
- Full extracted page JavaScript passed Node syntax validation. `git diff
  --check` passed. Existing Hugging Face environment deprecation warning is
  unchanged and unrelated; no environment/interpreter change was made.
- Live authenticated search, review, lookup and source readers each returned
  HTTP 200. Three actual public records and one source response passed the
  actual new frontend normalizer/bound checks locally. No transcript text,
  credential, token, capability, private result or error text was printed/saved.
- The served HTML bytes exactly match the current local file. An earlier probe
  overlapped an edit and was invalidated; the subsequent stable check passed.
- Actual localhost browser reload displayed only the authentication dialog.
  The main content's inert attribute was present, cited result/detail text was
  empty, and the next-review control was not visible. No token was injected.
- Independent native source review identified withheld-context/invalidation
  rules before implementation, then flagged a missing-completeness false claim.
  Both were addressed with focused tests. Final review evidence is reported in
  the task; this record does not claim the reviewer reran the parent tests.

## Preserved operational state and remaining scope

One existing Muninn process/listener on port 42069, PID 50096, remains healthy:
health 200, anonymous protected request 401, authenticated protected request
200, strict encrypted archive ready. No restart, model call, batch POST,
deletion, cancellation, resubmission or queue mutation was performed by this
work. The current service still reports 28 windows in 20 provider requests,
awaiting provider. That status is not completed interpretation/publication.

The existing credential worker was confirmed active (one worker, no duplicate),
and its nonsecret progress record still reports `awaiting_passphrase`. It was
not relaunched and no passphrase was requested through chat or environment.

Signed-in manual browser workflow/layout QA remains unrun. Generic review
decisions/grouping, credential triage completion, full historical backlog,
run-aware cost accounting and the broader installation acceptance remain
separate outstanding goal requirements. This component proof does not certify
the whole installation, historical ambiguity resolution, full suite or merge.
