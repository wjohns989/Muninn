# Local control-center first boundary

## Delivered scope

The full UI roadmap is committed in
`docs/plans/2026-10-06-local-control-center-design.md` (design commit `e6d4890`).
The first implementation preserves existing ingestion, ordinary search,
encrypted transcript search/paging, credential metadata and model-policy forms.
It adds an operational Overview and Backlog & costs, separate latest/all-version
enrollment, actual window/provider-request units, on-demand dedicated-key usage,
and explicit unreported review/recovery/accounting readiness. No completion
percentage or run cost is fabricated. Unknown storage counts no longer start at
zero. The decorative continuously animated graph is replaced by work status.

Session lock removes private DOM/form/result values, capabilities and pollers,
invalidates async sequences and handshakes, and makes the background application
inert. A request captures its token and session epoch; stale completions are
discarded before and after JSON parsing. Only a current-session unauthorized
response can relock. Same-token reauthentication is still a new epoch. New local
status reads are bounded to 15 seconds, on-demand provider usage to 25 seconds.
No provider call, policy write or inference is performed on public page load.

No backend schema, provider/model selection, consent, budget or batch was
changed. No restart, deletion, cancellation or resubmission was used. The
existing server reads the HTML and CSS files per request, so this frontend is
served by the existing installation. Tracked Git preimages remain recoverable.

## Proof

- Four new tests initially failed on the old page, including an actual old-
  session API response being returned. The final focused dashboard set passes
  **34 tests in 4.57 seconds**. These are actual extracted JS functions with
  synthetic data and existing route/JS parse/privacy tests, not model calls or
  a full project suite. The unchanged roaming Hugging Face environment warning
  remains nonblocking and was not addressed by changing the interpreter.
- API tests cover token changes before the response, old unauthorized replies,
  and an epoch change with the **same token string while JSON is pending**.
  Handshake tests complete two login attempts in reverse order and check that
  lock invalidates an outstanding same-token attempt. Lock tests include private
  text/forms, transcript capabilities, pollers and the legacy activity badge.
- Unit/status tests distinguish recorded succeeded/reused/runnable/failed/
  uncertain/privacy-parked windows from pending source versions. Malformed,
  negative or missing values stay unknown; HTML-like state strings are not
  rendered as HTML. Provider daily/monthly key usage is explicitly not run or
  app-ledger spend. Failure clears old cost/status values. No automatic POST is
  added to public initialization or status refresh.
- At **2026-10-06 07:31:58 UTC**, the existing service reported one listener,
  health 200, anonymous protected 401 and authenticated protected 200. Running
  the actual `operatingView` function over that live status rendered 28 windows /
  20 provider requests awaiting provider, 7,095 all-capture-lane recorded jobs,
  2,532 succeeded, 501 reused, 128 runnable queued, 842 failed, four uncertain,
  3,088 privacy-parked and 6,073 pending source versions. These are that sample's
  operational hints, not independently audited publication/run-success totals.
- That formatter separately rendered latest-version enrollment queued 2,457 /
  existing 924 / excluded 848, and all-version enrollment queued 2,124 /
  existing 3,631 / excluded 848; both complete **only for enrollment**. Catch-up
  was active with 42 sampled minutes left; expiry is not completion.
- The exact worker producer establishes `items = len(record['items'])` as windows
  and `provider_requests = len(payload(record['items'])['requests'])` as packed
  root requests (`historical_batch_worker.py`, units and repair status). The
  renderer also checks an explicit `windows` field when present.
- The real localhost browser shows the new title/login, focuses the token input
  and exposes only the login dialog in accessibility while the background is
  inert. Its 390px responsive check had client width 390 and scroll width 390;
  viewport was reset afterward. This is **locked-screen proof**, not signed-in
  manual workflow or authenticated layout proof. No token was injected or shown
  to the agent/browser tool.

## Review and preservation

The first native review did not return in a useful interval; its running handle
was closed. The replacement cheap native reviewer reported model capacity
failure. One available native reviewer cleared the auth design conditioned on
the shared API guards, then reviewed the actual integration. Its concrete
activity-badge cleanup flag was corrected and covered by the focused tests;
the producer unit-contract flag was resolved by source and live formatter proof.
The final independent changed-issue examination returned CLEAR for the corrected
badge/source path and kept the first-boundary proof limited to focused tests,
live formatter units and locked responsive view. It did not rerun those tests
or certify signed-in workflows, full accounting or the whole installation.

A subsequent real HTTP check returned 200 for both public assets and established
that their response-byte SHA-256 values exactly match the local HTML/CSS files.
The actual credential-worker inspection still found one active worker and zero
inaccessible process records. Worker presence is not a claim that its hidden
passphrase prompt has been answered or that triage has completed.

The design-only commit's installed pre-commit hook temporarily hid unstaged UI
edits and restored them automatically. No manual stash/reset/clean was used;
the restored source was checked and all 34 focused tests passed afterward.
The implementation commit stages all affected tracked edits together to avoid
triggering that hook's partial-staging behavior again. Unrelated untracked files,
including an empty literal quote filename from an earlier failed shell command,
are preserved, not deleted or included in the commit.

## Open gates (not substituted for the full goal)

Signed-in manual browser QA; interactive cited-memory look-up/source following;
generic ambiguity review/actions; separately scoped local credential-use/audit;
batch-consent UI; run-aware settled accounting/reconciliation; authoritative
backup currency and installed revision; natural-host client proof. The historical
model backlog and credential ambiguity work are not complete. This frontend
boundary does not certify the whole installation or whole legacy-data privacy.
