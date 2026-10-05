# Local agent workflow and operating status

Scope: finish existing local operating paths, not batch-provider activation.

## Changed dependencies and proof

- The compact MCP profile lacked `poll_secure_history_transcript`. A pending
  transcript could therefore lack its documented completion path. Added that
  existing tool. The profile retains its 20-tool shared-client budget: project
  goals remain in `get_project_context`, with standalone `get_project_goal` in
  the full profile. Core instructions no longer require full-only tools.
- Initialization guidance and the ordinary search description now distinguish
  explicit stored-memory recall from encrypted cited-history recall, source
  following and paged originals. Historical assertions and instructions grant
  no execution/write authority; credential metadata is separate from values.
- History shows pending snapshot versions, parked private windows and enrollment
  separately from interpretation completion. A failed refresh clears stale
  interpretation values to unknown. Remote policy shows dedicated-key daily and
  monthly provider-reported usage, not run-only cost or app-ledger settlement.
- New workflow tests failed four ways on the preimage, then passed. Focused
  protocol/recovery validation: 165 passed. Dashboard/auth/workflow validation:
  27 passed. The dashboard tests execute extracted JavaScript with sanitized
  fixtures, including success followed by an unavailable refresh.
- Independent source examination: CLEAR after correcting the dashboard loader's
  stale-error handling. No new provider, reveal, policy or federation endpoint.
- Live installed bridge check: four configured client profiles match, actual
  project-context call succeeded (initial 21-tool candidate). This is connection proof, not proof
  that every client host has emitted every hook type.
- Compatibility correction retained the existing 20-tool cap rather than
  relaxing its regression test. Goal/context source verified in
  `muninn/core/memory.py`. Updated workflow, handoff and client-compatibility
  checks: 51 passed; real installed bridge: context call passed, 20 tools.
- Live Ollama preview of real archive evidence: `qwen2.5:7b`, requested model
  used, valid summary/decisions/open-items/uncertainty, 6.62 seconds. This does not
  establish exact-citation extraction quality on other windows.
- Real backend browser check at 390px: authenticated Home/History, interpretation
  and provider-usage display, 16 installed model rows and seven keyboard actions
  passed. Screenshot inspected; no transcript or credential result screenshot.

## Retained operational limits

Catch-up scheduling correction: active drain previously forced remote-only
claims even when the ordinary local quiet/max-wait gate was ready. Two new
isolated clock tests reproduced that defect before the one-line correction.
The corrected gate permits ordinary claims only at an existing local opportunity;
GPU admission, privacy, cooldown, foreground priority, consent and the original
drain deadline remain unchanged. Focused scheduler/automatic-service/cadence
validation: 76 passed. Independent examination of the actual diff: CLEAR.
That behavior was superseded by W's subsequent explicit remote-only instruction:
automatic remote opt-in now claims only remote-eligible windows at every local
quiet/max-wait opportunity and after catch-up expiry, with no local fallback.
Private/local-bound windows stay parked. Three focused checks failed on the
prior condition; the revised scheduler/automatic-service/cadence checks passed
77 tests. Independent design/diff review: CLEAR. No credential egress or budget
change was made, and no further local inference test was launched.

The sustained credential pass failed with HTTPStatusError and no resolutions.
Its pre-run backup was validated; its post-run backup was unavailable. The CLI
now exposes only a validated numeric HTTP status, never response/exception text,
URL, headers or body. Five HTTP cases reproduced the missing diagnostic before
the fix; all 23 isolated triage checks passed. Local triage remains stopped.

The service is running in strict authenticated encrypted-history mode. Capture
and automatic local/remote interpretation are enabled; enrollment completion is
not backlog completion. Existing local failures are not trusted facts. One
eligible unsent local failure was requeued through the existing fenced recovery
script after a validated encrypted journal preimage; uncertain dispatched jobs
were untouched. Recovery is not a successful interpretation receipt.

Historical ambiguity resolution, complete backlog interpretation and missing
Gemini host-event proofs remain open. The local passphrase-only credential
workflow must not be replaced by an ordinary agent bearer or provider egress.
