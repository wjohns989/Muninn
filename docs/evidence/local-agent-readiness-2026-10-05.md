# Local agent workflow and operating status

Scope: finish existing local operating paths, not batch-provider activation.

## Changed dependencies and proof

- The compact MCP profile lacked `poll_secure_history_transcript`. A pending
  transcript could therefore lack its documented completion path. Added that
  existing tool; the profile now exposes 21 tools.
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
  project-context call succeeded, 21 tools. This is connection proof, not proof
  that every client host has emitted every hook type.
- Live Ollama preview of real archive evidence: `qwen2.5:7b`, requested model
  used, valid summary/decisions/open-items/uncertainty, 6.62 seconds. This does not
  establish exact-citation extraction quality on other windows.
- Real backend browser check at 390px: authenticated Home/History, interpretation
  and provider-usage display, 16 installed model rows and seven keyboard actions
  passed. Screenshot inspected; no transcript or credential result screenshot.

## Retained operational limits

The service is running in strict authenticated encrypted-history mode. Capture
and automatic local/remote interpretation are enabled; enrollment completion is
not backlog completion. Existing local failures are not trusted facts. One
eligible unsent local failure was requeued through the existing fenced recovery
script after a validated encrypted journal preimage; uncertain dispatched jobs
were untouched. Recovery is not a successful interpretation receipt.

Historical ambiguity resolution, complete backlog interpretation and missing
Gemini host-event proofs remain open. The local passphrase-only credential
workflow must not be replaced by an ordinary agent bearer or provider egress.
