# Cited retrieval: read-only boundary and representative proof

## Changed dependency

The three cited-memory search, lookup and source-following service methods now
use the existing read-only ledger. Authenticated source scanning and redaction
checks remain required, but a read-only scan cannot write a screening attestation.
Publication and other writer paths remain unchanged.

## Focused validation

- The new writer-construction assertion failed against the original search path.
- After switching constructors, the test exposed a second write path in source
  screening. The read-only guard fixes that path without skipping authentication.
- All 128 affected tests passed in 34.67 seconds: cited API, memory ledger,
  agent history workflow, MCP review mode and secure history API.
- ASGI coverage requires read-only construction, prohibits attestation writes,
  follows exact source citations through redacted transcript paging, and checks
  main-token/local-client enforcement. Cold search returns a private no-store 503
  without creating an empty ledger or claiming empty historical coverage.
- Independent native review cleared the source and test diff. This is focused
  component proof, not a full suite or whole-installation readiness claim.

## Live baseline before installation

Authenticated cited search returned two entries (19 matches), exact-reference
lookup returned a provisional model-inferred entry, and source-following returned
a 186-character redacted span plus a transcript capability. All three returned
HTTP 200 with no-store; search took 10.84 seconds. No private content or credential
value was printed or saved. Live transcript paging remains a separate check.

The existing Desktop launcher recognized the running authenticated installation
and reported outstanding capture/analysis jobs without creating a second server.
This checks its already-running branch, not its service-absent startup branch.

Retained encrypted data and batches are unchanged. No model call, new provider
submission, cancellation, deletion or credential policy change is part of this fix.
