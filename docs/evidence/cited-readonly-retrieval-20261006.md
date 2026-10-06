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

## Installed exact-candidate verification

Candidate `940e2c66434aa4e3a68dd63799b6c37602f6f123` was installed with the
unchanged reviewed reload helper, preservation mode and no deadline renewal.
The helper validated six selected encrypted database preimages and zero active
durable jobs before stopping only the owned service. These are not a complete
archive/vault/batch-store backup. The independent integration examination retained
the helper's prior safety proof but correctly required new-candidate live proof.

Installed cited search, exact lookup and source-following returned HTTP 200 with
no-store. Search returned one result in 24.34 seconds; the exact entry remained
provisional/model-inferred, and source context was 186 redacted characters.
The first live probe incorrectly required HTTP 200 for transcript startup; the
endpoint's documented pending HTTP 202 response started the projection normally.
The corrected probe reused the same source projection, observed ready with 836
pages, then fetched one 4,000-character redacted page with continuation and no-store.
It did not read all 836 pages, certify redaction infallibility or send source text
to any model. The isolated ASGI test separately verifies complete paging.

After installation: exactly one owned process/listener, health 200, anonymous
protected request 401, authenticated protected request 200, strict archive ready,
remote-only automatic interpretation enabled and the original catch-up deadline
still active. The status reported an automatic submitted batch of 28 windows in
20 provider requests; acceptance is not completed interpretation or billing proof.
The existing credential triage worker remained alive and was not replaced.

The portable candidate was pushed to the existing branch/PR. Complete historical
interpretation, credential ambiguity resolution and whole-installation readiness
remain open; this evidence closes the scoped read-only cited retrieval gap only.
