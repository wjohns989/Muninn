# Historical batch recovery: source-only checkpoint

## Concrete dependency closed

W corrected the account guardrail. Authenticated model inventory now includes
Luna Pro batch: OpenAI route, $0.05/M input and $0.25/M output. The previous
catalog-level blocker is resolved. No private batch or public batch was sent.

Implemented `muninn/history/historical_batch.py`: source-screened bounded item
preparation, ordered provider-pinned request payload, archive-key encrypted CAS
outbox, irreversible submission-uncertainty marker, exact custom-ID reconciliation,
separate cited-output validation, aggregate cost validation and durable terminal /
cleanup receipts. It deliberately exposes no HTTP or activation. Preparation is
not authority to submit and HTTP 200 is not accepted memory or a source ACK.

## Retained proof

- Forty isolated batch tests passed (9.82 seconds), including passphrase archive
  recovery independent of Windows unlock, CAS races, tampering, terminal failures,
  reordered/missing/extra/duplicate result IDs, missing/BYOK billing and cleanup
  ordering. Actual encrypted source fixtures validate exact supported citations
  and reject invalid schema, absent quotes, malformed response structure and
  truncated output; invalid replies remain encrypted and capture ACK stays zero.
- The unchanged cited-source/transport suites passed: 49 tests in 25.01 seconds.
- Ruff and whitespace checks passed. An in-memory mutation allowing resubmission
  from unknown state caused the intended restart regression to fail; no source
  mutation or fixture copy of local credentials was persisted.
- Independent design/result review cleared the isolated, inactive slice after
  the transport-success versus accepted-memory distinction was made explicit.
- Authenticated existing service remained HTTP 200; no reload, policy edit,
  credential write, inference, or queue mutation occurred during these checks.

## Still required before real batch use

Exclusive capture ownership; separately revocable batch retention consent;
verified aggregate budget escrow shared with synchronous admission; authenticated
HTTP submission/polling; idempotent publication and recovery; automatic outbox
backup/restore inclusion; then a small actual-window pilot with cost and retained
result receipts. W's latest instruction prohibits deleting anything, especially
batches: no automatic or operator-script DELETE is authorized. Retain encrypted
input/results locally; provider-managed expiry remains an external limitation.
Existing synchronous admission has no per-call upper charge reservation,
so simply adding a batch hold beside it is not a proven shared-budget guarantee.
Do not silently treat a daily provider limit as protection for monthly escrow.

This is not installed batch dispatch, full backlog completion, or full local-goal
completion. The live synchronous ZDR path and user-owned files remain unchanged.
