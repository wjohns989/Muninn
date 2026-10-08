# Signed-in cited-memory browser proof

## Actual current installation

An isolated authenticated Edge context exercised the existing service on
127.0.0.1:42069. Only local GET/HEAD and the four explicit read-only cited-memory
POST routes were allowed. Any other request would fail the probe. Browser
downloads and service workers were disabled; no private traces, screenshots,
credentials, cursors, record text or source context were printed or persisted.

The terminal browser check passed:

- Eligible review returned six items; the next page returned six disjoint
  record identities with the previous detail view cleared.
- Record lookup and cited-source following preserved exact record identity.
  The explicitly available, redacted context contained 1,362 Unicode characters
  and the DOM exactly matched the response. The 3,000-character bound held.
- Search returned ten records and the DOM count matched.
- A 390-pixel viewport had no horizontal document overflow.
- Locking cleared result, detail and transcript panes and hid the review cursor.
- Mutating or external requests: zero. Private output persisted: false.

The first review response was HTTP 200 at 11.101 seconds from browser creation.
This is not an isolated request latency. No provider inference, filing, review
decision, source-redaction job creation, service restart or policy edit occurred.

## Timing and prior failed observation

The previous browser check timed out after 30 seconds observing the eligible
review response while the separate restore drill was active. Its backend task
was not treated as cancelled and the probe did not retry or issue later reads.
The cause of that particular timeout remains unproven; contention is not a
confirmed diagnosis.

A separate current-source, read-only real-ledger measurement completed a
six-item review in 14.665 seconds. It visited 6,582 ledger events and made 6,589
authenticated source reads taking 14.296 seconds. Reader construction took
0.093 seconds. Neither the source schema nor any integrity/privacy check was
changed for this measurement. This establishes a real current cost, not a
future-size guarantee or measured optimization.

This closes the previously unrun signed-in cited-memory consumer check in
`cited-memory-ui-20261006.md`. It does not prove generic review writes,
credential use, natural client hook execution, full transcript retrieval,
constant-time or deadline-bounded reads, service candidate identity, the full
test suite, or completion of the entire local installation.
