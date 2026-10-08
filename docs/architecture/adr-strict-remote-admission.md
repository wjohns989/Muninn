# Durable strict remote admission with conservative cost floors

Status: implemented in source; live activation and hard-cap enforcement remain
separate acceptance gaps.

## Context and alternatives

A provider-usage probe alone allowed concurrent strict requests to spend the
same apparent headroom and allowed a lost response to vanish from admission.
An in-memory lock would not cover direct/worker calls in different processes or
restart. Exact price reservations require a verified upper request charge and
provider compatibility; they are not achieved by guessing token counts/prices.

## Decision

Use an additive journal in the existing owner-private policy SQLite database.
`BEGIN IMMEDIATE` serializes admission with policy changes. An enabled exact
consent generation, provider readiness and headroom in both periods are required.
A single unresolved row reserves all further strict admissions at that root.
The states are reserved, unknown, settled and released. There is no time-based
expiry: a new day/month cannot prove an old request was unbilled.

Before POST, commit unknown synchronously with bounded SQLite lock waiting, not
an unjoined writer thread. The reservation-to-dispatch path is within cleanup's
try/finally. The normal path releases only when POST never began and any attempted
durable job marker is confirmed unsent. Ambiguous marker/accounting failures stay
blocking. Cancellation is checked after the lease marker and before constructing
the HTTP client. Valid response cost is settled before parsing model output or
raising an HTTP-status error; missing/invalid billing information is not zero.

Preserve JSON and CLI decimal cost lexemes through upward micro-USD rounding.
The period floor is max(provider-reported usage, local settled charges), avoiding
double counting the same calls. A boundary-crossing charge conservatively counts
against both its start and completion periods. Settlement and proven-unsent
cleanup require the existing ledger, not current enabled consent/generation.

A disk marker precedes initialization, with an additive policy-table sentinel
and tables committed before an ordinary admission refusal. Established marker,
sentinel or table loss fails closed; interrupted initialization requires explicit
operator repair, not silent reset. No secrets, transcript, provider request ID
or process ID is stored. A local authenticated status API returns costs/counts,
and only the explicit local CLI exposes opaque admission IDs for reconciliation.
Operator confirmation must mean the call is no longer in flight and its actual
charge has been verified; this is not an automatic retry or a consent change.

## Consequences and proof

Remote strict calls serialize at each policy root. An unknown outcome can block
paid analysis until local reconciliation; capture/search/local analysis remain
independent. This trades remote throughput for durable conservative admission.

Tests use temporary private policy/accounting databases, fake HTTP, and a real
HTTPX response with a synthetic precise cost lexeme. They cover concurrent calls,
unknown response, cancellation, revocation cleanup/settlement, stale usage floors,
restart/UTC rollover, schema loss, invalid cost, explicit reconciliation, output
failure after billing, and token/loopback/no-store status boundaries. Three older
transport-test modules explicitly fake the added accounting boundary; dedicated
new integration tests use actual SQLite and do not request that fixture.

An unbounded individual request, stale provider usage, other roots/clients and
legacy analysis are outside this guard's dollar guarantee. A valid returned
result with unknown billing can still be used; further paid calls remain blocked.
The journal is not authenticated against owner tampering or rollback to a valid
older complete policy database. Full control-plane backup/restore proof and a
verified per-request upper cost bound remain necessary for automatic paid capture.
No provider dispatch, key access, policy mutation or service reload is performed
by these isolated tests or by creating this source implementation.
