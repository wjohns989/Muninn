# Credential parser and paid-receipt compatibility

Status: accepted design; focused isolated verification, not whole-installation completion.

## Context

The escaped-whitespace parser fix changes some recorded variable names and masked
request bodies. Reusing the old replay identity would cache pre-fix extraction as
current. Globally replacing the identity would strand encrypted decisions and
paid receipts. An inspection found 629 possible credential-name prefixes and
1,184 possible ambiguous-name prefixes. Their 209 distinct archive source scopes
span 15,304,906,061 bytes. These are candidates, not confirmed incorrect records;
no global live rescan or metadata rewrite was performed.

## Decision

- New context replays use authenticated parser revision 2. Revision 1 remains
  readable with its original completion seal, reviews and remote receipts.
  Reader selection authenticates the exact source/version/attempt, without
  changing shared reader identity or rewriting existing encrypted rows.
- A legacy/corrected variable-name join requires the same immutable source,
  candidate and reason, plus an exact escaped n/r/t boundary in original context.
  Literal prefixed names, mixed conflicting forms and mismatched values do not
  authorize aliasing. The queue identity and observed source name are preserved.
  Parser-2 replay is prepared on demand once for a requested source. Old replay
  can be selected only after fully draining both authenticated streams and
  proving equal complete ordered coverage; one matching old occurrence is not
  enough. Additional newly discovered occurrences require the new replay.
- Current masking is the only dispatch recipe. Old masking is reconstructed
  solely for exact paid-receipt lookup. An authenticated received receipt is
  settled/reused without another POST. An old intent permits a new current-body
  admission only with exact released/unsent ledger proof; it is retained.
- An unexplained paid/uncertain receipt at the same source attempt/page blocks
  automatic dispatch. Changed metadata or model identity is not permission to
  repeat a possibly charged occurrence. Existing privacy and budget gates apply.
  A new replay also checks other completed attempts for that exact immutable
  source: paid/uncertain work blocks new dispatch without explicit occurrence
  mapping. Proven-unsent attempts do not block it. This conservative fence does
  not claim automatic migration of old paid decisions to different coverage.

## Alternatives and trade-offs

Blindly strip name prefixes: cheaper, but corrupts real variable names. Rebuild
every snapshot: simpler cache invalidation, but expensive and risks duplicate
reviews. Rewrite old authentication identities: breaks portable recovery and
receipt provenance. The selected two-format reader adds small compatibility
logic while preserving recovery and paid-work reuse. Conservative unmatched
receipt handling can require operator resolution rather than automatic progress.

## Verification and limits

191 focused tests passed in 36.62 seconds across context replay, masked ZDR,
ambiguity triage/waiting, provenance and discovery. Red-first regressions showed
missing legacy recipe support and a second paid dispatch before the fix.
Fixtures cover authenticated old/new formats and portable restore, exact alias
evidence, frozen old masked context, paid reuse/cost counted once, unknown and
  mismatched receipt blocking, and proven-unsent current-body-only dispatch.
An additional red-first coverage test reproduced an old replay omitting the
second escaped bare API_KEY occurrence; the fixed preparation yields both.
All provider traffic in these checks is mocked; no local model or paid provider
call is test evidence. Live triage still requires its local unlock. During
isolated verification the waiting process imported older source/context classes;
that process was not accepted as proof of the new code. Installation followed
the independently reviewed bounded procedure below.

This does not resolve all historical ambiguity, update old vault metadata, prove
project-file alias correction, or complete the backlog/whole local installation.
Revisit when an explicitly authorized source-bound replay needs metadata repair
or when a new masking contract requires another exact receipt recipe.

## Installed execution evidence, 2026-10-06

Candidate `bee0854c5a4ef6bfcef1787a71c92f096dab6cd3` passed independent fresh-CLI
diff review. The old imported worker was stopped only after matching its exact
parent/interpreter/checkout/module and latest awaiting-passphrase state. No
triage worker remained before replacement; no vault value was requested in chat.

The existing guarded reload procedure installed that exact candidate. It
validated six encrypted database preimages and bounded recovery to one CPU
capture. These selected database preimages are not a whole-archive/vault backup
or a complete historical-batch-store backup. Paid ownership/admission gates
passed; no manual batch cancellation, deletion or resubmission was performed.

Fresh probes showed one listener on port 42069, health 200, authenticated
protected access 200, anonymous protected access 401, and strict archive ready.
Existing automatic capture/remote settings remained effective. The authorized
remote catch-up interval was renewed for 180 minutes; the live status confirmed
it active. Expiry is normal-cadence fallback, not backlog completion.

The latest reported checkpoint passed 15 windows in 10 provider requests with
zero invalid items. This is a recent-checkpoint count, not total-run coverage.
At the probe 3,935 snapshot source versions remained pending and 2,423 windows
were privacy-parked; these are separate units and do not yield an overall percent.

A fresh visible launcher and exactly one new triage worker were confirmed alive
in the correct checkout, with progress `awaiting_passphrase`. It uses the same
authorized Luna route, 200-page/60-row/12-call-per-page bounds, readiness wait,
and separate pre/post encrypted-vault backup destinations. No actual credential
classification, completed new backup, or resolved vault is claimed before unlock.

Read-only run accounting since epoch 1791154542.000905 showed $0.731478 in 718
settled local admissions (one released), versus $0.729765072 dedicated-provider
monthly usage above the $0.00252758 baseline. The $0.001712928 difference was
unreconciled. Provider daily usage was $0.073186845 with its $5 daily limit;
managed $5/day and $50/month policy was not raised. These amounts are a dated
observation, not measured savings or a guarantee about delayed provider billing.
