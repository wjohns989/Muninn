# Ordinary memory credential boundary

Status: accepted for scoped source; live installation pending candidate preflight.

## Evidence and requirement

A read-only audit of 1,450 local ordinary memories found three duplicate device
Wi-Fi-password assignments and one false-positive token-status description.
No values, memory identifiers or raw excerpts are recorded here. Ordinary memory
previously admitted text/metadata before telemetry, extraction, embedding and
plaintext indexing. SQLite record hydration returned originals to retrieval,
reranking, consolidation and agent briefing. Archiving alone does not block get
or restore.

Credential values belong only in the authenticated local vault. Ordinary search
must retain usable service/presence/location metadata without exposing values.
Existing originals and recovery history must not be deleted or silently rewritten.

## Proposed decision

Use one credential-specific, pure projection/validation module. Reject detected
credential-bearing new ordinary intake and updates before telemetry or models;
validate extracted output before indexing. Defend SQLite writes independently.
Return redacted, in-memory legacy projections at record hydration, covering the
actual consumers of ordinary records before reranking and extraction. Apply the
same rules to profile/goal/handoff data feeding agent briefings. Reject rewriting
legacy credential-bearing content/metadata/profile/goal from its projection;
score/access/status-only operations can preserve the unchanged original.
Read projections carry a private, nonserialized marker; automatic consolidation
and integrity candidates skip them after advancing the original page cursor.
Guarded rewrites use an owned SQLite writer transaction, not the legacy shared
connection that unrelated worker commits could release. Credential assignment
labels remain searchable; a credential value used as a dictionary key does not.

Do not reuse broad transcript screening here: opaque hashes and project paths
are useful ordinary metadata, not automatically credential values. Protect full
quoted/unquoted assignment values, sensitive structured fields, known credential
prefixes and private-key blocks. Preserve explicit redaction markers and bounded
credential-presence/status descriptions. This is conservative best-effort
detection, not proof against unlabeled or novel credential formats.

## Alternatives and limits

An HTTP-output-only scrub misses models and internal consumers. Archiving misses
direct get/restore. A second credential quarantine store contradicts vault-only
storage. Automatically replacing legacy originals would violate preservation.

Legacy plaintext originals and any existing vector/graph copies remain a pending
authenticated migration: hidden local vault unlock, validated encrypted preimage,
vault insertion with source provenance, then separately reviewed coordinated
store/index remediation. This patch must not claim vault-only legacy persistence
or complete graph/federation coverage. No vault unlock, schema migration, external
dispatch or deletion is part of this source change.

## Proof

Synthetic-only tests in temporary SQLite stores: reject before telemetry/models;
full punctuation/multiline value redaction; nested metadata/keys; unchanged normal
paths/hashes/status; legacy projection with byte-identical original preserved;
refusal of projection-driven rewrite; direct-write and update/restore bypasses;
goal/profile/handoff reads/writes; original-row writer-lock protection despite a
legacy-connection commit; consolidation cursor progress over protected pages;
targeted old-defect mutation in temporary data. Reuse unchanged capture/batch evidence and do
not dispatch models just to test this boundary.
