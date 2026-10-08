# Safe mixed-transcript memory retrieval

## Observable outcome

New private ZDR interpretations can publish exact safe original-coordinate
excerpts from mixed transcript units. Ordinary cited-memory lookup, search,
source following and review pages return these excerpts as provisional model
interpretations. They never turn withheld text into evidence or acquire direct
source-rule filing authority. Credential-risk candidates remain withheld.

This extends the earlier private-ZDR-window implementation; it does not
reinterpret or replace legacy candidates, staged extractions, publication IDs,
receipts, retained batches or their bills. No provider policy or budget changed.

## Proof and persistence boundary

- Only the local projection adapter adds `source_view`; it is not a model-output
  field. Its descriptor and ranges are encrypted into the existing stage.
- Stage reads, receipt checks and restore revalidate the canonical view after
  authenticated whole-unit EOF. Writable ledger initialization is forbidden in
  this nested-reader path; the source reader is explicitly read-only.
- Publication and expected-ref computation carry the same view through both
  service paths and batch replay. New candidates include it in their identity.
  Missing view means the unchanged legacy policy and identity.
- New public text/context, search matches and review actions recompute the
  canonical view and verify original coordinates, non-crossable ranges, claim
  and quote screening. A failed proof withholds those fields/actions.
- Allowlisted metadata still requires an authenticated original citation and
  ledger. The separately scoped transcript capability returns only the existing
  credential-redacted transcript projection, not raw data or credential values.
- Credential type/risk remains dominant. Human filing requires local portable
  passphrase unlock and expected-state CAS; filing does not verify model truth.
- A proposal batch drains a unit once for the range proof, not once per proposal.
  Source context reuses one proof within that call, never a persisted plaintext
  or stale cross-request projection. Context is bounded and labels its partial
  visibility and window-relative coordinates.

## Focused evidence

- Red-first: the new mixed-transcript publication test failed because
  `record_proposals` did not accept a source view.
- 226 affected ledger/source/projection/service/review/journal/batch tests passed
  before the final descriptor-type and redundant-drain tightening.
- 24 final projected-publication/private-claim checks passed after those changes.
  They cover strict canonical ranges (including numeric type aliases), empty
  stages, risk dominance, late authentication failure, original prefix/page
  boundaries, repeated quotes, local passphrase/CAS, portable restore, read-only
  expected refs, one drain for two proposals, and encrypted staged recovery with
  settled admission, exact receipt verification and no redispatch. The recovery
  scenario is exercised with both new and legacy stages.
- New-file lint and diff whitespace checks passed. No full-suite/merge claim.
- Independent scoped privacy/recovery review: CLEAR, including the final strict
  descriptor validation, read-only receipt path and one-drain change.
- Real parked-source read-only proof: 3,000-character window, 41 canonical
  ranges, safe provisional public candidate available. Zero records published,
  zero provider calls, and no raw text displayed or persisted by the check.

## Remaining scope

Redaction is strict-best-effort, not an omniscient credential classifier. This
does not prove paid private-route model quality, resolve the historical backlog
or retrofit safe visibility onto legacy withheld records. Larger multi-window
provider packing is separate work; no current batch was enlarged or resent.
Live installation and post-reload checks are recorded separately below.

## Local installation

Installed source revision: `c49b6427003c19dfc5f7a774a693a914914996de`.
The existing owned Muninn service alone was reloaded with its current capture
settings. Six encrypted database preimages validated; no in-flight model/search
job or CPU capture required recovery. Post-reload: one process on port 42069,
health 200, anonymous protected access 401, authenticated access 200, strict
archive ready, automatic capture/remote processing and remote-only mode retained.

Before/after encrypted-state fingerprints were identical:

- 711 managed paid-admission rows, SHA-256
  `24c4f1e6b0cf76427361cf12f0fff9f25bd51e19dcaa5e05d69ffb3b6195a7a2`.
- Eight total rows across the retained-batch database's tables (including its
  head, **not eight provider batches**), SHA-256
  `09db57d13643141f551813f7207c9498303c5d5af8cfed38ff6ae9a30ac37365`.

The same 49-request checkpoint remains awaiting its provider. Installation did
not admit another paid request, cancel a batch or change the spending policy.
Installed stdio bridge smoke passed: 20 tools, current Codex/Claude Code/Desktop/
Gemini profile configurations match, context lookup succeeds, and two cited
review pages with continuation succeed (47,489 ms for the smoke process).
This is a real bridge/profile check, not proof of every host's capture-hook event
or a paid new projected result. The backlog remains incomplete.
