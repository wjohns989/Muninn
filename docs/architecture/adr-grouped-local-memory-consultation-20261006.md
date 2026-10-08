# Grouped local consultation for noncredential memory ambiguity

2026-10-06. This closes the missing human-consultation consumer, not automatic
semantic filing or historical ambiguity completion.

## Contract

`memories triage` prompts for the portable history-archive passphrase in an
interactive local terminal. It reads one anchored review page (1–20 candidates)
and organizes safe unresolved memories by authenticated original snapshot,
project attribution/basis and memory type. Excerpts from one snapshot share an
opaque source group even when their citation references differ. Unknown project
attribution stays per-candidate. No raw source path or secret is added to public
API/MCP results; existing flat review results remain unchanged.

Original event times and their provenance basis are retained. Items are sorted
by event time within each group, with unknown times last. Cursor advancement
uses publication-order selection before this sorting. These are partial groups
within one anchored page, not full-source groups, global chronological coverage
or a determination that similar claims contradict each other.

The consumer presents the existing freshly authenticated safe source context
without minting a transcript capability or initializing its blind index. Read
and skip/quit operations open no ledger writer and create no preimage. Each
decision requires the exact state and candidate ID to be typed locally:

- File: `user_confirmed`.
- Reject: `user_rejected` (append-only decision; original data is not deleted).
- Clarification: `insufficient_context`.
- Possible contradiction: `possible_contradiction`, a review concern, not proof.

Before the first confirmed decision, the existing verified encrypted ledger-only
preimage is taken. The existing compare-and-swap method authenticates current
state/citation; every decision is read back. Citation, type, scope, timestamp and
truth labels remain unchanged. Filing does not transform a model assertion into
verified truth. Grouping grants neither model nor shared-bearer write authority.
The preimage depends on the archive key/source and is not a portable full backup.

## Local use

Run only when ready to review actual candidates. No model call is made:

```powershell
Set-Location -LiteralPath 'C:\Users\user\muninn_mcp'
& 'C:\Users\user\miniconda3\python.exe' -B -m muninn.cli memories triage `
  --archive-root 'C:\Users\user\muninn_mcp\.muninn_runtime\history_secure_archive' `
  --limit 6 `
  --backup-before 'C:\Users\user\muninn_backups\memory-review-20261006-first'
```

Use a fresh destination under a private existing backup parent. An existing
destination is never overwritten. Resume with the returned `--cursor` and the
same page size; use another fresh preimage destination for a later consultation.
The command's page-complete result is not whole-queue completion.

## Proof scope

Initial regression demonstrated the absent grouping consumer. The cancellation
hash check then exposed index creation by the existing context-following method;
the new no-capability read path fixed it. Focused synthetic encrypted-archive
checks cover source versus excerpt identity, separate/unknown scopes, timestamp
ordering with cursor coverage, credential exclusion, portable unlock, quit/bad
confirmation without writes, failed backups, stale decisions, one verified
baseline for multiple decisions, unchanged truth/citations and CLI integration.
Independent source design review cleared these boundaries before implementation.
Final affected verification: **114 passed in 50.75 seconds**, covering the new
grouped consumer, flat queue, CLI, ledger, cited-memory API and MCP review mode.
There was one unrelated Hugging Face environment deprecation warning. Actual
diff/results review independently cleared scoped integration. The installed
Miniconda interpreter's CLI help exposes `memories triage`; `git diff --check`
passed. No shared-service restart, model call or live review mutation was made.
Real operator decisions require local passphrase/confirmation and are not
claimed by isolated tests. General automatic semantic classification/filing
remains a separate dependency; the model interpretation route remains provisional.
