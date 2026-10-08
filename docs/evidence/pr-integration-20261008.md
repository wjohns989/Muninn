# Portable PR integration checks — 2026-10-08

PR #142 remains an integration candidate, not a merge or service-installation
approval. The shared Windows listener, credentials, queues and provider holds
were not changed by these fixes.

## Observed failures and changed dependencies

- Candidate `c33eb95` failed the privacy gate because the tracked public backup
  verifier matched the generic `verify_*.py` ignore rule. The exact verifier now
  has an exception; private/runtime exclusions and the privacy workflow are
  unchanged. Account-specific public documentation and synthetic fixture paths
  were replaced with accepted placeholders without changing live configuration.
- Its locked Linux suite stopped after 276 passes and six skips because retained
  diagnostic validation imported undeclared `jsonschema`. The core dependency
  and its locked dependency graph were added without upgrading or removing
  existing packages. The 59 affected local tests and offline lock consistency
  check passed. Independent review cleared that exact diff for commit/push.
- Candidate `6e4e313` passed privacy, clean-install imports, benchmark dry-run
  and replay. Its full Linux suite reached 284 passes and six skips, then found
  a fixture that relied on Windows DPAPI. The fixture now reopens its actual
  encrypted archive with its known synthetic passphrase, preserving the real
  unlock and expired-watcher no-GET assertion. Production unlock is unchanged.
- Candidate `e9f1eeb` passed those four gates and reached 488 passes and seven
  skips in its full Linux suite. It then called the explicitly Windows-only
  unattended backup operation from an unguarded test. That operation is not
  newly claimed to work on Linux: the original actual backup/restore test now
  has a function-level Windows guard, and a separate all-platform passphrase
  restore test authenticates the encrypted empty-window completion proof.

## Actual Windows proof and remaining gate

The full affected `test_capture_no_context.py` file passed 29 tests in 19.38
seconds on the canonical Windows interpreter. Both real unattended backup plus
restore and explicit-passphrase archive restore ran; neither was skipped. The
original negative, paid-work, rollback, lease and cancellation assertions remain
unchanged. Independent review cleared the actual test-only split for commit/push.

Candidate `44eda15` again passed those four gates; its full Linux suite reached
757 passes and eight skips before another unguarded Windows-only archive backup
call in classification recovery. The remaining direct/helper callers need a
grouped platform-proof audit, preserving portable encrypted managed-state
recovery and real Windows assertions. This is not full-suite success.

Focused local passes and the four already-green PR gates do not establish full
installation completion, live candidate identity or credential recovery. The
separately approved diagnostic adjustment and verified live backlog resumption
are recorded in [the resumption evidence](backlog-resumption-20261008.md);
no test fix supplied that authority or converted unknown provider billing into
an actual charge.
