# Read-only backup locator verification repair

## Failure and correction

The existing frozen section verifier ended with exit 1 and `AttributeError`.
A bounded read-only probe of its first capture row identified missing `_aad`
in `ReadOnlyJournal`. The adapter deliberately bypasses the writer constructor,
but had not copied the vault-bound locator authentication domain from
`CaptureJournal`. The frozen journal contains 3,553 capture locators; the earlier
paid-history helper fixture contained none and did not exercise this path.

The correction sets the identical `muninn-capture-locator-v1` domain plus vault
identity in the read-only adapter. Provider binding remains in `_open`; source
key validation and every other section check remain unchanged. It does not call
the writer constructor, migrate a journal, modify a retained stage, create a new
full copy, or affect the running service or provider accounting.

## Evidence

- Before the correction, the enhanced real-locator success test and three
  rejection cases all failed at missing `_aad`: four failed, one passed in
  10.09 seconds. The unchanged production stage probe failed at the same value.
- After the correction, five section-helper tests plus twelve capture-journal
  tests passed: 17 tests in 14.70 seconds. The helper covers ciphertext,
  provider-binding and source-key tampering; it forbids writer construction and
  compares the entire synthetic backup tree byte-for-byte on success/failure.
- Historical enrollment, historical versions and window recovery checks passed:
  76 tests in 43.57 seconds. These are affected checks, not a full-suite claim.
- All 3,553 frozen capture locators subsequently authenticated and matched their
  source identities without displaying their decrypted paths. The frozen
  source-evidence database retained the same device, file identity, size and
  modification time as its earlier complete verification.
- Native independent review cleared the minimal design and actual diff before
  retrying the read-only verifier. `git diff --check` passed.

## Remaining recovery gate

The old failed verifier is terminal, not a live wait. A replacement standalone
read-only verifier now checks the preserved stage's journal, credential context,
ledger and windows using the corrected adapter and the previously reviewed
call-scoped verification code. Its initial journal check was still running at
the last observation. Earlier archive/accounting/batch and source-evidence
proofs remain scoped to the frozen copy; final section completion and publication
are not yet established. No Restic enrollment, complete restore, retirement,
service restart, provider dispatch, upstream push or full installation completion
is claimed by this repair.
