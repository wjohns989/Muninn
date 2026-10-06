# Credential scanner escaped-whitespace regression

## Scope and defect

Project text and transcript assignment scans included the final letter of a
literal `\n`, `\r` or `\t` escape in the credential's service name. This made
`OPENROUTER_API_KEY` appear as `nOPENROUTER_API_KEY`, for example, and caused
bare `API_KEY` assignments to be missed entirely. Synthetic fixtures reproduce
both defects; no real credential value or source file was read for these tests.

## Change and proof

The shared assignment boundary consumes an escaped whitespace separator before
matching the identifier. It does not strip letters from identifiers or decode,
rewrite, index, log or transmit credential values. The strict `.env` scanner is
unchanged. Original bytes and source offsets are not transformed.

Before the change: 16 escaped-boundary cases failed and 6 genuine-prefix controls
passed. The regression checks cover both scanners, `\n`, `\r`, `\t`, `\r\n`,
qualified and bare names, and every chunk split across the escape and identifier.
The genuine `n`, `r` and `t` identifier prefixes are retained.

After the change: 144 focused tests passed in 23.52 seconds:

```text
tests/test_credential_discovery.py
tests/test_credential_context.py
tests/test_credential_zdr.py
tests/test_ordinary_credential_boundary.py
```

One unrelated Hugging Face environment-variable deprecation warning was emitted.
This is not a full-suite or merge-readiness claim.

Independent source/diff review: CLEAR. The reviewer found no changed value,
source-byte, provenance or storage boundary requiring a broader rewrite or
rescan. The focused regression suite was judged sufficient for this patch.

## Existing data and runtime limits

No vault records, source receipts, encrypted batches, policies or live queues
were changed or deleted. Already running processes keep their imported parser;
new scanner processes use this fix. No service restart was performed for it.
Previously scanned, unchanged sources may retain old metadata because scan
receipts intentionally suppress repeat work. This patch does not claim those
existing records have been repaired; any repair must preserve provenance and
encrypted recovery history, rather than blindly stripping name prefixes.
