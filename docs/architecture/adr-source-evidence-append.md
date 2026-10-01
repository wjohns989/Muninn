# Certified append parsing without skipping authentication

Status: implemented in source; live activation and historical throughput are
separate acceptance checks.

## Decision

Growing Codex/Claude JSONL transcripts may reuse an immediate parent's sealed
source-unit evidence only when the exact current entry/version occurs in the
authenticated same-origin manifest history and has a valid byte-prefix
certificate. The parent must have evidence under the same parser identity.
The first current raw iterator hashes the prefix and counts physical LF bytes,
retaining at most its current bounded archive chunk. Only an empty or LF-ended
prefix is eligible. The same iterator continues into suffix metadata parsing.
The second current raw iterator also authenticates the full prefix before
suffix body parsing. Both must reach whole-snapshot authenticated EOF.

Parent fragments are decrypted and sequence/coverage-validated, then encrypted
again under the child's existing source identity and staging attempt. Bounded
page reads release their SQLite transactions before yielding to the child
writer; a final pinned ciphertext fingerprint scan detects intervening parent
changes. Cancellation is checked during this scan as well as copying/skipping.
There is no new plaintext cache or persistence format.

Codex inherits the final parent unit's cwd/project basis, including omitted
metadata units; explicit invalid/absent cwd metadata still resets that basis.
Claude cwd remains record-local. Suffix unit ordinals include all parent units.
Physical coordinates retain the existing zero-based convention and add the
prefix LF count, including trailing blank lines. Message timestamps remain
provider evidence, never capture time or file mtime.

## Failure and fallback

Unavailable parent evidence, absent certificates, rewrites and Gemini retain
the full parser. A non-LF prefix closes its bounded probe and starts a full
metadata pass; its untouched body iterator remains the original full pass.
Corrupt certificates/evidence or late current-source failures reject the build,
clean only its derived stage and preserve the original parent. A cancelled
build cannot become a complete snapshot.

## Proof and limits

Isolated temporary encrypted-archive tests compare append output to cold output,
instrument both parser inputs and both raw EOFs, cover trailing blank lines,
context resets, small raw chunks, an empty parent, more than 64 parent pages,
missing/legacy/non-LF fallback, corrupt parent fragments, late failures in either
raw pass and cancellation during the final parent fingerprint scan. They use no
installed models, live data, provider dispatch or user credential configuration.

Reuse saves old-prefix JSON tokenization, not full current-source authentication,
parent fragment copying or all I/O. Cold-parser crash resumption, multi-generation
analysis-reuse flattening and whole-backlog throughput remain separate work.
