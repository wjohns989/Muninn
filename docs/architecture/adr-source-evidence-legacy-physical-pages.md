# Legacy source-evidence fragment compatibility

## Decision

Keep version-one encrypted physical pages and their original attempt, page,
fragment and unit identities. Existing citations and paid publication receipts
refer to those coordinates; splitting pages in the full reader would silently
move subsequent citation IDs. The bounded JSON envelope remains 65,536 characters
(at most four UTF-8 bytes per character), with ciphertext bounds checked before
decryption and decoding. Full and indexed readers emit one original physical
fragment per stored page. These internal legacy fragments are not model windows.

New source writers independently require consistent unit ordinals/metadata,
Boolean final markers with empty text, complete termination, and text fragments
no larger than 4,096 characters before sealing. Only when copying an authenticated
legacy prefix into a NEW child attempt are inherited fragments split. The parent
ciphertext and all existing references remain unchanged.

The cited planner retains physical coordinates and partitions text into windows
of at most 3,000 characters. Descriptor offsets can address the existing bounded
legacy envelope; window/prefix lengths remain bounded. The direct whole-page
model-input API still refuses pages over 4,096 characters. Cited egress screens
the entire authenticated unit and the exact bounded window, not just its excerpt.
This does not authorize any inference, provider-policy change or private egress.

## Evidence and scope

A read-only check of the existing full-backup stage found a sealed attempt with
433 pages and 412 source units. One page contained 4,097 characters. All page
authentication, record shape, physical sequence, metadata and final markers
matched; a fresh authenticated original-source parse produced identical full
unit metadata and text digests. Database identity was unchanged. The historical
lexer fix in 035b863 corrected escaped-character flushing without changing the
version-one source semantics. No transcript text was printed or persisted.

Synthetic regressions reject malformed sequences/metadata/finals and invalid new
writers, preserve portable sealed rows and references, check legacy physical
coordinates followed by another text page, validate citations after offset 4,096,
deny a secret elsewhere in the unit, and append a bounded child without changing
its parent. Existing source, ledger, cited and paid recovery proofs supplement
these tests. Independent review rejected reader-side renumbering and cleared the
physical-page-preserving alternative. Live completion remains a separate proof.
