# Historical all-version enrollment: candidate proof

This closes a concrete enrollment gap, not the interpretation backlog itself.
At the authenticated generation-2786 pin there are 6603 snapshot versions, 5755
eligible transcripts; 2124 older pre-watermark eligible versions had no enrichment
index entry. Index presence is not verified model publication or reuse coverage.

## Isolated validation

- Initial four new checks failed against the missing APIs, before implementation.
- Initial affected run: 59 passed in 39.40 seconds.
- Expanded affected run: 173 passed and one failed in 143.69 seconds. The failure
  was a new invalid fixture using a provider the archive deliberately rejects.
- Only the fixture provider was corrected to the accepted archive provider
  `export`, which remains excluded from transcript enrichment. No production
  archive validation was weakened. All 14 tests in
  `tests/test_capture_historical_versions.py` then passed in 7.40 seconds.
- The unchanged passes cover historical enrollment, capture enrichment/service,
  batch activation/jobs/worker/packing/repair and existing window reuse. This is
  focused affected validation, not a full pytest or merge-readiness claim.

New checks exercise older-only content, bounded resume, atomic interruption
rollback, stale-writer CAS, portable restore, completed-latest prerequisite,
tampered cursor/position/count/pin rejection, actual older-version batch selection
without provider POST, preservation of old grants/latest seal/paid owner/items,
query-only no-schema/no-write preview, excluded provider and empty-source bounds.

Independent design and actual named-source diff review: CLEAR. The final isolated
fixture change was reviewed separately as CLEAR, without duplicating passing tests.

## Actual read-only preview

The installed archive was unlocked locally using its existing configuration;
neither key/passphrase nor transcript content was printed. An explicit
`--all-versions --limit 128` preview against the same original pin reported 128
snapshot versions examined, 65 would queue, 62 already indexed, one excluded.
It did not initialize schema, enroll receipts, run inference or submit a batch.

## Installation gates and limits

Use the exact committed candidate and the existing independently reviewed
single-service reload helper. Preserve the running paid checkpoint, requests,
grants, authentication, remote-only policy and spending limits. After startup,
take a fresh ACL-protected online journal preimage and verify it before bounded
`recover=False` enrollment. The earlier reload preimages precede the new schema
and are not a substitute for this post-startup preimage.

The added pass traverses all versions at the original pin, not a replacement pin.
Its completion means metadata enrollment only. Normal post-watermark reconciliation
handles later captures. No global chronological coverage, fixed conversion from
versions to windows, reverse-version deduplication savings or overall completion
percentage is claimed. Genuinely uncovered work stays subject to existing serial
paid-checkpoint, privacy and $5/day/$50/month controls.

Live installation/enrollment verification remains a separate dependent proof.
