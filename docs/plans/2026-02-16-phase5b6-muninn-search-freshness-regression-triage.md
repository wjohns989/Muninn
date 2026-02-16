# Phase 5B.6: Muninn Search Freshness Regression Triage

Date: 2026-02-16  
Status: Discovered (triage queued)

## Issue Summary

During Phase 5B.5 continuity-memory seeding, newly added memories were persisted (`get_all_memories`) but not returned by `search_memory` in the same session, including exact-keyword queries.

## Observed Evidence

1. Added continuation memories successfully via `add_memory`:
   - IDs include:
     - `ea551e71-4b1b-49cc-9884-1c025d6ed3ba`
     - `56d7309d-bc39-4633-a336-a2a3bd2b4253`
     - `4f44e066-334f-4e3f-bc09-71ddd0e7ad67`
     - `4b5a0a6b-85de-443c-8130-a50dd303ce13`
     - `08f2d889-c315-4790-976c-d7fff102cb30`
2. Persistence verification:
   - `get_all_memories` returns the inserted continuation records.
3. Retrieval anomaly:
   - `search_memory` returned no results for exact-keyword queries targeting newly inserted continuation entries.
   - `search_memory` still returns older episodic records, indicating non-global failure.

## Risk

1. Fresh-memory discoverability risk for immediate handoff workflows.
2. Potential mismatch between persistence path and searchable index refresh path.

## Short-Term Mitigation

1. Continue storing continuity instructions in:
   - `docs/AGENT_CONTINUATION_RUNBOOK.md`
   - roadmap/phase docs.
2. Keep persisted continuation records in Muninn (`get_all_memories`) as fallback.

## Next Triage Actions

1. Reproduce locally through direct API path (`add` then `search`) with controlled metadata permutations.
2. Inspect indexing/refresh path between add-memory write and hybrid retrieval inputs.
3. Add regression test covering immediate post-add search visibility for continuity-style entries.
4. If required, add optional forced-index-refresh flag for high-priority continuity writes.
