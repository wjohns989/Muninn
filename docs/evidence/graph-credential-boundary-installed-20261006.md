# Installed ordinary graph credential boundary

Exact installed candidate: `b43753a3cd6f5b2bcdbe48b9e48928001b55938a`.
Design/diff and changed-issue independent review cleared the scoped boundary.
Final focused validation completed: 72 passed in 2.92 seconds, one unrelated
installed Hugging Face environment-variable deprecation warning. This is not
full-suite or merge-readiness proof. See the companion validation note and ADR.

## Installation

The existing reviewed Muninn-only reload helper verified this committed candidate,
validated six encrypted database preimages and found zero in-flight jobs before
the owned stop. Startup verified exactly one strict/archive-ready listener on
42069. No graph schema upgrade, ordinary-record remediation or other-service
restart was performed by this change. Existing configuration and drain deadline
were preserved. Those six component preimages are not a complete graph/batch/
archive restore drill and are not claimed as one.

## Representative actual HTTP proof

- Actual `/graph`: anonymous 401, main authenticated 200 with `no-store`.
- The default user returned 100 real entities. Their JSON was inspected only in
  local process RAM with the shared credential guard; it passed. Entity content,
  keys and passphrases were never printed or used as model inputs.
- A fresh nonexistent-user filter returned 200, zero entities and `no-store`.
  This tests real handler forwarding and real store filtering, not only a mock.
- Runtime probe: sole listener PID 50096, health 200, strict archive ready,
  authenticated history 200 and automatic remote-only processing unchanged.
- All-version enrollment remained complete at the original pin 2786 with
  6603 visited = 2124 queued + 3631 existing + 848 excluded.
- Existing retained paid checkpoint remained awaiting provider: 28 windows in
  20 requests. Drain remained active with 4310 seconds at the check, not renewed.
  One credential-triage worker remained active. These are observations, not a
  claim that private ambiguity, provider billing or model backlog completed.

## Scope and remaining limits

Source projections/guards introduce no storage rewrite or deletion. Synthetic
fixtures directly prove original-row nonmutation; this live HTTP probe is not a
byte-level census of the entire original graph. Known-format detected values are
guarded; arbitrary unlabeled or novel credentials are not certified safe. Legacy
plaintext originals and prior vector/graph copies still require separate reviewed
coordinated vault migration with hidden local unlock. Federation authorization,
automatic legacy schema migration and full-store recovery were not reviewed here.

The checklist emphasized complete input validation, critical output sinks and
error non-disclosure. Actual-diff review additionally caught canonical ID handling
and numeric-annotated inputs; these were fixed and tested before installation.
Independent actual-results examination was CLEAR for these scoped installation,
authentication, filtering, no-store and detected-value read-boundary claims.
The overall local-completion goal remains active, not redefined around this slice.
