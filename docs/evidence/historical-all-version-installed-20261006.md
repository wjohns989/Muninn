# Installed historical all-version enrollment

Exact installed code candidate: `01c1607ae6711606ab2be07281979f47c163d232`.
Focused validation and design evidence:
`historical-all-version-validation-20261006.md` and
`../architecture/historical-all-version-enrollment-20261006.md`.

## Actual installation and recovery safeguards

The existing Muninn-only reload helper accepted the committed candidate, validated
six encrypted database preimages and observed zero in-flight jobs before stopping
the owned process. Startup verified one listener on port 42069, strict archive
readiness and preserved capture flags. No other service was restarted. The existing
backlog-drain deadline was preserved, not renewed.

After startup, an additional ACL-protected online capture-journal preimage was
created and integrity checked. It includes the new additive table and precedes
all-version enrollment. Retained local location:
`.muninn_runtime/history_secure_archive/operator-preimages/20261006T063453-all-versions-enrollment/`.
It is an encrypted-journal component preimage, not a new whole-archive/vault backup.

## Actual enrollment and identity proof

Bounded explicit CLI enrollment used `recover=False`, limit 128 and at most 100
metadata steps; no model call was made by this command. Its final verifier
authenticated the selected union of historical receipts/grants.

| Original pin 2786 | Snapshot versions |
| --- | ---: |
| Visited | 6603 |
| Newly queued | 2124 |
| Existing index records | 3631 |
| Excluded provider/kind | 848 |

Traversal finished across the original 4229 source positions. Before/after byte
equality preserved the original latest sealed cursor and all 2457 preexisting
sealed grants. The authenticated retained active batch's request-body digest and
provider identity also remained unchanged. No batch was deleted, cancelled or
resent by the enrollment workflow. No spending/privacy policy was changed.

## Fresh installed HTTP proof

Sole listener owner PID 75896 uses the existing miniconda interpreter. Health 200;
anonymous protected request 401; authenticated protected/history requests 200;
no auth token in anonymous root. Strict archive readiness and automatic remote-only
processing remained true. The authenticated status now exposes the completed
all-version enrollment separately from the unchanged latest-v1 enrollment.

At this check: pending snapshot source versions 6067; all-capture-lane window-job
index total 7095, with 2532 succeeded, 501 reused, 842 failed, four outcome-unknown,
128 pending and 3088 privacy-parked retry jobs. These are operational hints, not a
new independent publication census. Existing batch: 28 windows in 20 provider
requests, awaiting provider. All-version enrollment did not increase these window
completion counts. Normal later-capture reconciliation remains active.

## Claim boundary

This proves installation, metadata enrollment and preserved identities, not full
backlog interpretation. Pending sources are snapshot versions, not unique files
or windows. There is no fixed source-to-window/batch conversion; final window
total and overall percentage are unknown until planning finishes. New eligible
work remains behind the existing serial paid-checkpoint, privacy and $5/day,
$50/month controls. No inferred reverse-version reuse savings are claimed.

Independent actual-results examination was CLEAR for the enrollment/preservation
claim, with endpoint/status verification initially UNKNOWN while running. The
fresh installed HTTP results above close that dependent check. The final independent
review disposition was CLEAR for endpoint/auth/archive/status and the scoped
enrollment/preservation proof. The overall local-completion
goal remains active: model backlog, credential ambiguity and other acceptance
gaps are not resolved by this component proof.
