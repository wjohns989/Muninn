# Larger-context Luna packing: bounded proof

Candidate: `88e24259c9b49db9cc909fec0ff72282ac7daee1`.

- Initial new-contract test failed because packing was absent. The unchanged
  legacy fixture uses eleven provider requests for eleven windows; the new
  authenticated-plan fixture uses two requests for the same eleven windows.
- 126 focused tests passed in 92.32 seconds across packing, legacy batches,
  ownership, worker/recovery, failed-only repairs and activation. After the
  final duplicate-JSON-field rejection, all 20 packing tests passed in 18.90
  seconds. Unaffected legacy proofs are reused, not claimed as a full suite.
- The actual encrypted-journal fake-provider test submitted two windows as one
  request, restarted while awaiting the reply, published the valid sibling,
  sent only the bad window as a one-request unpacked child, then durably passed
  the parent. Two aggregate fixture bills of $0.012 summed to $0.024 once each.
- Portable restore preserved the packed wire plan and authenticated repair
  linkage. Invalid metadata, plan/scope mismatch, gaps, missing/duplicate slots,
  false quotes and duplicate JSON fields are tested. The activation byte-bound
  fallback retains the same selected legacy items.
- Independent native review cleared the source change, then the exact shared
  service reload and bounded first authorized serial checkpoint as live
  validation. Its duplicate-field hardening note was fixed and tested. Provider
  compatibility/quality and whole-goal readiness remain unproved. A provider-level
  rejection or unknown identity/bill can hold the checkpoint without automatic
  repair; only eligible item-output failures are promised failed-only repair.

## Immutable retained work before reload

The existing parent `9ad42b8e71d9427cae6f8d763e6c46dc` is terminal-saved with
49 original windows / 49 provider requests. The service reported 48 publications
and one repair. Its item digest is
`bc45537cf396851c8a69e3c29e00017ab4f58c8c6be3ff95e13b9a047fb6d60a`;
exact wire digest is
`06144bc5fb590caccd90839b26f625eec53f1a3a179dd9e2c1690d828dda801e`.

The submitted one-window / one-request child
`7860f399e047415a95b74c4af2e50e78` has item digest
`44740ea850af16f871b90e65bb08e557ee6da674d354880ea68bfa14da198336`
and wire digest
`b6c2bcf127f691b68e29e63b7e85b1a755a97d194c076b77f8adc99a32f8f305`.
Digests use the existing `_json(items)` and `_wire_json(payload(items))`.
No transcript, secret, passphrase, or provider result text is recorded here.

## Installed verification

The existing `reload_shared_local.py` procedure installed the exact candidate,
preserved capture settings and verified six encrypted database preimages with
zero in-flight durable jobs. These six preimages cover the helper's selected
journal/SQLite stores, **not** a complete `historical-batches.db` preimage.
No migration, batch deletion/cancellation or paid resubmission was performed by
this installation.

After reload: one existing-installation process/listener on port 42069; health
200; anonymous protected request 401; authenticated protected request 200;
strict archive ready; automatic capture/analysis/remote-only settings retained.
Both parent and child item/wire digests above match exactly after reload, as do
their provider IDs. Parent remains terminal-saved; child remains submitted.
Live status distinguishes one repair window/one provider request from its
49-window/49-request parent. The current known checkpoint was not repacked.

The existing active ten-minute cost monitor was updated, not duplicated, to
count actual packed root requests separately from windows, preserve per-window
publication/repair denominators, settle aggregate costs once, and report measured
live compatibility rather than infer speed or savings from request reduction.

## Remaining proof at installation

This record does not establish live packed provider acceptance, inference
quality, turnaround improvement, measured savings, complete historical coverage
or whole-installation readiness. The first newly prepared authorized serial
checkpoint must supply that representative acceptance/citation/billing evidence.
An awaiting known repair is not completion or permission to resend its parent.

## First live packed results, observed 2026-10-06

The next packed parent `941f312ed3224a39ae0fa4956fb787ac`, provider batch
`batch-1791257569-HAudpeQ5KYvsSGmBgokW`, contains 34 transcript windows in
16 root requests. Its retained authenticated terminal response is completed
with 16 completed / 16 total provider requests and zero transport failures.
The exact terminal custom-request ID mapping and original member bindings
passed. Its finite non-BYOK aggregate provider charge is **$0.01298015**;
this is one parent bill, not 34 individual bills.

The parent's exact owner- and generation-2-bound local admission is settled with resolution
`response` and **$0.012981**, reflecting the ledger's upward rounding to whole
microdollars. This matches the retained provider charge at that precision.
That check also used query-only SQL and reported zero changes.

Independent review cleared the scoped partial-acceptance interpretation, keeping
the child pending and speed/savings unproved. Its remaining settlement question
was resolved by the exact owner/generation/resolution/amount comparison above;
it is not a claim that the child's bill or the whole parent checkpoint passed.

The journal reports 33 succeeded windows and one pending repair window. A
read-only cross-check of those 33 successful stages and encrypted publication
receipts passed: their expected references match the original authenticated
window descriptors, exact quotes/proposals and model identity. The verified
ledger contains all corresponding references, spanning **34 durable cited
candidate entries**. These entries remain source observations/provisional
interpretations, not independently certified truths.

The check used `CitedAnalysisSource(..., read_only=True)`, including read-only
ledger and source-evidence stores, and a query-only journal connection built
without its initializing constructor. The encrypted ledger chain was checked
once in a pinned reference reader. The journal reported zero SQL changes.
No private source, credential value or model-result text was printed or saved.

Only the remaining window is in the linked first repair
`1294ea083ffd470fb85146f54fad25c9`, provider batch
`batch-1791258769-CvCbDaMSEjxc7Kg4TyKl`. Its actual retained plan is one
window / one unpacked provider request. Provider GET reports `in_progress`,
zero completed / one total request, and unknown usage cost. Successful siblings
are retained, not resent. The parent checkpoint is **not complete** until the
repair passes and exact settlement/publication checks allow advancement.

Packing used about 53% fewer requests than a one-request-per-window layout
(16 versus 34). This proves representative live provider compatibility and
cited durable publication, **not** measured turnaround improvement or monetary
savings. Whole-backlog coverage and whole-installation readiness remain open.
