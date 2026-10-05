# Pre-send accounting recovery

## Changed property

A failed, refused or cancelled journal cleanup callback previously left a
pre-send `reserved` admission blocking all later paid work. Temporary contention
also permanently halted the temporary catch-up scheduler.

The transport now releases only `reserved` before awaiting journal cleanup.
Reservations older than 900 seconds are fenced and released within the next
admission transaction. The old worker cannot win `mark_unknown()` and therefore
cannot POST. Unknown billing states never expire; settled charges and budget
floors are unchanged. Contention remains a bounded retry rather than a permanent
catch-up halt.

## Retained proof

- Seven added regressions failed before implementation; 44 existing cases passed.
- Independent inspection cleared the design and actual repair diff.
- The old lease-exception assertion intentionally expected a stranded reservation;
  it now distinguishes the transport's reserved billing state from queue ambiguity.
- The final affected suite passed 85 cases in 13.25 seconds:
  `test_remote_accounting`, `test_remote_admission_transport`,
  `test_capture_backlog_drain`, `test_secure_history_analysis`, and
  `test_secure_analysis_journal`.
- Added transport cancellation checks cover successful queue marking both before
  and after `mark_unknown()`. A false cleanup proof releases reserved only, while
  an unknown marker still blocks. Other cases retain uncertain POST outcomes,
  missing costs, policy revocation, parallel admission exclusion and cap floors.

All inference and cost probes in these tests are isolated temporary fixtures.
This proof does not show that a live installation has loaded the repair, that a
historical backlog is complete, or that batch transport has been implemented.

## Batch pivot boundary

Historical batch processing requires separate explicit temporary-retention
consent and no-training provider verification. It must not weaken normal live
ZDR settings. A separate encrypted batch outbox must durably retain submission
identity, item/result binding, aggregate billing, and deletion recovery. An
uncertain submission cannot be resubmitted; results and cost must be saved
locally before requesting deletion. These requirements remain implementation
work, not properties proved by this accounting change.
