# Live capture reconciliation checkpoint

The running authenticated strict installation reported a retryable SQLite
operation in its last reconciliation status. This is not an integrity-error
diagnostic. The existing reconciler catches OperationalError without exposing
database paths, and periodic capture discovery continues independently.

## Authenticated local check and bounded action

- A query-only journal reader authenticated baseline 1612 and the sealed cursor:
  after generation 2822, through generation 3206, source index 161, version index 0.
- The pinned generation-3206 manifest authenticated and its cursor position was
  valid. The current archive generation was 3214. SQL changes in that read were 0.
- One existing reconcile_enrichment(limit=128) invocation completed, queued 0 new
  receipts and advanced the cursor to source index 206, version index 1. It made
  no model call or manual batch submission. It did not finish all reconciliation.

## Invocation mistake and bounded follow-up

The one-off invocation incorrectly constructed CaptureJournal with its default
recover=True. That startup default can reset live raw-capture claims and must not
be used by a cooperating live maintenance caller. Both persistent operator writers
already specify recover=False (enroll_history_backlog.py and
recover_capture_windows.py); no speculative source change is needed.

There was no immediately-before census of raw claims, so this check cannot prove
that no claim was transiently reset. Afterward a query-only journal census reported
3470 archived raw captures, 2 unavailable, no pending or active raw capture, and
no active analysis leases. The existing 4 outcome_unknown jobs and 842 failed jobs
were unchanged. No recent raw-capture updated_at records were observed within
the inspected 180-second interval. Analysis startup recovery only touches expired
leases; it does not revoke a live analysis lease. These counts are operational
index hints, not independently validated publication or archived-byte coverage.

The owned service remained healthy and the existing 28-window/20-request batch
remained awaiting_provider with the original active drain deadline. No deletion,
batch cancellation, credential readout, provider-policy change or service restart
was performed. These follow-up checks do not establish complete archive recovery,
all historical interpretation, batch billing or absence of every transient effect.

## Next dependency

Existing serial batch work continues under its current authorization. Credential
ambiguity review is still a separate live worker awaiting a hidden local passphrase.
The service's last-reconciliation warning records its previous failed attempt;
the independent bounded invocation does not rewrite that in-memory status.
