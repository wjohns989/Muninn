# Batch progress visibility without new dispatch authority

Changed dependency: the existing batch worker's authenticated status and the
local dashboard now distinguish provider progress from successful polling.

The observational health sample exposes only counters, timestamps and static
labels: last successful GET, consecutive poll errors, provider creation age,
24-hour deadline, completed/failed/total requests, observation basis, and an
explicit `internal_provider_activity: unknown`. Missing or malformed metrics
remain unknown, never zero. Zero reported outcomes use the provider creation
time, so a Muninn restart cannot reset that age warning. Nonzero progress timing
is explicitly based on current-process observations, not claimed durable history.

One hour without reported progress warns `degraded_unknown`; more than three
minutes since the last successful GET warns `polling_stale`. Crossing the stated
completion allowance warns `deadline_overdue`, not invented provider failure.
Provider terminal state does not prove local identity, billing, citations or
publication passed. Transport and identity errors, including recovery candidates
and repair-only requests, reach the parent health sample. No raw error text or
private request/result content enters this status.

Submission, consent, paid fences, recovery ownership, serial checkpoints and
failed-only repair limits are unchanged. No health condition starts another POST,
cancels/deletes a batch, changes a route, or declares backlog completion. The
dashboard uses fixed labels and text-only sinks, shows observed counts/age/poll
time/deadline, and explicitly says results are not streaming.

Validation: 50 affected worker/repair/gathering/dashboard checks passed in 47.52s.
Independent review then identified a candidate-ID mismatch error-visibility gap;
the new mismatch regression failed before its one-line correction. Final eight
health checks passed in 5.32s. Review CLEAR on the corrected source/test shape;
the parent observed actual test completion. Earlier component proofs are not
claimed as a new full suite. Live installation and rendered current-provider
evidence are separate follow-up gates.

Current provider read at implementation time: the retained 60-request batch
reported `in_progress`, zero completed, zero failed, no results or usage, after
about four hours. That is degraded/unknown progress, not proof of a dead API or
successful inference. No diagnostic model request was launched.

Reference: [OpenRouter batch status/results contract](https://openrouter.ai/docs/batch-quickstart).

## Installed proof

The owned reload of candidate `9459b6f` passed its existing paid-stop fence,
seven encrypted database preimages and zero-inflight-job check. One exact
Miniconda/canonical-repo service PID 18936 (creation epoch 1791430246.7302876)
owns port 42069. Capture/remote settings were preserved, not reauthorized or
changed; four full restart preimages remain protected while incremental restart
compaction operates separately.

Authenticated history status returned 200 and anonymous access 401. The current
batch stayed `awaiting_provider`, with 60 requests/windows. Its health is
`degraded_unknown`: provider `in_progress`, completed 0, failed 0, total 60,
last successful poll 1791430275.2363436, zero consecutive polling errors,
age 15535.1 seconds, deadline 1791501172. The warning survived restart because
its zero-outcome age comes from provider creation, not service uptime.

The isolated live Edge check passed operating-status and actual window-count
comparison (11209 recorded capture jobs). Parent inspection of the rendered
screen confirmed readable warning, age 4.3 hours, counts 0/60, last-poll UTC and
deadline UTC. No token, transcript text or credential value is displayed. The
check did not issue transcript search, credential reveal or model inference.
The screenshot is a temporary local status artifact, not a portable recovery
receipt. Backlog completion and automatic alternative routing remain unproven:
this change supplies honest visibility, not permission to bypass checkpoints.

The existing ten-minute heartbeat was updated through the application tool, not
by rewriting scheduler files. Its exact policy preimage was copied and SHA-256
checked first. Read-back confirmed the same ID/kind/name/status/cadence/target
and the complete prior prompt preserved as a prefix. Added instructions report
new degraded/unknown, polling and deadline warnings without repeated unchanged
notifications, provider dispatch, cancellation or checkpoint bypass.

This reload's existing restart-pool compaction completed in 47.1s. It retired
one older 3266338816-byte source-evidence preimage only after full reconstruction,
hash and database recovery checks, adding 114322440 bytes of encrypted chunks.
Four independent full restart preimages remain protected; the retired copy is
recoverable through its retained manifest and pool. No batch or standalone full
backup was deleted. This measured reduction is separate from the unfinished
whole-bundle migration.
