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
