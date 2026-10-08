"""Non-authoritative batch telemetry. Never authorizes retries or publication."""
from __future__ import annotations

import math

WARNING_SECONDS = 3600
STALE_POLL_SECONDS = 180


class BatchHealth:
    def __init__(self):
        self.created = self.last_poll = self.last_change = None
        self.counts = None
        self.provider_state = None
        self.errors = 0

    def observe(self, reply, *, expected, now, poll=True):
        created = reply.get('created_at')
        self.created = (created if type(created) in (int, float) and math.isfinite(created)
                        and 0 <= created <= now + 60 else None)
        self.provider_state = reply.get('status')
        if poll:
            self.last_poll = now
            self.errors = 0
        raw = reply.get('request_counts') or {}
        values = tuple(raw.get(k) for k in ('total', 'completed', 'failed')) if isinstance(raw, dict) else ()
        valid = (len(values) == 3 and all(type(v) is int and v >= 0 for v in values)
                 and values[0] == expected and values[1] + values[2] <= expected)
        current = values if valid else None
        if current is not None:
            if self.counts is None or current[1:] != self.counts[1:]:
                self.last_change = now
            if current[1:] == (0, 0) and self.created is not None:
                self.last_change = self.created
        self.counts = current

    def poll_failed(self):
        self.errors += 1

    def snapshot(self, now):
        deadline = self.created + 86400 if self.created is not None else None
        age = max(0, now - self.created) if self.created is not None else None
        unchanged = max(0, now - self.last_change) if self.last_change is not None else None
        if self.provider_state in {'completed', 'failed', 'expired', 'cancelled'}:
            state = 'provider_' + self.provider_state
        elif self.errors:
            state = 'polling_error'
        elif self.last_poll is not None and now - self.last_poll > STALE_POLL_SECONDS:
            state = 'polling_stale'
        elif deadline is not None and now > deadline:
            state = 'deadline_overdue'
        elif self.counts is None:
            state = 'metrics_unknown'
        elif unchanged is not None and unchanged >= WARNING_SECONDS:
            state = 'degraded_unknown'
        elif self.counts[1] + self.counts[2] > 0:
            state = 'progress_observed'
        else:
            state = 'awaiting_progress'
        return {'state': state, 'provider_state': self.provider_state,
                'last_successful_poll_at': self.last_poll, 'consecutive_poll_errors': self.errors,
                'age_seconds': age, 'deadline_at': deadline,
                'unchanged_seconds': unchanged, 'warning_after_seconds': WARNING_SECONDS,
                'completed_requests': self.counts[1] if self.counts else None,
                'failed_requests': self.counts[2] if self.counts else None,
                'total_requests': self.counts[0] if self.counts else None,
                'progress_basis': ('provider_creation_zero_reported_outcomes'
                                   if self.counts and self.counts[1:] == (0, 0) and self.created is not None
                                   else 'current_process_observations'),
                'internal_provider_activity': 'unknown'}
