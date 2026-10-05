"""In-memory timing gate for the opt-in local capture-analysis lane.

This helper has no persistence, task, environment, or provider behavior. A new
instance starts with a full quiet grace period so a service restart cannot make
recent capture activity immediately eligible.
"""

from __future__ import annotations

import math
import threading
import time
from numbers import Real
from typing import Callable


_MAX_SECONDS = 86400.0


def _duration(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number of seconds")
    result = float(value)
    if not math.isfinite(result) or not 0 < result <= _MAX_SECONDS:
        raise ValueError(f"{name} must be greater than zero and at most 86400 seconds")
    return result


class SmallCaptureCadence:
    """Track startup/activity quiet time and a separate analysis-attempt cooldown."""

    def __init__(self, *, quiet_seconds: float = 300,
                 interval_seconds: float = 30,
                 max_wait_seconds: float = 1800,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.quiet_seconds = _duration(quiet_seconds, "quiet_seconds")
        self.interval_seconds = _duration(interval_seconds, "interval_seconds")
        self.max_wait_seconds = _duration(max_wait_seconds, "max_wait_seconds")
        if self.max_wait_seconds < self.quiet_seconds:
            raise ValueError("max_wait_seconds must be at least quiet_seconds")
        if not callable(clock):
            raise ValueError("clock must be callable")
        self._clock = clock
        self._lock = threading.Lock()
        now = self._now()
        self._last_activity = now
        self._last_attempt_or_start = now
        self._last_attempt: float | None = None

    def _now(self) -> float:
        try:
            value = self._clock()
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError
            now = float(value)
            if not math.isfinite(now):
                raise ValueError
            return now
        except (TypeError, OverflowError, ValueError) as exc:
            raise ValueError("clock must return a finite numeric value") from exc

    def note_activity(self) -> None:
        """Reset quiet time after the caller confirms an accepted capture."""
        with self._lock:
            self._last_activity = self._now()

    def planning_ready(self) -> bool:
        """Plan after quiet time or bounded delay, independent of model cooldown."""
        with self._lock:
            now = self._now()
            return (max(0.0, now - self._last_activity) >= self.quiet_seconds
                    or max(0.0, now - self._last_attempt_or_start) >= self.max_wait_seconds)

    def analysis_ready(self) -> bool:
        """Analysis requires quiet time and an elapsed attempt interval."""
        with self._lock:
            now = self._now()
            ready = (max(0.0, now - self._last_activity) >= self.quiet_seconds
                     or max(0.0, now - self._last_attempt_or_start) >= self.max_wait_seconds)
            cooldown = (self._last_attempt is None
                        or max(0.0, now - self._last_attempt) >= self.interval_seconds)
            return ready and cooldown

    def note_attempt(self) -> None:
        """Start the cooldown when the owning worker begins an analysis attempt."""
        with self._lock:
            now = self._now()
            self._last_attempt = now
            self._last_attempt_or_start = now

    def attempt_ready(self) -> bool:
        """Keep the attempt interval when an explicit catch-up bypasses quiet time."""
        with self._lock:
            return (self._last_attempt is None
                    or max(0.0, self._now() - self._last_attempt) >= self.interval_seconds)

    def snapshot(self) -> dict[str, bool | float]:
        """Return timing-only status; never exposes source or capture identity."""
        with self._lock:
            now = self._now()
            quiet_elapsed = max(0.0, now - self._last_activity)
            quiet_remaining = max(0.0, self.quiet_seconds - quiet_elapsed)
            max_wait_elapsed = max(0.0, now - self._last_attempt_or_start)
            max_wait_remaining = max(0.0, self.max_wait_seconds - max_wait_elapsed)
            if self._last_attempt is None:
                cooldown_remaining = 0.0
            else:
                attempt_elapsed = max(0.0, now - self._last_attempt)
                cooldown_remaining = max(0.0, self.interval_seconds - attempt_elapsed)
            planning_ready = quiet_remaining == 0.0 or max_wait_remaining == 0.0
            analysis_ready = planning_ready and cooldown_remaining == 0.0
            return {
                "planning_ready": planning_ready,
                "analysis_ready": analysis_ready,
                "quiet_remaining_seconds": quiet_remaining,
                "max_wait_remaining_seconds": max_wait_remaining,
                "attempt_cooldown_remaining_seconds": cooldown_remaining,
            }


class CaptureBacklogDrain:
    """Temporary remote catch-up; expiry survives reload without extending authority."""

    HALT_REASONS = {"remote_cost_unresolved", "remote_consent_revoked",
                    "remote_admission_threshold_reached", "daily_zdr_cap_unverified",
                    "remote_accounting_unavailable", "remote_accounting_unconfigured"}

    def __init__(self, deadline: float = 0, *, clock: Callable[[], float] = time.time):
        now = clock()
        if (isinstance(deadline, bool) or not isinstance(deadline, Real)
                or not math.isfinite(deadline) or deadline < 0
                or deadline > now + 10800):
            raise ValueError("Backlog drain must expire within three hours")
        self.deadline = float(deadline)
        self._clock = clock
        self.halted_reason: str | None = None

    def active(self, *, pending: int, remote_enabled: bool) -> bool:
        return (remote_enabled is True and type(pending) is int and pending >= 100
                and self.halted_reason is None and self._clock() < self.deadline)

    def halt(self, reason: str) -> None:
        if reason in self.HALT_REASONS:
            self.halted_reason = reason

    def snapshot(self) -> dict[str, bool | float | str | None]:
        remaining = max(0.0, self.deadline - self._clock())
        return {"enabled": remaining > 0 and self.halted_reason is None,
                "remaining_seconds": remaining, "halted_reason": self.halted_reason}
