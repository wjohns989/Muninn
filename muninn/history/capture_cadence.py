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
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.quiet_seconds = _duration(quiet_seconds, "quiet_seconds")
        self.interval_seconds = _duration(interval_seconds, "interval_seconds")
        if not callable(clock):
            raise ValueError("clock must be callable")
        self._clock = clock
        self._lock = threading.Lock()
        now = self._now()
        self._last_activity = now
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
        """Planning depends on quiet time, not model-attempt cooldown."""
        with self._lock:
            now = self._now()
            return max(0.0, now - self._last_activity) >= self.quiet_seconds

    def analysis_ready(self) -> bool:
        """Analysis requires quiet time and an elapsed attempt interval."""
        with self._lock:
            now = self._now()
            quiet = max(0.0, now - self._last_activity) >= self.quiet_seconds
            cooldown = (self._last_attempt is None
                        or max(0.0, now - self._last_attempt) >= self.interval_seconds)
            return quiet and cooldown

    def note_attempt(self) -> None:
        """Start the cooldown when the owning worker begins an analysis attempt."""
        with self._lock:
            self._last_attempt = self._now()

    def snapshot(self) -> dict[str, bool | float]:
        """Return timing-only status; never exposes source or capture identity."""
        with self._lock:
            now = self._now()
            quiet_elapsed = max(0.0, now - self._last_activity)
            quiet_remaining = max(0.0, self.quiet_seconds - quiet_elapsed)
            if self._last_attempt is None:
                cooldown_remaining = 0.0
            else:
                attempt_elapsed = max(0.0, now - self._last_attempt)
                cooldown_remaining = max(0.0, self.interval_seconds - attempt_elapsed)
            planning_ready = quiet_remaining == 0.0
            analysis_ready = planning_ready and cooldown_remaining == 0.0
            return {
                "planning_ready": planning_ready,
                "analysis_ready": analysis_ready,
                "quiet_remaining_seconds": quiet_remaining,
                "attempt_cooldown_remaining_seconds": cooldown_remaining,
            }
