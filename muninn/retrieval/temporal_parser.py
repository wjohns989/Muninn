"""
Muninn Temporal Query Parser
-----------------------------
Natural-language time-phrase extraction and resolution for metadata-filtered retrieval.

Parses phrases like "last 7 days", "yesterday", "in January", "this week" from
free-text queries and converts them to Unix timestamp ranges (start, end).

Used by HybridRetriever to expand temporal search when the
`temporal_query_expansion` feature flag is enabled.

Design Principles:
- Pure regex/rule-based: no NLP dependencies required
- Thread-safe: stateless (all state in local variables)
- Conservative: only emits a range when confidence is high; returns None on ambiguity
- Strips matched phrases from the query to avoid double-penalising temporal terms
  in BM25/vector search (matched phrase is returned separately)
"""

from __future__ import annotations

import calendar
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, List

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class TimeRange:
    """A resolved half-open temporal interval [start, end) in Unix seconds."""

    start: float  # inclusive lower bound (Unix timestamp, UTC)
    end: float    # exclusive upper bound (Unix timestamp, UTC)

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(
                f"TimeRange start ({self.start}) must be <= end ({self.end})"
            )

    def __str__(self) -> str:
        s = datetime.fromtimestamp(self.start, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        e = datetime.fromtimestamp(self.end, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return f"TimeRange({s} → {e})"

    def contains(self, ts: float) -> bool:
        return self.start <= ts < self.end


# ---------------------------------------------------------------------------
# Month name mapping (full + 3-letter)
# ---------------------------------------------------------------------------

_MONTH_NAMES: dict[str, int] = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

# ---------------------------------------------------------------------------
# Compiled regular expressions  (order matters — tried most specific first)
# ---------------------------------------------------------------------------

# "between Monday and Friday" or "between Jan 1 and Jan 7" are complex;
# we handle the simpler "between <date> and <date>" pattern for day-of-week.
_WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

_UNIT_SECONDS = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60,
    "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600,
    "day": 86400, "days": 86400,
    "week": 604800, "weeks": 604800,
    "month": 2592000, "months": 2592000,       # 30-day approximation
    "year": 31536000, "years": 31536000,        # 365-day approximation
}

# Build alternation strings for regex
_UNIT_PAT = "|".join(sorted(_UNIT_SECONDS.keys(), key=len, reverse=True))
_MONTH_PAT = "|".join(sorted(_MONTH_NAMES.keys(), key=len, reverse=True))
_WEEKDAY_PAT = "|".join(sorted(_WEEKDAYS.keys(), key=len, reverse=True))

# Ordered list of (pattern, handler_key) tuples.
#
# IMPORTANT: compound phrases (before/after + temporal noun) must appear
# BEFORE their constituent simple phrases to avoid the simple patterns
# consuming part of the compound match.  E.g. "before last week" must be
# tried before "last week"; "after yesterday" before "yesterday".
_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # --- Compound patterns first (most specific) ---

    # "between <weekday> and <weekday>" — current-week range
    (re.compile(
        r"\bbetween\s+(?P<wd1>" + _WEEKDAY_PAT + r")\s+and\s+(?P<wd2>" + _WEEKDAY_PAT + r")\b",
        re.IGNORECASE,
    ), "between_weekdays"),

    # "before last/past <unit>" — all history up to (now - unit)
    (re.compile(
        r"\bbefore\s+(?:last|past)\s+(?P<unit2>" + _UNIT_PAT + r")\b",
        re.IGNORECASE,
    ), "before_last_unit"),

    # "after last/past <unit>" — (now - unit) until now
    (re.compile(
        r"\bafter\s+(?:last|past)\s+(?P<unit3>" + _UNIT_PAT + r")\b",
        re.IGNORECASE,
    ), "after_last_unit"),

    # "after yesterday" — start of today until now
    (re.compile(r"\bafter\s+yesterday\b", re.IGNORECASE), "after_yesterday"),

    # --- Quantity + unit patterns ---

    # "last N days/weeks/months/hours/years" or "past N ..."
    (re.compile(
        r"\b(?:last|past)\s+(?P<n>\d+)\s+(?P<unit>" + _UNIT_PAT + r")\b",
        re.IGNORECASE,
    ), "last_n_units"),

    # "N days/weeks/months ago"
    (re.compile(
        r"\b(?P<n>\d+)\s+(?P<unit>" + _UNIT_PAT + r")\s+ago\b",
        re.IGNORECASE,
    ), "n_units_ago"),

    # --- Named period patterns (simple) ---

    # "last week" / "past week"
    (re.compile(r"\b(?:last|past)\s+week\b", re.IGNORECASE), "last_week"),

    # "last month" / "past month"
    (re.compile(r"\b(?:last|past)\s+month\b", re.IGNORECASE), "last_month"),

    # "last year" / "past year"
    (re.compile(r"\b(?:last|past)\s+year\b", re.IGNORECASE), "last_year"),

    # "last hour" / "past hour"
    (re.compile(r"\b(?:last|past)\s+hour\b", re.IGNORECASE), "last_hour"),

    # "this week"
    (re.compile(r"\bthis\s+week\b", re.IGNORECASE), "this_week"),

    # "this month"
    (re.compile(r"\bthis\s+month\b", re.IGNORECASE), "this_month"),

    # "this year"
    (re.compile(r"\bthis\s+year\b", re.IGNORECASE), "this_year"),

    # "yesterday"
    (re.compile(r"\byesterday\b", re.IGNORECASE), "yesterday"),

    # "today"
    (re.compile(r"\btoday\b", re.IGNORECASE), "today"),

    # "recent" / "recently"
    (re.compile(r"\brecent(?:ly)?\b", re.IGNORECASE), "recent"),

    # "in <MonthName>" e.g. "in January", "in Mar"
    (re.compile(r"\bin\s+(?P<month>" + _MONTH_PAT + r")\b", re.IGNORECASE), "in_month"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _start_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _end_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=23, minute=59, second=59, microsecond=999999)


def _start_of_week(dt: datetime) -> datetime:
    """Return Monday 00:00:00 of the week containing dt."""
    day_offset = dt.weekday()  # 0 = Monday
    return _start_of_day(dt - timedelta(days=day_offset))


def _start_of_month(dt: datetime) -> datetime:
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _start_of_year(dt: datetime) -> datetime:
    return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)


def _ts(dt: datetime) -> float:
    return dt.timestamp()


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------


class TemporalQueryParser:
    """
    Extract a time range from a natural-language query string.

    Usage::

        parser = TemporalQueryParser()
        result = parser.parse("auth issues from last week")
        if result:
            time_range, cleaned_query = result
            # cleaned_query → "auth issues from"
            # time_range.start, time_range.end → timestamps
    """

    def parse(
        self,
        query: str,
        reference_time: Optional[float] = None,
    ) -> Optional[Tuple[TimeRange, str]]:
        """
        Parse a temporal phrase from *query*.

        Args:
            query: The raw query string.
            reference_time: Unix timestamp to treat as "now". Defaults to
                ``time.time()``. Provided for deterministic testing.

        Returns:
            ``(TimeRange, cleaned_query)`` if a temporal phrase was found,
            ``None`` otherwise.  ``cleaned_query`` is the original query with
            the matched phrase removed and extra whitespace collapsed.
        """
        now_ts = reference_time if reference_time is not None else time.time()
        now = datetime.fromtimestamp(now_ts, tz=timezone.utc)

        for pattern, handler in _PATTERNS:
            m = pattern.search(query)
            if m is None:
                continue

            tr = self._resolve(handler, m, now, now_ts)
            if tr is None:
                continue

            # Strip matched span from query and collapse whitespace
            cleaned = (query[: m.start()] + " " + query[m.end() :]).strip()
            cleaned = re.sub(r"\s{2,}", " ", cleaned)
            return tr, cleaned

        return None

    # ------------------------------------------------------------------
    # Handler dispatch
    # ------------------------------------------------------------------

    def _resolve(
        self,
        handler: str,
        m: re.Match,
        now: datetime,
        now_ts: float,
    ) -> Optional[TimeRange]:
        try:
            return getattr(self, f"_handle_{handler}")(m, now, now_ts)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Individual handlers
    # ------------------------------------------------------------------

    def _handle_last_n_units(self, m: re.Match, now: datetime, now_ts: float) -> TimeRange:
        n = int(m.group("n"))
        unit = m.group("unit").lower().rstrip("s") + "s"  # normalise to plural
        # Look up seconds
        sec = _UNIT_SECONDS.get(unit) or _UNIT_SECONDS.get(m.group("unit").lower(), 86400)
        delta = n * sec
        return TimeRange(start=now_ts - delta, end=now_ts)

    def _handle_n_units_ago(self, m: re.Match, now: datetime, now_ts: float) -> TimeRange:
        """'N units ago' → a window centred around that point in time."""
        n = int(m.group("n"))
        unit = m.group("unit").lower()
        sec = _UNIT_SECONDS.get(unit, 86400)
        centre = now_ts - n * sec
        # Window = half the unit on each side, min 1 hour
        half = max(sec / 2, 3600.0)
        return TimeRange(start=centre - half, end=min(centre + half, now_ts))

    def _handle_last_week(self, m: re.Match, now: datetime, now_ts: float) -> TimeRange:
        """Previous calendar week (Mon–Sun)."""
        this_week_start = _start_of_week(now)
        last_week_start = this_week_start - timedelta(weeks=1)
        last_week_end = this_week_start
        return TimeRange(start=_ts(last_week_start), end=_ts(last_week_end))

    def _handle_last_month(self, m: re.Match, now: datetime, now_ts: float) -> TimeRange:
        """Previous calendar month."""
        first_of_this_month = _start_of_month(now)
        # Subtract one day from start of this month to land in last month
        last_day_of_prev = first_of_this_month - timedelta(days=1)
        first_of_prev = _start_of_month(last_day_of_prev)
        return TimeRange(start=_ts(first_of_prev), end=_ts(first_of_this_month))

    def _handle_last_year(self, m: re.Match, now: datetime, now_ts: float) -> TimeRange:
        """Previous calendar year."""
        start_this_year = _start_of_year(now)
        start_last_year = start_this_year.replace(year=start_this_year.year - 1)
        return TimeRange(start=_ts(start_last_year), end=_ts(start_this_year))

    def _handle_last_hour(self, m: re.Match, now: datetime, now_ts: float) -> TimeRange:
        return TimeRange(start=now_ts - 3600.0, end=now_ts)

    def _handle_this_week(self, m: re.Match, now: datetime, now_ts: float) -> TimeRange:
        start = _start_of_week(now)
        return TimeRange(start=_ts(start), end=now_ts)

    def _handle_this_month(self, m: re.Match, now: datetime, now_ts: float) -> TimeRange:
        start = _start_of_month(now)
        return TimeRange(start=_ts(start), end=now_ts)

    def _handle_this_year(self, m: re.Match, now: datetime, now_ts: float) -> TimeRange:
        start = _start_of_year(now)
        return TimeRange(start=_ts(start), end=now_ts)

    def _handle_yesterday(self, m: re.Match, now: datetime, now_ts: float) -> TimeRange:
        yesterday = now - timedelta(days=1)
        start = _ts(_start_of_day(yesterday))
        end = _ts(_start_of_day(now))
        return TimeRange(start=start, end=end)

    def _handle_today(self, m: re.Match, now: datetime, now_ts: float) -> TimeRange:
        start = _ts(_start_of_day(now))
        return TimeRange(start=start, end=now_ts)

    def _handle_recent(self, m: re.Match, now: datetime, now_ts: float) -> TimeRange:
        """'recent' / 'recently' → last 7 days."""
        return TimeRange(start=now_ts - 7 * 86400.0, end=now_ts)

    def _handle_in_month(self, m: re.Match, now: datetime, now_ts: float) -> TimeRange:
        """'in January' → January of nearest year (past or current)."""
        month_num = _MONTH_NAMES[m.group("month").lower()]
        year = now.year
        # If the month hasn't started yet this year, use the previous year
        if month_num > now.month:
            year -= 1
        # Build start and end of that month
        _, last_day = calendar.monthrange(year, month_num)
        tz = timezone.utc
        start = datetime(year, month_num, 1, tzinfo=tz)
        end = datetime(year, month_num, last_day, 23, 59, 59, tzinfo=tz)
        # Never extend beyond now
        end_ts = min(_ts(end), now_ts)
        return TimeRange(start=_ts(start), end=end_ts)

    def _handle_before_last_unit(self, m: re.Match, now: datetime, now_ts: float) -> TimeRange:
        """'before last week' → from epoch to start of last week."""
        unit = m.group("unit2").lower()
        sec = _UNIT_SECONDS.get(unit, 86400)
        end_of_range = now_ts - sec
        # Use Unix epoch as start (effectively "all history before this point")
        return TimeRange(start=0.0, end=end_of_range)

    def _handle_after_last_unit(self, m: re.Match, now: datetime, now_ts: float) -> TimeRange:
        """'after last week' → from end of last-unit until now."""
        unit = m.group("unit3").lower()
        sec = _UNIT_SECONDS.get(unit, 86400)
        start = now_ts - sec
        return TimeRange(start=start, end=now_ts)

    def _handle_after_yesterday(self, m: re.Match, now: datetime, now_ts: float) -> TimeRange:
        """'after yesterday' → from start of today until now."""
        start = _ts(_start_of_day(now))
        return TimeRange(start=start, end=now_ts)

    def _handle_between_weekdays(self, m: re.Match, now: datetime, now_ts: float) -> TimeRange:
        """
        'between Monday and Friday' → range within the current week.
        If the first weekday is after the second (e.g. Fri→Mon), wraps to next week.
        """
        wd1 = _WEEKDAYS[m.group("wd1").lower()]
        wd2 = _WEEKDAYS[m.group("wd2").lower()]

        week_start = _start_of_week(now)  # Monday this week
        day1 = week_start + timedelta(days=wd1)
        day2 = week_start + timedelta(days=wd2)

        if wd2 < wd1:
            # wrap: e.g. "between Friday and Monday" → Fri to next Mon
            day2 += timedelta(weeks=1)

        # Clamp end to now
        end_ts = min(_ts(_end_of_day(day2)), now_ts)
        return TimeRange(start=_ts(day1), end=end_ts)


# ---------------------------------------------------------------------------
# Module-level convenience singleton
# ---------------------------------------------------------------------------

_default_parser: Optional[TemporalQueryParser] = None


def get_temporal_parser() -> TemporalQueryParser:
    """Return a shared TemporalQueryParser instance (lazy init, thread-safe)."""
    global _default_parser
    if _default_parser is None:
        _default_parser = TemporalQueryParser()
    return _default_parser
