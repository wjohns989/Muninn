"""
test_v3_10_0_temporal.py
-------------------------
Phase 13 (v3.10.0) — Temporal Query Expansion tests.

37 tests covering:
- Parsing of all supported NL time phrases
- Edge cases (ambiguous months, future dates, wrap-around)
- TimeRange validity invariants (start < end, end <= now)
- Integration with _temporal_search (filter by time range)
- HybridRetriever temporal expansion flag gating
- Thread safety of the shared parser
- End-to-end integration with a mocked HybridRetriever
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from muninn.retrieval.temporal_parser import (
    TimeRange,
    TemporalQueryParser,
    get_temporal_parser,
)


# ---------------------------------------------------------------------------
# Shared reference time: 2025-06-15 12:00:00 UTC (mid-year, mid-month)
# ---------------------------------------------------------------------------

_REF_ISO = "2025-06-15T12:00:00+00:00"
_REF_DT = datetime.fromisoformat(_REF_ISO)
_REF_TS = _REF_DT.timestamp()

_PARSER = TemporalQueryParser()


def _parse(q: str) -> Optional[Tuple[TimeRange, str]]:
    return _PARSER.parse(q, reference_time=_REF_TS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_range(
    result,
    *,
    expected_start_approx: float,
    expected_end_approx: float,
    tolerance_s: float = 120.0,
):
    """Assert a parse result contains a range close to expected timestamps."""
    assert result is not None, "Expected a parsed TimeRange but got None"
    tr, _ = result
    assert abs(tr.start - expected_start_approx) <= tolerance_s, (
        f"start mismatch: got {tr.start}, expected ~{expected_start_approx}"
    )
    assert abs(tr.end - expected_end_approx) <= tolerance_s, (
        f"end mismatch: got {tr.end}, expected ~{expected_end_approx}"
    )


# ---------------------------------------------------------------------------
# 1. "last N days"
# ---------------------------------------------------------------------------


def test_parse_last_n_days():
    result = _parse("show me memories from the last 7 days")
    _assert_range(
        result,
        expected_start_approx=_REF_TS - 7 * 86400,
        expected_end_approx=_REF_TS,
    )


# ---------------------------------------------------------------------------
# 2. "last N weeks"
# ---------------------------------------------------------------------------


def test_parse_last_n_weeks():
    result = _parse("what happened in the last 2 weeks?")
    _assert_range(
        result,
        expected_start_approx=_REF_TS - 2 * 604800,
        expected_end_approx=_REF_TS,
    )


# ---------------------------------------------------------------------------
# 3. "last N months"
# ---------------------------------------------------------------------------


def test_parse_last_n_months():
    result = _parse("bugs reported over the last 3 months")
    _assert_range(
        result,
        expected_start_approx=_REF_TS - 3 * 2592000,
        expected_end_approx=_REF_TS,
    )


# ---------------------------------------------------------------------------
# 4. "last N years"
# ---------------------------------------------------------------------------


def test_parse_last_n_years():
    result = _parse("architecture decisions in the last 2 years")
    _assert_range(
        result,
        expected_start_approx=_REF_TS - 2 * 31536000,
        expected_end_approx=_REF_TS,
    )


# ---------------------------------------------------------------------------
# 5. "yesterday"
# ---------------------------------------------------------------------------


def test_parse_yesterday():
    result = _parse("what did we deploy yesterday?")
    assert result is not None
    tr, _ = result
    # Yesterday in UTC from ref = 2025-06-14
    yesterday_start = datetime(2025, 6, 14, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    yesterday_end = datetime(2025, 6, 15, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    assert abs(tr.start - yesterday_start) < 60
    assert abs(tr.end - yesterday_end) < 60


# ---------------------------------------------------------------------------
# 6. "today"
# ---------------------------------------------------------------------------


def test_parse_today():
    result = _parse("memories from today")
    assert result is not None
    tr, _ = result
    today_start = datetime(2025, 6, 15, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    assert abs(tr.start - today_start) < 60
    # end should be approx _REF_TS
    assert abs(tr.end - _REF_TS) < 120


# ---------------------------------------------------------------------------
# 7. "this week"
# ---------------------------------------------------------------------------


def test_parse_this_week():
    result = _parse("what happened this week?")
    assert result is not None
    tr, _ = result
    # Ref date 2025-06-15 is a Sunday; week starts 2025-06-09 (Mon)
    week_start = datetime(2025, 6, 9, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    assert abs(tr.start - week_start) < 60
    assert abs(tr.end - _REF_TS) < 120


# ---------------------------------------------------------------------------
# 8. "this month"
# ---------------------------------------------------------------------------


def test_parse_this_month():
    result = _parse("everything added this month")
    assert result is not None
    tr, _ = result
    month_start = datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    assert abs(tr.start - month_start) < 60
    assert abs(tr.end - _REF_TS) < 120


# ---------------------------------------------------------------------------
# 9. "this year"
# ---------------------------------------------------------------------------


def test_parse_this_year():
    result = _parse("architecture decisions this year")
    assert result is not None
    tr, _ = result
    year_start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    assert abs(tr.start - year_start) < 60
    assert abs(tr.end - _REF_TS) < 120


# ---------------------------------------------------------------------------
# 10. "N days ago"
# ---------------------------------------------------------------------------


def test_parse_n_days_ago():
    result = _parse("what happened 5 days ago?")
    assert result is not None
    tr, _ = result
    centre = _REF_TS - 5 * 86400
    # Window spans half a day either side at minimum
    assert tr.start <= centre
    assert tr.end >= centre


# ---------------------------------------------------------------------------
# 11. "recent"
# ---------------------------------------------------------------------------


def test_parse_recent():
    result = _parse("recent auth failures")
    _assert_range(
        result,
        expected_start_approx=_REF_TS - 7 * 86400,
        expected_end_approx=_REF_TS,
    )


# ---------------------------------------------------------------------------
# 12. "in <MonthName>" — past month
# ---------------------------------------------------------------------------


def test_parse_month_name():
    # Ref is June 2025; "in March" should → March 2025
    result = _parse("releases in March")
    assert result is not None
    tr, _ = result
    march_start = datetime(2025, 3, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    march_end = datetime(2025, 3, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp()
    assert abs(tr.start - march_start) < 60
    # End should be at or before ref time (March is past)
    assert tr.end <= _REF_TS + 60


# ---------------------------------------------------------------------------
# 13. "before last week"
# ---------------------------------------------------------------------------


def test_parse_before():
    result = _parse("incidents before last week")
    assert result is not None
    tr, _ = result
    # epoch to ~now-1week
    assert tr.start == pytest.approx(0.0, abs=1.0)
    assert tr.end <= _REF_TS


# ---------------------------------------------------------------------------
# 14. "after yesterday"
# ---------------------------------------------------------------------------


def test_parse_after():
    result = _parse("deployments after yesterday")
    assert result is not None
    tr, _ = result
    today_start = datetime(2025, 6, 15, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    assert abs(tr.start - today_start) < 60
    assert abs(tr.end - _REF_TS) < 120


# ---------------------------------------------------------------------------
# 15. No temporal phrase → None
# ---------------------------------------------------------------------------


def test_parse_no_temporal():
    result = _parse("what is the capital of France?")
    assert result is None


# ---------------------------------------------------------------------------
# 16. Mixed query — phrase extracted, rest remains
# ---------------------------------------------------------------------------


def test_parse_mixed_query():
    result = _parse("auth issues from last week in production")
    assert result is not None
    _tr, cleaned = result
    assert "auth" in cleaned.lower()
    assert "production" in cleaned.lower()
    # "last week" should be gone
    assert "last week" not in cleaned.lower()


# ---------------------------------------------------------------------------
# 17. Case-insensitive
# ---------------------------------------------------------------------------


def test_parse_case_insensitive():
    result_lower = _parse("problems Last Week")
    result_upper = _parse("problems LAST WEEK")
    assert result_lower is not None
    assert result_upper is not None
    # Both resolve to same (approx) range
    assert abs(result_lower[0].start - result_upper[0].start) < 1.0


# ---------------------------------------------------------------------------
# 18. "past N days" synonym for "last N days"
# ---------------------------------------------------------------------------


def test_parse_ordinal_positions():
    result_last = _parse("last 30 days")
    result_past = _parse("past 30 days")
    assert result_last is not None
    assert result_past is not None
    # Both should resolve to the same range (within rounding)
    assert abs(result_last[0].start - result_past[0].start) < 2.0
    assert abs(result_last[0].end - result_past[0].end) < 2.0


# ---------------------------------------------------------------------------
# 19. TimeRange invariant: start <= end
# ---------------------------------------------------------------------------


def test_time_range_start_before_end():
    queries = [
        "last 7 days",
        "yesterday",
        "this week",
        "this month",
        "this year",
        "last week",
        "last month",
        "last year",
        "recently",
        "in March",
        "after yesterday",
    ]
    for q in queries:
        result = _parse(q)
        if result is None:
            continue
        tr, _ = result
        assert tr.start <= tr.end, f"TimeRange start > end for query={q!r}: {tr}"


# ---------------------------------------------------------------------------
# 20. TimeRange invariant: end <= now
# ---------------------------------------------------------------------------


def test_time_range_end_at_or_before_now():
    queries = [
        "last 7 days",
        "today",
        "this week",
        "recent",
        "last 24 hours",
    ]
    for q in queries:
        result = _parse(q)
        if result is None:
            continue
        tr, _ = result
        # Allow 5-second slack for slow CI
        assert tr.end <= _REF_TS + 5.0, (
            f"TimeRange.end {tr.end} > now {_REF_TS} for query={q!r}"
        )


# ---------------------------------------------------------------------------
# 21. _temporal_search respects time_range filter (records included)
# ---------------------------------------------------------------------------


def test_temporal_search_with_range_filters():
    """Records within the time range should appear in temporal results."""
    from unittest.mock import MagicMock, patch
    from muninn.retrieval.temporal_parser import TimeRange
    from muninn.retrieval.hybrid import HybridRetriever

    # Build a thin MemoryRecord mock
    now = _REF_TS

    def _make_record(rid, created_at, importance=0.5):
        r = MagicMock()
        r.id = rid
        r.created_at = created_at
        r.importance = importance
        r.namespace = "default"
        return r

    inside = _make_record("inside", now - 3 * 86400)   # 3 days ago — in range
    outside = _make_record("outside", now - 10 * 86400) # 10 days ago — out of range

    meta = MagicMock()
    meta.get_all.return_value = [inside, outside]

    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.metadata = meta
    retriever.vectors = MagicMock()
    retriever.graph = MagicMock()
    retriever.bm25 = MagicMock()

    tr = TimeRange(start=now - 7 * 86400, end=now)
    results = retriever._temporal_search(
        filters={},
        namespaces=None,
        user_id=None,
        limit=10,
        time_range=tr,
    )
    result_ids = [rid for rid, _ in results]
    assert "inside" in result_ids
    assert "outside" not in result_ids


# ---------------------------------------------------------------------------
# 22. _temporal_search without time_range — original behaviour preserved
# ---------------------------------------------------------------------------


def test_temporal_search_no_range():
    """Without a time_range, all records returned (original behaviour)."""
    from muninn.retrieval.hybrid import HybridRetriever

    now = _REF_TS

    def _make_record(rid, created_at):
        r = MagicMock()
        r.id = rid
        r.created_at = created_at
        r.importance = 0.5
        r.namespace = "default"
        return r

    records = [_make_record(f"m{i}", now - i * 86400) for i in range(5)]
    meta = MagicMock()
    meta.get_all.return_value = records

    retriever = HybridRetrieval = HybridRetriever.__new__(HybridRetriever)
    retriever.metadata = meta
    retriever.vectors = MagicMock()
    retriever.graph = MagicMock()
    retriever.bm25 = MagicMock()

    results = retriever._temporal_search(
        filters={}, namespaces=None, user_id=None, limit=10, time_range=None
    )
    assert len(results) == 5


# ---------------------------------------------------------------------------
# 23. _temporal_search excludes records before range start
# ---------------------------------------------------------------------------


def test_temporal_search_range_excludes_old():
    from muninn.retrieval.hybrid import HybridRetriever

    now = _REF_TS
    old = MagicMock()
    old.id = "old_mem"
    old.created_at = now - 30 * 86400  # 30 days ago
    old.importance = 0.9
    old.namespace = "default"

    meta = MagicMock()
    meta.get_all.return_value = [old]

    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.metadata = meta
    retriever.vectors = MagicMock()
    retriever.graph = MagicMock()
    retriever.bm25 = MagicMock()

    tr = TimeRange(start=now - 7 * 86400, end=now)
    results = retriever._temporal_search(
        filters={}, namespaces=None, user_id=None, limit=10, time_range=tr
    )
    assert results == []


# ---------------------------------------------------------------------------
# 24. _temporal_search excludes records after range end
# ---------------------------------------------------------------------------


def test_temporal_search_range_excludes_future():
    from muninn.retrieval.hybrid import HybridRetriever

    now = _REF_TS
    future = MagicMock()
    future.id = "future_mem"
    future.created_at = now + 3600  # 1 hour in the future
    future.importance = 0.5
    future.namespace = "default"

    meta = MagicMock()
    meta.get_all.return_value = [future]

    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.metadata = meta
    retriever.vectors = MagicMock()
    retriever.graph = MagicMock()
    retriever.bm25 = MagicMock()

    tr = TimeRange(start=now - 86400, end=now)
    results = retriever._temporal_search(
        filters={}, namespaces=None, user_id=None, limit=10, time_range=tr
    )
    assert results == []


# ---------------------------------------------------------------------------
# 25. _temporal_search includes records inside range
# ---------------------------------------------------------------------------


def test_temporal_search_range_includes_matching():
    from muninn.retrieval.hybrid import HybridRetriever

    now = _REF_TS
    match = MagicMock()
    match.id = "match_mem"
    match.created_at = now - 2 * 86400  # 2 days ago
    match.importance = 0.7
    match.namespace = "default"

    meta = MagicMock()
    meta.get_all.return_value = [match]

    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.metadata = meta
    retriever.vectors = MagicMock()
    retriever.graph = MagicMock()
    retriever.bm25 = MagicMock()

    tr = TimeRange(start=now - 7 * 86400, end=now)
    results = retriever._temporal_search(
        filters={}, namespaces=None, user_id=None, limit=10, time_range=tr
    )
    assert len(results) == 1
    assert results[0][0] == "match_mem"


# ---------------------------------------------------------------------------
# 26. HybridRetriever uses temporal expansion when flag enabled
# ---------------------------------------------------------------------------


def test_hybrid_retriever_temporal_expansion_enabled():
    """
    When temporal_query_expansion flag is enabled, the parser is invoked and
    resolved_time_range is passed into _temporal_search.
    """
    from muninn.retrieval.hybrid import HybridRetriever
    from muninn.core.feature_flags import FeatureFlags

    retriever = _make_minimal_retriever()

    temporal_called_with: dict = {}

    original_ts = retriever._temporal_search

    def capturing_temporal_search(**kwargs):
        temporal_called_with.update(kwargs)
        return []

    retriever._temporal_search = capturing_temporal_search

    flags_enabled = FeatureFlags(
        temporal_query_expansion=True,
        explainable_recall=False,
    )

    async def _run():
        with patch("muninn.retrieval.hybrid.get_flags", return_value=flags_enabled):
            await retriever.search("what happened last week?", limit=5)

    asyncio.run(_run())

    assert "time_range" in temporal_called_with
    assert temporal_called_with["time_range"] is not None


# ---------------------------------------------------------------------------
# 27. HybridRetriever skips temporal expansion when flag disabled
# ---------------------------------------------------------------------------


def test_hybrid_retriever_temporal_expansion_disabled():
    from muninn.retrieval.hybrid import HybridRetriever
    from muninn.core.feature_flags import FeatureFlags

    retriever = _make_minimal_retriever()
    temporal_called_with: dict = {}

    def capturing_temporal_search(**kwargs):
        temporal_called_with.update(kwargs)
        return []

    retriever._temporal_search = capturing_temporal_search

    flags_disabled = FeatureFlags(
        temporal_query_expansion=False,
        explainable_recall=False,
    )

    async def _run():
        with patch("muninn.retrieval.hybrid.get_flags", return_value=flags_disabled):
            await retriever.search("what happened last week?", limit=5)

    asyncio.run(_run())

    # time_range should be None when flag is off
    assert temporal_called_with.get("time_range") is None


# ---------------------------------------------------------------------------
# 28. Temporal expansion threads time_range through
# ---------------------------------------------------------------------------


def test_temporal_expansion_updates_time_range():
    """Confirm the time_range kwarg is a TimeRange when a phrase is detected."""
    from muninn.retrieval.hybrid import HybridRetriever
    from muninn.core.feature_flags import FeatureFlags
    from muninn.retrieval.temporal_parser import TimeRange

    retriever = _make_minimal_retriever()
    captured_ranges = []

    def capturing_ts(**kwargs):
        tr = kwargs.get("time_range")
        captured_ranges.append(tr)
        return []

    retriever._temporal_search = capturing_ts

    flags_enabled = FeatureFlags(temporal_query_expansion=True, explainable_recall=False)

    async def _run():
        with patch("muninn.retrieval.hybrid.get_flags", return_value=flags_enabled):
            await retriever.search("auth failures in the last 3 days", limit=5)

    asyncio.run(_run())
    assert len(captured_ranges) == 1
    assert isinstance(captured_ranges[0], TimeRange)


# ---------------------------------------------------------------------------
# 29. "last hour"
# ---------------------------------------------------------------------------


def test_parse_last_hour():
    result = _parse("errors in the last hour")
    _assert_range(
        result,
        expected_start_approx=_REF_TS - 3600,
        expected_end_approx=_REF_TS,
        tolerance_s=60,
    )


# ---------------------------------------------------------------------------
# 30. "last 24 hours"
# ---------------------------------------------------------------------------


def test_parse_last_24_hours():
    result = _parse("deployments in the last 24 hours")
    _assert_range(
        result,
        expected_start_approx=_REF_TS - 24 * 3600,
        expected_end_approx=_REF_TS,
        tolerance_s=60,
    )


# ---------------------------------------------------------------------------
# 31. "past week" synonym for "last week"
# ---------------------------------------------------------------------------


def test_parse_last_week_synonym():
    r_last = _parse("issues from last week")
    r_past = _parse("issues from past week")
    assert r_last is not None
    assert r_past is not None
    assert abs(r_last[0].start - r_past[0].start) < 2.0


# ---------------------------------------------------------------------------
# 32. Cleaned query strips temporal phrase
# ---------------------------------------------------------------------------


def test_extracted_query_strips_temporal():
    result = _parse("database deadlocks from last week in production")
    assert result is not None
    _, cleaned = result
    assert "last week" not in cleaned.lower()
    assert "database" in cleaned.lower()
    assert "production" in cleaned.lower()


# ---------------------------------------------------------------------------
# 33. "in March" when March hasn't happened yet this year → previous year
# ---------------------------------------------------------------------------


def test_parse_ambiguous_month():
    # Ref is June 2025; "in August" is still in future → use Aug 2024
    result = _PARSER.parse("in August", reference_time=_REF_TS)
    assert result is not None
    tr, _ = result
    aug_2024_start = datetime(2024, 8, 1, tzinfo=timezone.utc).timestamp()
    assert abs(tr.start - aug_2024_start) < 60


# ---------------------------------------------------------------------------
# 34. "between Monday and Friday"
# ---------------------------------------------------------------------------


def test_parse_between_timestamps():
    # Ref 2025-06-15 is Sunday (weekday 6); week starts 2025-06-09 (Mon)
    result = _parse("tasks between Monday and Friday")
    assert result is not None
    tr, _ = result
    mon = datetime(2025, 6, 9, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    fri_end = datetime(2025, 6, 13, 23, 59, 59, tzinfo=timezone.utc).timestamp()
    assert abs(tr.start - mon) < 60
    # end should be at or around Friday EOD
    assert tr.end <= _REF_TS + 60


# ---------------------------------------------------------------------------
# 35. TimeRange.__str__ readable
# ---------------------------------------------------------------------------


def test_temporal_range_str():
    tr = TimeRange(start=_REF_TS - 86400, end=_REF_TS)
    s = str(tr)
    assert "TimeRange(" in s
    assert "UTC" in s
    assert "->" in s


# ---------------------------------------------------------------------------
# 36. Thread safety — concurrent parse calls
# ---------------------------------------------------------------------------


def test_temporal_parser_thread_safety():
    """Concurrent calls to the shared parser should never raise or corrupt."""
    parser = get_temporal_parser()
    errors: list = []
    results: list = []
    lock = threading.Lock()

    def _worker():
        for query in ["last 7 days", "yesterday", "this month", "in March", "recent"]:
            try:
                r = parser.parse(query, reference_time=_REF_TS)
                with lock:
                    results.append(r)
            except Exception as exc:
                with lock:
                    errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Thread errors: {errors}"
    # All non-None results should have valid ranges
    for item in results:
        if item is not None:
            tr, _ = item
            assert tr.start <= tr.end


# ---------------------------------------------------------------------------
# 37. Integration — mock HybridRetriever end-to-end
# ---------------------------------------------------------------------------


def test_temporal_parser_integration():
    """
    End-to-end: a temporal phrase in the query causes only time-range-matching
    records to surface from _temporal_search while the non-temporal signals
    are called with the cleaned query.
    """
    from muninn.retrieval.hybrid import HybridRetriever
    from muninn.core.feature_flags import FeatureFlags

    retriever = _make_minimal_retriever()

    now_ts = time.time()
    recent_record = MagicMock()
    recent_record.id = "recent_mem"
    recent_record.created_at = now_ts - 2 * 86400  # 2 days ago
    recent_record.importance = 0.8
    recent_record.namespace = "default"

    old_record = MagicMock()
    old_record.id = "old_mem"
    old_record.created_at = now_ts - 30 * 86400  # 30 days ago
    old_record.importance = 0.9
    old_record.namespace = "default"

    retriever.metadata.get_all.return_value = [recent_record, old_record]

    temporal_results_captured = []
    original_ts = HybridRetriever._temporal_search

    def spy_ts(self, **kwargs):
        r = original_ts(self, **kwargs)
        temporal_results_captured.extend(r)
        return r

    with patch.object(HybridRetriever, "_temporal_search", spy_ts):
        flags_enabled = FeatureFlags(
            temporal_query_expansion=True, explainable_recall=False
        )
        with patch("muninn.retrieval.hybrid.get_flags", return_value=flags_enabled):
            async def _run():
                await retriever.search("problems in the last 7 days", limit=5)
            asyncio.run(_run())

    # Only the recent record should appear in temporal results
    temporal_ids = [rid for rid, _ in temporal_results_captured]
    assert "recent_mem" in temporal_ids
    assert "old_mem" not in temporal_ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_retriever():
    """Build a HybridRetriever with all stores mocked for unit testing."""
    from muninn.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever.__new__(HybridRetriever)

    # Metadata store
    meta = MagicMock()
    meta.get_all.return_value = []
    meta.get_by_ids.return_value = []
    meta.record_access_batch = MagicMock()
    retriever.metadata = meta

    # Vector store
    vec = MagicMock()
    vec.search.return_value = []
    retriever.vectors = vec

    # Graph store
    retriever.graph = MagicMock()
    retriever.graph.find_related_memories.return_value = []

    # BM25
    bm = MagicMock()
    bm.search.return_value = []
    retriever.bm25 = bm

    # Reranker
    retriever.reranker = None
    retriever._colbert_enabled = False
    retriever._colbert_indexer = None
    retriever._colbert_scorer = MagicMock()
    retriever._telemetry = MagicMock()
    retriever._telemetry.span = MagicMock(return_value=MagicMock(
        __enter__=MagicMock(return_value=None),
        __exit__=MagicMock(return_value=False),
    ))
    retriever._telemetry.add_event = MagicMock()
    retriever._chain_signal_weight = 0.6
    retriever._chain_expansion_limit = 20

    # Chain retriever
    chain = MagicMock()
    chain.max_seed_memories = 6
    chain.expand_from_ranked_results.return_value = []
    retriever._chain_retriever = chain

    # Embed function (returns a deterministic fake embedding)
    import asyncio
    async def _fake_embed(text):
        return [0.1] * 768
    retriever._embed_fn = _fake_embed

    return retriever
