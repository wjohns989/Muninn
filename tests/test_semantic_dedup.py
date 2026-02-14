"""Tests for muninn.dedup.semantic_dedup — Semantic Deduplication (v3.2.0)."""

import pytest
from unittest.mock import MagicMock, patch
from muninn.dedup.semantic_dedup import SemanticDedup, DedupStrategy, DedupResult
from muninn.core.types import MemoryRecord, MemoryType, Provenance


def _make_record(content="test memory", memory_id=None, **kwargs):
    """Helper to create MemoryRecord instances."""
    defaults = {
        "content": content,
        "memory_type": MemoryType.EPISODIC,
        "provenance": Provenance.AUTO_EXTRACTED,
    }
    defaults.update(kwargs)
    record = MemoryRecord(**defaults)
    if memory_id:
        record.id = memory_id
    return record


class TestSemanticDedupInit:
    """Initialization and validation."""

    def test_default_thresholds(self):
        dedup = SemanticDedup()
        assert dedup.threshold == 0.95
        assert dedup.content_overlap_threshold == 0.8

    def test_custom_thresholds(self):
        dedup = SemanticDedup(threshold=0.90, content_overlap_threshold=0.7)
        assert dedup.threshold == 0.90
        assert dedup.content_overlap_threshold == 0.7

    def test_threshold_zero_raises(self):
        with pytest.raises(ValueError, match="threshold must be in"):
            SemanticDedup(threshold=0.0)

    def test_threshold_negative_raises(self):
        with pytest.raises(ValueError, match="threshold must be in"):
            SemanticDedup(threshold=-0.5)

    def test_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="threshold must be in"):
            SemanticDedup(threshold=1.5)

    def test_overlap_threshold_negative_raises(self):
        with pytest.raises(ValueError, match="content_overlap_threshold must be in"):
            SemanticDedup(content_overlap_threshold=-0.1)

    def test_overlap_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="content_overlap_threshold must be in"):
            SemanticDedup(content_overlap_threshold=1.5)

    def test_threshold_exactly_one_allowed(self):
        dedup = SemanticDedup(threshold=1.0)
        assert dedup.threshold == 1.0


class TestContentOverlap:
    """Token-level Jaccard similarity."""

    def test_identical_strings(self):
        result = SemanticDedup._content_overlap("hello world", "hello world")
        assert result == 1.0

    def test_completely_different(self):
        result = SemanticDedup._content_overlap("hello world", "foo bar baz")
        assert result == 0.0

    def test_partial_overlap(self):
        result = SemanticDedup._content_overlap("hello world foo", "hello world bar")
        # intersection = {hello, world} = 2, union = {hello, world, foo, bar} = 4
        assert abs(result - 0.5) < 0.001

    def test_empty_both(self):
        result = SemanticDedup._content_overlap("", "")
        assert result == 1.0

    def test_empty_one(self):
        result = SemanticDedup._content_overlap("hello world", "")
        assert result == 0.0

    def test_empty_other(self):
        result = SemanticDedup._content_overlap("", "hello world")
        assert result == 0.0

    def test_case_insensitive(self):
        result = SemanticDedup._content_overlap("Hello World", "hello world")
        assert result == 1.0

    def test_subset_less_than_one(self):
        # "hello" is a subset of "hello world" but not identical
        result = SemanticDedup._content_overlap("hello", "hello world")
        # intersection = {hello} = 1, union = {hello, world} = 2
        assert abs(result - 0.5) < 0.001

    def test_result_always_in_range(self):
        texts = [
            ("a b c d e f", "a b c x y z"),
            ("one", "one two three four"),
            ("test memory content", "test memory content with extra"),
        ]
        for a, b in texts:
            result = SemanticDedup._content_overlap(a, b)
            assert 0.0 <= result <= 1.0


class TestDetermineStrategy:
    """Strategy selection logic."""

    def _dedup(self):
        return SemanticDedup()

    def test_very_high_similarity_and_overlap_returns_skip(self):
        dedup = self._dedup()
        existing = _make_record("hello world test content")
        strategy = dedup._determine_strategy(
            new_content="hello world test content",
            existing_record=existing,
            similarity=0.99,
            overlap=0.96,
        )
        assert strategy == DedupStrategy.SKIP

    def test_high_similarity_low_overlap_checks_novel(self):
        dedup = self._dedup()
        # "alpha beta gamma" has significant novel tokens vs "hello world test"
        existing = _make_record("hello world test")
        strategy = dedup._determine_strategy(
            new_content="hello world test alpha beta gamma delta",
            existing_record=existing,
            similarity=0.96,
            overlap=0.50,
        )
        # >20% novel tokens → UPDATE_EXISTING
        assert strategy == DedupStrategy.UPDATE_EXISTING

    def test_few_novel_tokens_returns_skip(self):
        dedup = self._dedup()
        existing = _make_record("the quick brown fox jumps over the lazy dog")
        strategy = dedup._determine_strategy(
            new_content="the quick brown fox jumps over the lazy dog today",
            existing_record=existing,
            similarity=0.96,
            overlap=0.85,
        )
        # Only 1 novel token ("today") out of 10 = 10% < 20% → SKIP
        assert strategy == DedupStrategy.SKIP

    def test_boundary_exactly_0_98_sim_0_95_overlap(self):
        dedup = self._dedup()
        existing = _make_record("same content")
        strategy = dedup._determine_strategy(
            new_content="same content",
            existing_record=existing,
            similarity=0.98,
            overlap=0.95,
        )
        assert strategy == DedupStrategy.SKIP


class TestCheckDuplicate:
    """Integration tests for the check_duplicate workflow."""

    def _dedup(self, threshold=0.95, overlap=0.8):
        return SemanticDedup(threshold=threshold, content_overlap_threshold=overlap)

    def _mock_stores(self, search_results=None, get_record=None):
        """Create mock vector and metadata stores."""
        vector_store = MagicMock()
        vector_store.search.return_value = search_results or []

        metadata_store = MagicMock()
        metadata_store.get.return_value = get_record
        return vector_store, metadata_store

    def test_no_matches_returns_none(self):
        dedup = self._dedup()
        vs, ms = self._mock_stores(search_results=[])
        result = dedup.check_duplicate([0.1, 0.2, 0.3], "test content", vs, ms)
        assert result is None

    def test_match_below_threshold_returns_none(self):
        dedup = self._dedup(threshold=0.95)
        vs, ms = self._mock_stores(
            search_results=[("mem-1", 0.93)],  # Below threshold
            get_record=_make_record("existing content"),
        )
        result = dedup.check_duplicate([0.1, 0.2, 0.3], "test content", vs, ms)
        assert result is None

    def test_match_above_threshold_with_high_overlap_returns_skip(self):
        dedup = self._dedup(threshold=0.95, overlap=0.5)
        existing = _make_record("the user prefers dark mode", memory_id="mem-1")
        vs, ms = self._mock_stores(
            search_results=[("mem-1", 0.99)],
            get_record=existing,
        )
        result = dedup.check_duplicate(
            [0.1, 0.2], "the user prefers dark mode", vs, ms,
        )
        assert result is not None
        assert result.is_duplicate is True
        assert result.existing_memory_id == "mem-1"
        assert result.similarity == 0.99
        assert result.strategy == DedupStrategy.SKIP

    def test_match_with_novel_content_returns_update(self):
        dedup = self._dedup(threshold=0.95, overlap=0.3)
        existing = _make_record("user prefers dark mode", memory_id="mem-1")
        vs, ms = self._mock_stores(
            search_results=[("mem-1", 0.96)],
            get_record=existing,
        )
        # New content has significant novel tokens (>20%)
        result = dedup.check_duplicate(
            [0.1, 0.2],
            "user prefers dark mode and uses vim keybindings exclusively for editing",
            vs, ms,
        )
        assert result is not None
        assert result.is_duplicate is True
        assert result.strategy == DedupStrategy.UPDATE_EXISTING

    def test_exclude_ids_respected(self):
        dedup = self._dedup(threshold=0.95, overlap=0.5)
        vs, ms = self._mock_stores(
            search_results=[("mem-1", 0.99)],
            get_record=_make_record("same content", memory_id="mem-1"),
        )
        result = dedup.check_duplicate(
            [0.1, 0.2], "same content", vs, ms, exclude_ids=["mem-1"],
        )
        assert result is None

    def test_filters_are_forwarded_to_vector_search(self):
        dedup = self._dedup(threshold=0.95, overlap=0.5)
        existing = _make_record("the user prefers dark mode", memory_id="mem-1")
        vs, ms = self._mock_stores(
            search_results=[("mem-1", 0.99)],
            get_record=existing,
        )
        dedup.check_duplicate(
            [0.1, 0.2],
            "the user prefers dark mode",
            vs,
            ms,
            filters={"namespace": "project-a", "user_id": "user-1"},
        )
        assert vs.search.call_args.kwargs["filters"] == {
            "namespace": "project-a",
            "user_id": "user-1",
        }

    def test_filters_enforced_against_metadata_scope_mismatch(self):
        dedup = self._dedup(threshold=0.95, overlap=0.5)
        existing = _make_record("the user prefers dark mode", memory_id="mem-1")
        existing.namespace = "project-b"
        existing.metadata = {"user_id": "user-2"}
        vs, ms = self._mock_stores(
            search_results=[("mem-1", 0.99)],
            get_record=existing,
        )
        result = dedup.check_duplicate(
            [0.1, 0.2],
            "the user prefers dark mode",
            vs,
            ms,
            filters={"namespace": "project-a", "user_id": "user-1"},
        )
        assert result is None

    def test_duplicate_debug_log_does_not_include_memory_content(self, caplog):
        dedup = self._dedup(threshold=0.95, overlap=0.5)
        sensitive = "api key is sk-secret-123"
        existing = _make_record(sensitive, memory_id="mem-1", namespace="project-a", metadata={"user_id": "user-1"})
        vs, ms = self._mock_stores(
            search_results=[("mem-1", 0.99)],
            get_record=existing,
        )

        with caplog.at_level("DEBUG", logger="Muninn.Dedup"):
            dedup.check_duplicate(
                [0.1, 0.2],
                sensitive,
                vs,
                ms,
                filters={"namespace": "project-a", "user_id": "user-1"},
            )

        assert "Semantic duplicate match found" in caplog.text
        assert sensitive not in caplog.text

    def test_vector_store_error_returns_none(self):
        dedup = self._dedup()
        vs = MagicMock()
        vs.search.side_effect = RuntimeError("vector store unavailable")
        ms = MagicMock()
        result = dedup.check_duplicate([0.1, 0.2], "test", vs, ms)
        assert result is None

    def test_metadata_store_returns_none_skips_match(self):
        dedup = self._dedup(threshold=0.95, overlap=0.5)
        vs, ms = self._mock_stores(
            search_results=[("mem-1", 0.99)],
            get_record=None,  # metadata.get returns None
        )
        result = dedup.check_duplicate([0.1, 0.2], "content", vs, ms)
        assert result is None

    def test_content_below_overlap_threshold_returns_none(self):
        dedup = self._dedup(threshold=0.95, overlap=0.9)
        existing = _make_record("completely different existing text here", memory_id="mem-1")
        vs, ms = self._mock_stores(
            search_results=[("mem-1", 0.96)],
            get_record=existing,
        )
        result = dedup.check_duplicate(
            [0.1, 0.2], "brand new unique content about something else", vs, ms,
        )
        # High embedding similarity but low content overlap → not a duplicate
        assert result is None


class TestMergeContent:
    """Content merging for UPDATE_EXISTING strategy."""

    def _dedup(self):
        return SemanticDedup()

    def test_novel_sentences_appended(self):
        dedup = self._dedup()
        result = dedup.merge_content(
            new_content="User likes Python. User also uses TypeScript.",
            existing_content="User likes Python.",
        )
        assert "User likes Python" in result
        assert "TypeScript" in result

    def test_no_novel_content_returns_existing(self):
        dedup = self._dedup()
        result = dedup.merge_content(
            new_content="User likes Python.",
            existing_content="User likes Python.",
        )
        assert result == "User likes Python."

    def test_merge_preserves_existing(self):
        dedup = self._dedup()
        existing = "The system uses PostgreSQL for metadata storage."
        result = dedup.merge_content(
            new_content="The system also uses Redis for caching.",
            existing_content=existing,
        )
        assert "PostgreSQL" in result
        assert "Redis" in result

    def test_merge_adds_trailing_period(self):
        dedup = self._dedup()
        result = dedup.merge_content(
            new_content="New info here.",
            existing_content="Old content without period",
        )
        # Should add period to existing before appending
        assert result.endswith(".")


class TestDedupResult:
    """DedupResult model validation."""

    def test_default_values(self):
        result = DedupResult()
        assert result.is_duplicate is False
        assert result.existing_memory_id is None
        assert result.similarity == 0.0
        assert result.strategy == DedupStrategy.SKIP

    def test_model_dump(self):
        result = DedupResult(
            is_duplicate=True,
            existing_memory_id="mem-123",
            similarity=0.97,
            strategy=DedupStrategy.UPDATE_EXISTING,
            explanation="test",
        )
        data = result.model_dump()
        assert data["is_duplicate"] is True
        assert data["existing_memory_id"] == "mem-123"
        assert data["strategy"] == "update_existing"
