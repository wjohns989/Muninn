"""Tests for muninn.conflict.detector + resolver — NLI Conflict Detection (v3.2.0).

Note: Tests that require the actual DeBERTa model are marked with @pytest.mark.slow
and require transformers+torch. Tests that only exercise the logic without inference
run without those deps using mocks.
"""

import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from muninn.conflict.detector import (
    ConflictDetector,
    ConflictResult,
    ConflictResolution,
)
from muninn.conflict.resolver import ConflictResolver
from muninn.core.types import MemoryRecord, MemoryType, Provenance


def _make_record(content="test memory", memory_id=None, importance=0.5,
                 created_at=None, **kwargs):
    """Helper to create MemoryRecord instances."""
    defaults = {
        "content": content,
        "memory_type": MemoryType.EPISODIC,
        "provenance": Provenance.AUTO_EXTRACTED,
        "importance": importance,
    }
    if created_at is not None:
        defaults["created_at"] = created_at
    defaults.update(kwargs)
    record = MemoryRecord(**defaults)
    if memory_id:
        record.id = memory_id
    return record


# ──────────────────────────────────────────────
# ConflictResult Model Tests
# ──────────────────────────────────────────────

class TestConflictResult:
    """ConflictResult Pydantic model validation."""

    def test_valid_result(self):
        result = ConflictResult(
            new_content="A contradicts B",
            existing_memory_id="mem-1",
            existing_content="B is the truth",
            contradiction_score=0.85,
            entailment_score=0.05,
            neutral_score=0.10,
            suggested_resolution=ConflictResolution.FLAG_FOR_REVIEW,
        )
        assert result.contradiction_score == 0.85
        assert result.suggested_resolution == ConflictResolution.FLAG_FOR_REVIEW

    def test_scores_bounded(self):
        with pytest.raises(Exception):
            ConflictResult(
                new_content="x",
                existing_memory_id="mem-1",
                existing_content="y",
                contradiction_score=1.5,  # Invalid: > 1.0
                entailment_score=0.0,
                neutral_score=0.0,
                suggested_resolution=ConflictResolution.MERGE,
            )

    def test_model_dump(self):
        result = ConflictResult(
            new_content="x",
            existing_memory_id="mem-1",
            existing_content="y",
            contradiction_score=0.8,
            entailment_score=0.1,
            neutral_score=0.1,
            suggested_resolution=ConflictResolution.SUPERSEDE,
            explanation="test explanation",
        )
        data = result.model_dump()
        assert data["suggested_resolution"] == "supersede"
        assert data["explanation"] == "test explanation"


class TestConflictResolutionEnum:
    """ConflictResolution strategy enum."""

    def test_all_strategies_exist(self):
        assert ConflictResolution.SUPERSEDE == "supersede"
        assert ConflictResolution.MERGE == "merge"
        assert ConflictResolution.FLAG_FOR_REVIEW == "flag_for_review"
        assert ConflictResolution.KEEP_EXISTING == "keep_existing"

    def test_string_representation(self):
        assert str(ConflictResolution.SUPERSEDE) == "ConflictResolution.SUPERSEDE"


# ──────────────────────────────────────────────
# ConflictDetector Tests (mocked — no model)
# ──────────────────────────────────────────────

class TestConflictDetectorInit:
    """Initialization and graceful degradation."""

    def test_unavailable_when_deps_missing(self):
        with patch(
            "muninn.conflict.detector.ConflictDetector._initialize"
        ) as mock_init:
            mock_init.return_value = None
            detector = ConflictDetector.__new__(ConflictDetector)
            detector.model_name = ConflictDetector.DEFAULT_MODEL
            detector.contradiction_threshold = 0.7
            detector.similarity_prefilter = 0.6
            detector._model = None
            detector._tokenizer = None
            detector._available = False
            assert detector.is_available is False

    def test_default_config_values(self):
        with patch(
            "muninn.conflict.detector.ConflictDetector._initialize"
        ):
            detector = ConflictDetector.__new__(ConflictDetector)
            detector.model_name = ConflictDetector.DEFAULT_MODEL
            detector.contradiction_threshold = 0.7
            detector.similarity_prefilter = 0.6
            detector._model = None
            detector._tokenizer = None
            detector._available = False
            assert detector.model_name == "cross-encoder/nli-deberta-v3-small"
            assert detector.contradiction_threshold == 0.7

    def test_custom_threshold(self):
        with patch(
            "muninn.conflict.detector.ConflictDetector._initialize"
        ):
            detector = ConflictDetector.__new__(ConflictDetector)
            detector.model_name = "custom-model"
            detector.contradiction_threshold = 0.9
            detector.similarity_prefilter = 0.5
            detector._available = False
            assert detector.contradiction_threshold == 0.9


class TestConflictDetectorDetection:
    """Detection logic tests using mocks."""

    def _make_detector(self, available=True):
        """Create a detector with mocked initialization."""
        detector = ConflictDetector.__new__(ConflictDetector)
        detector.model_name = "test-model"
        detector.contradiction_threshold = 0.7
        detector.similarity_prefilter = 0.6
        detector._model = MagicMock() if available else None
        detector._tokenizer = MagicMock() if available else None
        detector._available = available
        return detector

    def test_empty_content_returns_empty(self):
        detector = self._make_detector()
        result = detector.detect_conflicts("", [_make_record()])
        assert result == []

    def test_no_existing_memories_returns_empty(self):
        detector = self._make_detector()
        result = detector.detect_conflicts("new content", [])
        assert result == []

    def test_unavailable_returns_empty(self):
        detector = self._make_detector(available=False)
        result = detector.detect_conflicts(
            "new content",
            [_make_record("existing content")],
        )
        assert result == []

    def test_whitespace_content_returns_empty(self):
        detector = self._make_detector()
        result = detector.detect_conflicts("   \n\t  ", [_make_record()])
        assert result == []


class TestSuggestResolution:
    """Resolution strategy suggestion logic."""

    def _make_detector(self):
        detector = ConflictDetector.__new__(ConflictDetector)
        detector.model_name = "test"
        detector.contradiction_threshold = 0.7
        detector.similarity_prefilter = 0.6
        detector._available = True
        return detector

    def test_old_low_importance_supersede(self):
        detector = self._make_detector()
        # Old (>7 days) + low importance (<0.5) → SUPERSEDE
        old_record = _make_record(
            importance=0.3,
            created_at=time.time() - 86400 * 10,  # 10 days old
        )
        result = detector._suggest_resolution(
            existing=old_record,
            contradiction_score=0.8,
            entailment_score=0.1,
        )
        assert result == ConflictResolution.SUPERSEDE

    def test_partial_entailment_merge(self):
        detector = self._make_detector()
        # Moderate contradiction (<0.85) + partial entailment (>0.1) → MERGE
        recent_record = _make_record(
            importance=0.7,
            created_at=time.time() - 3600,  # 1 hour old
        )
        result = detector._suggest_resolution(
            existing=recent_record,
            contradiction_score=0.75,
            entailment_score=0.15,
        )
        assert result == ConflictResolution.MERGE

    def test_very_high_contradiction_flag(self):
        detector = self._make_detector()
        # Very high contradiction (>=0.85) with important recent memory → FLAG
        recent_record = _make_record(
            importance=0.8,
            created_at=time.time() - 3600,  # 1 hour old
        )
        result = detector._suggest_resolution(
            existing=recent_record,
            contradiction_score=0.92,
            entailment_score=0.03,
        )
        assert result == ConflictResolution.FLAG_FOR_REVIEW

    def test_recent_high_importance_not_superseded(self):
        detector = self._make_detector()
        # Recent (< 7 days) + high importance → should NOT be SUPERSEDE
        recent_important = _make_record(
            importance=0.9,
            created_at=time.time() - 3600,
        )
        result = detector._suggest_resolution(
            existing=recent_important,
            contradiction_score=0.75,
            entailment_score=0.05,
        )
        assert result != ConflictResolution.SUPERSEDE


class TestGenerateExplanation:
    """Explanation generation."""

    def test_strong_contradiction(self):
        result = ConflictDetector._generate_explanation(
            contradiction_score=0.90,
            resolution=ConflictResolution.FLAG_FOR_REVIEW,
            existing_content="The database runs on PostgreSQL.",
        )
        assert "Strong" in result
        assert "0.90" in result

    def test_moderate_contradiction(self):
        result = ConflictDetector._generate_explanation(
            contradiction_score=0.75,
            resolution=ConflictResolution.MERGE,
            existing_content="User prefers dark mode.",
        )
        assert "Moderate" in result

    def test_weak_contradiction(self):
        result = ConflictDetector._generate_explanation(
            contradiction_score=0.55,
            resolution=ConflictResolution.KEEP_EXISTING,
            existing_content="Test content.",
        )
        assert "Weak" in result

    def test_explanation_contains_resolution(self):
        result = ConflictDetector._generate_explanation(
            contradiction_score=0.85,
            resolution=ConflictResolution.SUPERSEDE,
            existing_content="Old fact.",
        )
        assert "Replace" in result or "action" in result.lower()


# ──────────────────────────────────────────────
# ConflictResolver Tests
# ──────────────────────────────────────────────

class TestConflictResolver:
    """Resolver execution tests."""

    def _make_resolver(self, existing_record=None):
        """Create a resolver with mocked stores."""
        metadata = MagicMock()
        metadata.get.return_value = existing_record
        vectors = MagicMock()
        graph = MagicMock()
        bm25 = MagicMock()
        return ConflictResolver(metadata, vectors, graph, bm25)

    def _make_conflict(self, resolution, contradiction_score=0.8):
        return ConflictResult(
            new_content="The sky is green.",
            existing_memory_id="mem-old",
            existing_content="The sky is blue.",
            contradiction_score=contradiction_score,
            entailment_score=0.1,
            neutral_score=0.1,
            suggested_resolution=resolution,
        )

    def test_supersede_reduces_importance(self):
        existing = _make_record("The sky is blue.", memory_id="mem-old", importance=0.8)
        resolver = self._make_resolver(existing_record=existing)
        new_record = _make_record("The sky is green.", memory_id="mem-new")

        conflict = self._make_conflict(ConflictResolution.SUPERSEDE)
        result = resolver.resolve(conflict, new_record=new_record)

        assert result["resolution"] == "supersede"
        assert result["superseded_memory_id"] == "mem-old"
        # Verify metadata.update was called to reduce importance
        resolver.metadata.update.assert_called_once()
        call_kwargs = resolver.metadata.update.call_args
        assert call_kwargs[1]["importance"] == pytest.approx(0.08, abs=0.01)

    def test_merge_combines_content(self):
        existing = _make_record("The sky is blue.", memory_id="mem-old")
        resolver = self._make_resolver(existing_record=existing)

        conflict = self._make_conflict(ConflictResolution.MERGE)
        result = resolver.resolve(conflict)

        assert result["resolution"] == "merge"
        assert result["skip_new_storage"] is True
        # Verify metadata was updated with merged content
        resolver.metadata.update.assert_called_once()

    def test_merge_no_existing_record_falls_back_to_flag(self):
        resolver = self._make_resolver(existing_record=None)
        conflict = self._make_conflict(ConflictResolution.MERGE)
        result = resolver.resolve(conflict)
        assert result["resolution"] == "flag_for_review"

    def test_keep_existing_skips_new(self):
        resolver = self._make_resolver()
        conflict = self._make_conflict(ConflictResolution.KEEP_EXISTING)
        result = resolver.resolve(conflict)
        assert result["resolution"] == "keep_existing"
        assert result["skip_new_storage"] is True
        assert result["kept_memory_id"] == "mem-old"

    def test_flag_for_review_allows_new_storage(self):
        resolver = self._make_resolver()
        conflict = self._make_conflict(ConflictResolution.FLAG_FOR_REVIEW)
        result = resolver.resolve(conflict)
        assert result["resolution"] == "flag_for_review"
        assert result["skip_new_storage"] is False

    def test_flag_contains_conflict_data(self):
        resolver = self._make_resolver()
        conflict = self._make_conflict(ConflictResolution.FLAG_FOR_REVIEW)
        result = resolver.resolve(conflict)
        assert "conflict" in result
        assert result["conflict"]["contradiction_score"] == 0.8

    def test_unknown_strategy_falls_back_to_flag(self):
        """If a new strategy is added but resolver doesn't handle it, flag."""
        resolver = self._make_resolver()
        conflict = self._make_conflict(ConflictResolution.FLAG_FOR_REVIEW)
        # Manually override to simulate unknown
        conflict.suggested_resolution = "nonexistent_strategy"
        result = resolver.resolve(conflict)
        assert result["resolution"] == "flag_for_review"

    def test_supersede_marks_metadata(self):
        existing = _make_record("old fact", memory_id="mem-old", importance=0.5)
        existing.metadata = {"source": "test"}
        resolver = self._make_resolver(existing_record=existing)
        new_record = _make_record("new fact", memory_id="mem-new")

        conflict = self._make_conflict(ConflictResolution.SUPERSEDE, 0.9)
        resolver.resolve(conflict, new_record=new_record)

        call_kwargs = resolver.metadata.update.call_args[1]
        metadata = call_kwargs["metadata"]
        assert "superseded_by" in metadata
        assert metadata["superseded_by"] == "mem-new"
        assert "superseded_at" in metadata

    def test_merge_updates_bm25(self):
        existing = _make_record("old content", memory_id="mem-old")
        resolver = self._make_resolver(existing_record=existing)

        conflict = self._make_conflict(ConflictResolution.MERGE)
        resolver.resolve(conflict)

        resolver.bm25.add.assert_called_once()
        call_args = resolver.bm25.add.call_args[0]
        assert call_args[0] == "mem-old"
        assert "Updated:" in call_args[1]

    def test_merge_preserves_user_scope_in_vector_payload(self):
        existing = _make_record("old content", memory_id="mem-old")
        resolver = self._make_resolver(existing_record=existing)
        resolver.embed_fn = lambda _content: [0.1, 0.2, 0.3]

        conflict = self._make_conflict(ConflictResolution.MERGE)
        resolver.resolve(conflict, user_id="user-123")

        resolver.vectors.upsert.assert_called_once()
        vector_payload = resolver.vectors.upsert.call_args[1]["metadata"]
        assert vector_payload["user_id"] == "user-123"
