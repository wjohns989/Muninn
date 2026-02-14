"""Tests for muninn.core.feature_flags — Centralized feature flag management."""

import os
import pytest
from muninn.core.feature_flags import (
    FeatureFlags,
    get_flags,
    reset_flags,
)


class TestFeatureFlagsDefaults:
    """Test default flag values match Phase 1/2/3 expectations."""

    def test_phase1_flags_default_on(self):
        """Phase 1 features should be ON by default (low cost, high value)."""
        flags = FeatureFlags()
        assert flags.explainable_recall is True
        assert flags.instructor_extraction is True
        assert flags.platform_abstraction is True
        assert flags.goal_compass is True

    def test_phase2_flags_default_off(self):
        """Phase 2 features should be OFF by default (higher cost, opt-in)."""
        flags = FeatureFlags()
        assert flags.conflict_detection is False
        assert flags.semantic_dedup is False
        assert flags.adaptive_weights is False
        assert flags.retrieval_feedback is False

    def test_phase3_flags_default_off(self):
        """Phase 3 features should be OFF by default (require additional deps)."""
        flags = FeatureFlags()
        assert flags.memory_chains is False
        assert flags.multi_source_ingestion is False
        assert flags.python_sdk is False
        assert flags.otel_genai is False


class TestFeatureFlagsFromEnv:
    """Test env var driven flag construction."""

    def setup_method(self):
        reset_flags()

    def teardown_method(self):
        reset_flags()
        # Clean up env vars
        for key in list(os.environ.keys()):
            if key.startswith("MUNINN_"):
                os.environ.pop(key, None)

    def test_from_env_reads_env_vars(self):
        os.environ["MUNINN_EXPLAIN_RECALL"] = "0"
        os.environ["MUNINN_CONFLICT_DETECTION"] = "1"
        flags = FeatureFlags.from_env()
        assert flags.explainable_recall is False
        assert flags.conflict_detection is True

    def test_from_env_missing_vars_use_defaults(self):
        flags = FeatureFlags.from_env()
        assert flags.explainable_recall is True  # default "1"
        assert flags.conflict_detection is False  # default "0"

    def test_from_env_truthy_values(self):
        """'1', 'true', 'yes', 'on' should all resolve to True."""
        os.environ["MUNINN_EXPLAIN_RECALL"] = "yes"
        flags = FeatureFlags.from_env()
        assert flags.explainable_recall is True

    def test_from_env_falsy_values(self):
        """Non-truthy values should resolve to False."""
        os.environ["MUNINN_EXPLAIN_RECALL"] = "nope"
        flags = FeatureFlags.from_env()
        assert flags.explainable_recall is False


class TestFeatureFlagsImmutability:
    """Test frozen dataclass behavior."""

    def test_flags_are_frozen(self):
        flags = FeatureFlags()
        with pytest.raises(AttributeError):
            flags.explainable_recall = False

    def test_flags_are_hashable(self):
        """Frozen dataclasses are hashable — useful for caching."""
        flags = FeatureFlags()
        assert hash(flags) is not None


class TestFeatureFlagsUtilities:
    """Test utility methods."""

    def test_is_enabled(self):
        flags = FeatureFlags()
        assert flags.is_enabled("explainable_recall") is True
        assert flags.is_enabled("conflict_detection") is False

    def test_is_enabled_unknown_flag_raises(self):
        flags = FeatureFlags()
        with pytest.raises(AttributeError, match="nonexistent_flag"):
            flags.is_enabled("nonexistent_flag")

    def test_require_enabled(self):
        flags = FeatureFlags()
        # Should not raise for enabled flag
        flags.require("explainable_recall")

    def test_require_disabled_raises(self):
        flags = FeatureFlags()
        with pytest.raises(RuntimeError, match="conflict_detection"):
            flags.require("conflict_detection")

    def test_active_flags(self):
        flags = FeatureFlags()
        active = flags.active_flags
        assert "explainable_recall" in active
        assert "instructor_extraction" in active
        assert "platform_abstraction" in active
        assert "goal_compass" in active
        assert "conflict_detection" not in active
        assert "retrieval_feedback" not in active

    def test_to_dict(self):
        flags = FeatureFlags()
        d = flags.to_dict()
        assert isinstance(d, dict)
        assert d["explainable_recall"] is True
        assert d["conflict_detection"] is False
        assert len(d) == 12  # 4 phase1 + 4 phase2 + 4 phase3


class TestSingletonBehavior:
    """Test module-level singleton pattern."""

    def setup_method(self):
        reset_flags()

    def teardown_method(self):
        reset_flags()

    def test_get_flags_returns_same_instance(self):
        f1 = get_flags()
        f2 = get_flags()
        assert f1 is f2

    def test_reset_flags_clears_singleton(self):
        f1 = get_flags()
        reset_flags()
        f2 = get_flags()
        # New instance (may be equal but not same object)
        assert f1 is not f2
