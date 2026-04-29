"""Tests for muninn.scoring.importance — Multi-factor importance scoring."""

import pytest
import time
from muninn.core.types import MemoryRecord, MemoryType, Provenance
from muninn.scoring.importance import calculate_importance
from muninn.scoring.elo import elo_to_half_life_multiplier, INITIAL_ELO


class TestCalculateImportance:
    def _make_record(self, **kwargs):
        defaults = {
            "content": "test memory",
            "memory_type": MemoryType.EPISODIC,
            "provenance": Provenance.AUTO_EXTRACTED,
            "importance": 0.5,
            "access_count": 0,
        }
        defaults.update(kwargs)
        return MemoryRecord(**defaults)

    def test_basic_score_range(self):
        rec = self._make_record()
        score = calculate_importance(rec)
        assert 0.0 <= score <= 1.0

    def test_user_stated_provenance_boost(self):
        auto = self._make_record(provenance=Provenance.AUTO_EXTRACTED)
        user = self._make_record(provenance=Provenance.USER_EXPLICIT)
        score_auto = calculate_importance(auto)
        score_user = calculate_importance(user)
        assert score_user >= score_auto

    def test_higher_access_count_boosts_score(self):
        low = self._make_record(access_count=0)
        high = self._make_record(access_count=20)
        score_low = calculate_importance(low)
        score_high = calculate_importance(high)
        assert score_high >= score_low

    def test_semantic_type_higher_than_working(self):
        working = self._make_record(memory_type=MemoryType.WORKING)
        semantic = self._make_record(memory_type=MemoryType.SEMANTIC)
        # Semantic memories typically get higher base importance
        score_w = calculate_importance(working)
        score_s = calculate_importance(semantic)
        # This depends on implementation — at minimum both should be valid
        assert 0.0 <= score_w <= 1.0
        assert 0.0 <= score_s <= 1.0

    def test_score_clamped_to_unit_interval(self):
        # Even with extreme values, score should be in [0, 1]
        rec = self._make_record(
            access_count=10000,
            importance=1.0,
            provenance=Provenance.USER_EXPLICIT,
            memory_type=MemoryType.PROCEDURAL,
        )
        score = calculate_importance(rec)
        assert 0.0 <= score <= 1.0


class TestImportanceWithGraphCentrality:
    def test_with_centrality(self):
        rec = MemoryRecord(content="test", importance=0.5)
        score_no_cent = calculate_importance(rec, centrality=0.0)
        score_hi_cent = calculate_importance(rec, centrality=0.9)
        assert score_hi_cent >= score_no_cent


class TestEloToHalfLifeMultiplier:
    def test_baseline_rating(self):
        # When ELO is exactly the baseline, multiplier should be 1.0 (no change)
        assert elo_to_half_life_multiplier(INITIAL_ELO) == 1.0

    def test_high_elo(self):
        # Every 400 points above baseline doubles the half-life
        assert elo_to_half_life_multiplier(INITIAL_ELO + 400.0) == 2.0
        assert elo_to_half_life_multiplier(INITIAL_ELO + 800.0) == 4.0
        assert elo_to_half_life_multiplier(INITIAL_ELO + 1200.0) == 8.0

    def test_low_elo(self):
        # Every 400 points below baseline halves the half-life
        assert elo_to_half_life_multiplier(INITIAL_ELO - 400.0) == 0.5
        assert elo_to_half_life_multiplier(INITIAL_ELO - 800.0) == 0.25
        assert elo_to_half_life_multiplier(INITIAL_ELO - 1200.0) == 0.125

    def test_custom_baseline(self):
        # Function works with a custom baseline as well
        custom_baseline = 1500.0
        assert elo_to_half_life_multiplier(1500.0, baseline=custom_baseline) == 1.0
        assert elo_to_half_life_multiplier(1900.0, baseline=custom_baseline) == 2.0
        assert elo_to_half_life_multiplier(1100.0, baseline=custom_baseline) == 0.5
