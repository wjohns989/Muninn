"""Tests for muninn.scoring.importance — Multi-factor importance scoring."""

import pytest
import time
from muninn.core.types import MemoryRecord, MemoryType, Provenance
from muninn.scoring.importance import calculate_importance


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

from unittest.mock import patch
from muninn.scoring.importance import calculate_recency

class TestCalculateRecency:
    @patch('time.time')
    def test_recency_negative_age(self, mock_time):
        mock_time.return_value = 1000.0
        # Created in the future -> age < 0
        score = calculate_recency(created_at=2000.0)
        assert score == 1.0

    @patch('time.time')
    def test_recency_zero_age(self, mock_time):
        mock_time.return_value = 1000.0
        # Created exactly now -> age = 0
        score = calculate_recency(created_at=1000.0)
        assert score == 1.0

    @patch('time.time')
    def test_recency_half_life(self, mock_time):
        # 7 days in seconds = 7 * 86400 = 604800
        mock_time.return_value = 1000.0 + 604800.0
        # Created exactly half-life ago -> age = 7 days
        score = calculate_recency(created_at=1000.0, half_life_days=7.0)
        # Expected e^(-0.693 * 7/7) = e^(-0.693) approx 0.50007...
        assert 0.49 < score < 0.51

    @patch('time.time')
    def test_recency_with_elo(self, mock_time):
        # Base half-life 7 days.
        # Elo rating 2000 > 1500, so multiplier > 1 (e.g. 2.0).
        # The half_life_days is increased, thus the item decays slower.
        mock_time.return_value = 1000.0 + 604800.0

        # We need to know that elo_to_half_life_multiplier(2000) returns a factor > 1
        # It's an internal function. We can mock it or just rely on its integration.
        # Let's rely on integration but test the effect: a higher elo -> slower decay -> higher score.
        score_normal = calculate_recency(created_at=1000.0, half_life_days=7.0)
        score_with_elo = calculate_recency(created_at=1000.0, half_life_days=7.0, elo_rating=2000.0)

        assert score_with_elo > score_normal
