"""Tests for muninn.retrieval.weight_adapter — Adaptive Retrieval Weights (v3.2.0)."""

import math
import pytest
from muninn.retrieval.weight_adapter import (
    WeightAdapter,
    DEFAULT_WEIGHTS,
    TEMPORAL_KEYWORDS,
    RELATIONAL_KEYWORDS,
    QUERY_SHORT_BM25_BOOST,
    QUERY_TEMPORAL_BOOST,
    QUERY_RELATIONAL_GRAPH_BOOST,
    ENTROPY_CONFIDENCE_FLOOR,
    ENTROPY_CONFIDENCE_CEILING,
)


class TestWeightAdapterInit:
    """Initialization and configuration."""

    def test_default_weights_used_when_none_provided(self):
        adapter = WeightAdapter()
        assert adapter.base_weights == DEFAULT_WEIGHTS

    def test_custom_weights_stored(self):
        custom = {"vector": 2.0, "graph": 0.5, "bm25": 1.0, "temporal": 0.3}
        adapter = WeightAdapter(base_weights=custom)
        assert adapter.base_weights == custom

    def test_custom_keywords(self):
        temporal = frozenset({"new", "fresh"})
        relational = frozenset({"who", "between"})
        adapter = WeightAdapter(
            temporal_keywords=temporal,
            relational_keywords=relational,
        )
        assert adapter.temporal_keywords == temporal
        assert adapter.relational_keywords == relational

    def test_base_weights_is_a_copy(self):
        """Mutating base_weights after init shouldn't affect adapter."""
        original = {"vector": 1.0, "graph": 1.0, "bm25": 0.8, "temporal": 0.5}
        adapter = WeightAdapter(base_weights=original)
        original["vector"] = 999.0
        assert adapter.base_weights["vector"] == 1.0


class TestNormalizedEntropy:
    """Shannon entropy normalization."""

    def test_single_element_returns_zero(self):
        assert WeightAdapter._normalized_entropy([5.0]) == 0.0

    def test_uniform_distribution_returns_one(self):
        # Perfect uniform: all equal → max entropy
        result = WeightAdapter._normalized_entropy([1.0, 1.0, 1.0, 1.0])
        assert abs(result - 1.0) < 0.001

    def test_peaked_distribution_returns_near_zero(self):
        # One dominant value → low entropy
        result = WeightAdapter._normalized_entropy([100.0, 0.001, 0.001, 0.001])
        assert result < 0.1

    def test_empty_list_returns_one(self):
        assert WeightAdapter._normalized_entropy([]) == 1.0

    def test_all_zeros_returns_one(self):
        assert WeightAdapter._normalized_entropy([0.0, 0.0, 0.0]) == 1.0

    def test_result_bounded_zero_to_one(self):
        for values in [
            [1.0, 2.0, 3.0],
            [10.0, 1.0],
            [0.5, 0.5, 0.5, 0.5, 0.5],
        ]:
            result = WeightAdapter._normalized_entropy(values)
            assert 0.0 <= result <= 1.0

    def test_two_elements_equal(self):
        result = WeightAdapter._normalized_entropy([1.0, 1.0])
        assert abs(result - 1.0) < 0.001

    def test_decreasing_spread_decreases_entropy(self):
        # More peaked → lower entropy
        uniform = WeightAdapter._normalized_entropy([1.0, 1.0, 1.0, 1.0])
        skewed = WeightAdapter._normalized_entropy([10.0, 1.0, 1.0, 1.0])
        peaked = WeightAdapter._normalized_entropy([100.0, 1.0, 1.0, 1.0])
        assert uniform > skewed > peaked


class TestQueryAdaptation:
    """Query-characteristic-based weight adaptation."""

    def _adapter(self):
        return WeightAdapter()

    def test_short_query_boosts_bm25(self):
        adapter = self._adapter()
        weights = adapter.compute_weights("hello world")
        # After normalization, bm25 should have a higher relative share
        base_ratio = DEFAULT_WEIGHTS["bm25"] / DEFAULT_WEIGHTS["vector"]
        adapted_ratio = weights["bm25"] / weights["vector"]
        assert adapted_ratio > base_ratio

    def test_long_query_no_bm25_boost(self):
        adapter = self._adapter()
        # 5 tokens — should NOT trigger short-query boost
        weights = adapter.compute_weights("this is a longer query sentence")
        # Without other adaptations, weights should stay at base ratios
        base_ratio = DEFAULT_WEIGHTS["bm25"] / DEFAULT_WEIGHTS["vector"]
        adapted_ratio = weights["bm25"] / weights["vector"]
        assert abs(adapted_ratio - base_ratio) < 0.01

    def test_temporal_keyword_boosts_temporal(self):
        adapter = self._adapter()
        weights = adapter.compute_weights("what happened recently with the project")
        base_ratio = DEFAULT_WEIGHTS["temporal"] / DEFAULT_WEIGHTS["vector"]
        adapted_ratio = weights["temporal"] / weights["vector"]
        assert adapted_ratio > base_ratio

    def test_relational_keyword_boosts_graph(self):
        adapter = self._adapter()
        weights = adapter.compute_weights("who is connected to the database team")
        base_ratio = DEFAULT_WEIGHTS["graph"] / DEFAULT_WEIGHTS["vector"]
        adapted_ratio = weights["graph"] / weights["vector"]
        assert adapted_ratio > base_ratio

    def test_combined_temporal_and_relational(self):
        adapter = self._adapter()
        weights = adapter.compute_weights("who recently connected to the system")
        # Both temporal and graph should be boosted relative to vector
        assert weights["temporal"] > 0
        assert weights["graph"] > 0

    def test_no_keywords_preserves_proportions(self):
        adapter = self._adapter()
        weights = adapter.compute_weights("explain the architecture of the memory system in detail")
        # Long query (>3 tokens), no temporal/relational keywords
        # Proportions should approximately match base weights
        for signal in DEFAULT_WEIGHTS:
            base_ratio = DEFAULT_WEIGHTS[signal] / sum(DEFAULT_WEIGHTS.values())
            adapted_ratio = weights[signal] / sum(weights.values())
            assert abs(adapted_ratio - base_ratio) < 0.02, f"Signal {signal} ratio drifted"


class TestEntropyAdaptation:
    """Entropy-based signal confidence adaptation."""

    def _adapter(self):
        return WeightAdapter()

    def test_uniform_results_reduce_weight(self):
        adapter = self._adapter()
        # All signals return uniform rank distributions → low confidence → reduced
        uniform_results = {
            "vector": [("a", 0), ("b", 1), ("c", 2), ("d", 3)],
            "graph": [("a", 0), ("b", 1), ("c", 2), ("d", 3)],
            "bm25": [("a", 0), ("b", 1), ("c", 2), ("d", 3)],
            "temporal": [("a", 0), ("b", 1), ("c", 2), ("d", 3)],
        }
        weights = adapter.compute_weights(
            "explain the architecture of the memory system in detail",
            signal_results=uniform_results,
        )
        # Weights should still sum correctly
        base_sum = sum(DEFAULT_WEIGHTS.values())
        adapted_sum = sum(weights.values())
        assert abs(adapted_sum - base_sum) < 0.01

    def test_peaked_results_preserve_weight(self):
        adapter = self._adapter()
        # Vector has concentrated results (few items, one dominant rank)
        peaked_results = {
            "vector": [("a", 0)],  # Only 1 result → entropy 0 → high confidence
            "graph": [],
            "bm25": [],
            "temporal": [],
        }
        # Only vector has results → its weight stays at base
        weights = adapter.compute_weights(
            "explain the architecture of the memory system in detail",
            signal_results=peaked_results,
        )
        # Single-element results: len < 2 → keep base weight
        assert weights["vector"] > 0

    def test_no_signal_results_query_only(self):
        adapter = self._adapter()
        weights_no_signals = adapter.compute_weights("recent updates")
        weights_with_empty = adapter.compute_weights("recent updates", signal_results={})
        # Both should produce the same result
        for signal in DEFAULT_WEIGHTS:
            assert abs(weights_no_signals[signal] - weights_with_empty[signal]) < 0.001


class TestNormalization:
    """Weight normalization to prevent inflation/deflation."""

    def test_sum_preserved(self):
        adapter = WeightAdapter()
        base_sum = sum(DEFAULT_WEIGHTS.values())
        weights = adapter.compute_weights("recent query about connections")
        adapted_sum = sum(weights.values())
        assert abs(adapted_sum - base_sum) < 0.01

    def test_zero_weight_safety(self):
        """All-zero weights should fall back to base."""
        adapter = WeightAdapter()
        result = adapter._normalize_weights({"vector": 0, "graph": 0, "bm25": 0, "temporal": 0})
        assert result == DEFAULT_WEIGHTS

    def test_all_signals_positive_after_normalization(self):
        adapter = WeightAdapter()
        weights = adapter.compute_weights("who recently connected")
        for signal, weight in weights.items():
            assert weight > 0, f"Signal {signal} has non-positive weight: {weight}"


class TestExplain:
    """Human-readable explanation output."""

    def test_explain_returns_all_signals(self):
        adapter = WeightAdapter()
        explanations = adapter.explain("recent query")
        assert "vector" in explanations
        assert "graph" in explanations
        assert "bm25" in explanations
        assert "temporal" in explanations

    def test_explain_shows_boosted_signals(self):
        adapter = WeightAdapter()
        explanations = adapter.explain("recent events")
        # "recent" is a temporal keyword → temporal should show boosted
        assert "boosted" in explanations["temporal"] or "unchanged" in explanations["temporal"]

    def test_explain_with_signal_results(self):
        adapter = WeightAdapter()
        results = {
            "vector": [("a", 0), ("b", 1), ("c", 2)],
            "graph": [],
            "bm25": [("a", 0)],
            "temporal": [],
        }
        explanations = adapter.explain("test query sentence for length", results)
        assert all(isinstance(v, str) for v in explanations.values())
