"""Tests for muninn.core.recall_trace — Explainable recall traces."""

import pytest
from muninn.core.recall_trace import (
    SignalContribution,
    RecallTrace,
    create_signal_contribution,
    explain_vector_signal,
    explain_bm25_signal,
    explain_graph_signal,
    explain_temporal_signal,
    SIGNAL_EXPLAINERS,
)


class TestSignalContribution:
    """Test individual signal contribution model."""

    def test_construction(self):
        sc = SignalContribution(
            signal="vector",
            raw_score=0.91,
            rank=0,
            rrf_contribution=0.0163,
            weight=1.0,
            explanation="High semantic similarity (0.91) at rank #0",
        )
        assert sc.signal == "vector"
        assert sc.raw_score == 0.91
        assert sc.rank == 0
        assert sc.rrf_contribution == pytest.approx(0.0163)
        assert sc.weight == 1.0

    def test_serialization(self):
        sc = SignalContribution(
            signal="bm25",
            raw_score=4.2,
            rank=1,
            rrf_contribution=0.0129,
            weight=0.8,
            explanation="Moderate keyword match",
        )
        d = sc.model_dump()
        assert isinstance(d, dict)
        assert d["signal"] == "bm25"
        assert d["raw_score"] == 4.2


class TestRecallTrace:
    """Test complete recall trace model."""

    def _make_trace(self) -> RecallTrace:
        """Build a sample trace with multiple signals."""
        trace = RecallTrace(memory_id="abc-123")
        trace.signals = [
            create_signal_contribution("vector", 0.91, 0, 0.0163, 1.0),
            create_signal_contribution("bm25", 4.2, 0, 0.0130, 0.8),
            create_signal_contribution("graph", 1.0, 2, 0.0159, 1.0),
            create_signal_contribution("temporal", 0.72, 5, 0.0076, 0.5),
        ]
        return trace

    def test_compute_final_score(self):
        trace = self._make_trace()
        trace.compute_final_score()
        expected = 0.0163 + 0.0130 + 0.0159 + 0.0076
        assert trace.final_score == pytest.approx(expected, abs=1e-4)

    def test_determine_dominant_signal(self):
        trace = self._make_trace()
        trace.determine_dominant_signal()
        assert trace.dominant_signal == "vector"  # 0.0163 is highest

    def test_generate_explanation(self):
        trace = self._make_trace()
        trace.generate_explanation()
        assert "vector" in trace.explanation.lower()
        assert len(trace.explanation) > 20

    def test_finalize(self):
        trace = self._make_trace()
        trace.finalize()
        assert trace.final_score > 0
        assert trace.dominant_signal != "unknown"
        assert len(trace.explanation) > 0

    def test_importance_boost_in_explanation(self):
        trace = self._make_trace()
        trace.importance_boost = 0.15
        trace.generate_explanation()
        assert "boost" in trace.explanation.lower()

    def test_rerank_score_in_explanation(self):
        trace = self._make_trace()
        trace.rerank_score = 0.847
        trace.generate_explanation()
        assert "rerank" in trace.explanation.lower()

    def test_empty_trace(self):
        trace = RecallTrace(memory_id="empty")
        trace.finalize()
        assert trace.final_score == 0.0
        assert "no retrieval" in trace.explanation.lower()

    def test_to_dict(self):
        trace = self._make_trace()
        trace.finalize()
        d = trace.to_dict()
        assert isinstance(d, dict)
        assert d["memory_id"] == "abc-123"
        assert len(d["signals"]) == 4
        assert d["dominant_signal"] == "vector"

    def test_single_signal_trace(self):
        trace = RecallTrace(memory_id="single")
        trace.signals = [
            create_signal_contribution("vector", 0.85, 0, 0.0163, 1.0),
        ]
        trace.finalize()
        assert trace.dominant_signal == "vector"
        assert trace.final_score == pytest.approx(0.0163)
        assert "vector" in trace.explanation.lower()


class TestExplanationFunctions:
    """Test signal-specific explanation generators."""

    def test_vector_high(self):
        e = explain_vector_signal(0.92, 0)
        assert "High" in e
        assert "0.92" in e

    def test_vector_moderate(self):
        e = explain_vector_signal(0.75, 2)
        assert "Moderate" in e

    def test_vector_weak(self):
        e = explain_vector_signal(0.55, 5)
        assert "Weak" in e

    def test_vector_marginal(self):
        e = explain_vector_signal(0.30, 10)
        assert "Marginal" in e

    def test_bm25_strong(self):
        e = explain_bm25_signal(6.0, 0)
        assert "Strong" in e

    def test_bm25_moderate(self):
        e = explain_bm25_signal(3.0, 1)
        assert "Moderate" in e

    def test_bm25_weak(self):
        e = explain_bm25_signal(1.5, 3)
        assert "Weak" in e

    def test_graph_signal(self):
        e = explain_graph_signal(1.0, 2)
        assert "graph" in e.lower()

    def test_temporal_signal(self):
        e = explain_temporal_signal(0.72, 5)
        assert "Temporal" in e


class TestCreateSignalContribution:
    """Test the factory function."""

    def test_known_signal(self):
        sc = create_signal_contribution("vector", 0.91, 0, 0.0163, 1.0)
        assert sc.signal == "vector"
        assert "similarity" in sc.explanation.lower()

    def test_unknown_signal(self):
        sc = create_signal_contribution("custom", 0.5, 3, 0.01, 0.5)
        assert sc.signal == "custom"
        assert "rank #3" in sc.explanation

    def test_all_known_signals(self):
        for signal_name in SIGNAL_EXPLAINERS:
            sc = create_signal_contribution(signal_name, 0.5, 1, 0.01, 1.0)
            assert sc.signal == signal_name
            assert len(sc.explanation) > 0
