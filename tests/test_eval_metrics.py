"""Tests for eval.metrics retrieval metrics."""

from eval.metrics import (
    evaluate_batch,
    evaluate_case,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    summarize_latency_ms,
)


def test_recall_at_k():
    relevant = {"a", "b"}
    ranked = ["x", "a", "y", "b"]
    assert recall_at_k(relevant, ranked, 1) == 0.0
    assert recall_at_k(relevant, ranked, 2) == 0.5
    assert recall_at_k(relevant, ranked, 4) == 1.0


def test_recall_at_k_ignores_duplicate_hits():
    relevant = {"a", "b"}
    ranked = ["a", "a", "a", "b"]
    assert recall_at_k(relevant, ranked, 3) == 0.5
    assert recall_at_k(relevant, ranked, 4) == 1.0


def test_mrr_at_k():
    relevant = {"a", "b"}
    ranked = ["x", "a", "b"]
    assert mrr_at_k(relevant, ranked, 1) == 0.0
    assert mrr_at_k(relevant, ranked, 3) == 0.5


def test_ndcg_at_k_binary():
    relevant = {"a", "b"}
    ranked = ["a", "x", "b"]
    value = ndcg_at_k(relevant, ranked, 3)
    assert 0.0 < value <= 1.0


def test_ndcg_at_k_ignores_duplicate_relevant_ids():
    relevant = {"a"}
    ranked = ["a", "a", "a"]
    assert ndcg_at_k(relevant, ranked, 3) == 1.0


def test_evaluate_case():
    metrics = evaluate_case(relevant_ids=["a"], ranked_ids=["a", "b"], k=2)
    assert metrics["recall"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["ndcg"] == 1.0


def test_evaluate_batch():
    report = evaluate_batch(
        [
            {"relevant_ids": ["a"], "ranked_ids": ["a", "b"], "latency_ms": 10.0},
            {"relevant_ids": ["z"], "ranked_ids": ["x", "y"], "latency_ms": 30.0},
        ],
        ks=(1, 2),
    )
    assert report["cases"] == 2
    assert "@1" in report["cutoffs"]
    assert "@2" in report["cutoffs"]
    assert 0.0 <= report["cutoffs"]["@1"]["recall"] <= 1.0
    assert report["latency_ms"]["p95"] >= report["latency_ms"]["p50"]


def test_summarize_latency_ms():
    summary = summarize_latency_ms([12.0, 20.0, 100.0, 50.0])
    assert summary["count"] == 4.0
    assert summary["avg"] > 0.0
    assert summary["p95"] >= summary["p50"]


def test_evaluate_batch_includes_track_breakdown():
    report = evaluate_batch(
        [
            {"relevant_ids": ["a"], "ranked_ids": ["a"], "latency_ms": 10.0, "track": "accurate_retrieval"},
            {"relevant_ids": ["b"], "ranked_ids": ["x"], "latency_ms": 30.0, "track": "selective_forgetting"},
            {"relevant_ids": ["c"], "ranked_ids": ["c"], "latency_ms": 20.0, "track": "accurate_retrieval"},
        ],
        ks=(1,),
    )

    assert "tracks" in report
    assert report["tracks"]["accurate_retrieval"]["cases"] == 2
    assert report["tracks"]["selective_forgetting"]["cases"] == 1
    assert report["tracks"]["accurate_retrieval"]["cutoffs"]["@1"]["recall"] == 1.0
    assert report["tracks"]["selective_forgetting"]["cutoffs"]["@1"]["recall"] == 0.0
