from eval.gates import (
    check_latency_budget,
    check_track_coverage,
    compare_reports,
    compare_track_reports,
)


def test_compare_reports_detects_regression():
    baseline = {
        "cutoffs": {
            "@5": {"recall": 0.8, "mrr": 0.7, "ndcg": 0.75},
        }
    }
    current = {
        "cutoffs": {
            "@5": {"recall": 0.75, "mrr": 0.7, "ndcg": 0.70},
        }
    }
    violations = compare_reports(current, baseline, max_metric_regression=0.01)
    assert any("@5.recall" in v for v in violations)
    assert any("@5.ndcg" in v for v in violations)


def test_compare_reports_allows_within_budget():
    baseline = {
        "cutoffs": {
            "@10": {"recall": 0.5, "mrr": 0.4, "ndcg": 0.45},
        }
    }
    current = {
        "cutoffs": {
            "@10": {"recall": 0.49, "mrr": 0.39, "ndcg": 0.44},
        }
    }
    violations = compare_reports(current, baseline, max_metric_regression=0.02)
    assert violations == []


def test_check_latency_budget():
    report = {"latency_ms": {"p95": 95.0}}
    assert check_latency_budget(report, max_p95_ms=100.0) == []
    violations = check_latency_budget(report, max_p95_ms=50.0)
    assert len(violations) == 1
    assert "exceeds budget" in violations[0]


def test_compare_track_reports_detects_regression():
    baseline = {
        "tracks": {
            "accurate_retrieval": {
                "cutoffs": {
                    "@5": {"recall": 0.9, "mrr": 0.8, "ndcg": 0.85},
                }
            }
        }
    }
    current = {
        "tracks": {
            "accurate_retrieval": {
                "cutoffs": {
                    "@5": {"recall": 0.88, "mrr": 0.79, "ndcg": 0.83},
                }
            }
        }
    }
    violations = compare_track_reports(current, baseline, max_metric_regression=0.005)
    assert any("track accurate_retrieval @5.recall" in v for v in violations)
    assert any("track accurate_retrieval @5.ndcg" in v for v in violations)


def test_check_track_coverage():
    report = {
        "tracks": {
            "accurate_retrieval": {"cases": 22},
            "test_time_learning": {"cases": 4},
        }
    }
    violations = check_track_coverage(
        report,
        required_track_cases={
            "accurate_retrieval": 20,
            "test_time_learning": 6,
            "long_range_understanding": 100,
        },
    )
    assert any("test_time_learning" in v for v in violations)
    assert any("long_range_understanding" in v for v in violations)
