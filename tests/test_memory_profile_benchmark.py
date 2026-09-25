from __future__ import annotations

import math
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_choose_isolated_port_never_returns_production_port():
    from eval.memory_profile_benchmark import PRODUCTION_PORT, choose_isolated_port

    port = choose_isolated_port()

    assert port != PRODUCTION_PORT
    assert 1 <= port <= 65535
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", port))


def test_isolation_guard_accepts_loopback_temp_storage(tmp_path):
    from eval.memory_profile_benchmark import IsolationGuard

    data_dir = tmp_path / "data"
    IsolationGuard(root=tmp_path, data_dir=data_dir, host="127.0.0.1", port=42170).validate()


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.com"])
def test_isolation_guard_rejects_non_loopback_hosts(tmp_path, host):
    from eval.memory_profile_benchmark import IsolationGuard

    with pytest.raises(ValueError, match="loopback"):
        IsolationGuard(root=tmp_path, data_dir=tmp_path / "data", host=host, port=42170).validate()


def test_isolation_guard_rejects_production_port(tmp_path):
    from eval.memory_profile_benchmark import PRODUCTION_PORT, IsolationGuard

    with pytest.raises(ValueError, match="production port"):
        IsolationGuard(
            root=tmp_path,
            data_dir=tmp_path / "data",
            host="127.0.0.1",
            port=PRODUCTION_PORT,
        ).validate()


def test_isolation_guard_rejects_storage_outside_root(tmp_path):
    from eval.memory_profile_benchmark import IsolationGuard

    with pytest.raises(ValueError, match="temporary benchmark root"):
        IsolationGuard(
            root=tmp_path / "root",
            data_dir=tmp_path / "outside",
            host="127.0.0.1",
            port=42170,
        ).validate()


def test_child_environment_is_minimal_and_offline(tmp_path):
    from eval.memory_profile_benchmark import build_child_environment

    inherited = {
        "PATH": "safe-path",
        "SYSTEMROOT": "C:/Windows",
        "OPENAI_API_KEY": "must-not-leak",
        "ANTHROPIC_API_KEY": "must-not-leak",
        "MUNINN_DATA_DIR": "C:/production",
        "MUNINN_AUTH_TOKEN": "production-token",
        "SCHWAB_TOKEN_PATH": "C:/private/token.json",
    }

    env = build_child_environment(
        inherited=inherited,
        root=tmp_path,
        data_dir=tmp_path / "data",
        fixtures_dir=tmp_path / "fixtures",
        port=42170,
        benchmark_token="synthetic-only-token",
        reranker_enabled=False,
        conflict_detection_enabled=False,
    )

    assert env["PATH"] == "safe-path"
    assert env["MUNINN_DATA_DIR"] == str(tmp_path / "data")
    assert env["MUNINN_PORT"] == "42170"
    assert env["MUNINN_AUTH_TOKEN"] == "synthetic-only-token"
    assert env["MUNINN_SERVER_AUTH_TOKEN"] == "synthetic-only-token"
    assert env["MUNINN_API_KEY"] == "synthetic-only-token"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["MUNINN_LEGACY_DISCOVERY_ENABLED"] == "false"
    assert env["MUNINN_MULTI_SOURCE_INGESTION"] == "1"
    assert env["MUNINN_MCP_AUTOSTART_ON_LAUNCH"] == "0"
    assert env["MUNINN_OLLAMA_URL"] == "http://127.0.0.1:1"
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "SCHWAB_TOKEN_PATH" not in env
    assert "production-token" not in env.values()


def test_quality_metrics_for_ranked_results():
    from eval.memory_profile_benchmark import compute_quality_metrics

    metrics = compute_quality_metrics(
        [
            ("expected-a", ["expected-a", "other"]),
            ("expected-b", ["other", "expected-b"]),
            ("expected-c", ["other"]),
        ],
        k=5,
    )

    assert metrics["query_count"] == 3
    assert metrics["recall_at_k"] == pytest.approx(2 / 3)
    assert metrics["mrr_at_k"] == pytest.approx((1 + 0.5 + 0) / 3)
    expected_ndcg = (1.0 + (1.0 / math.log2(3)) + 0.0) / 3
    assert metrics["ndcg_at_k"] == pytest.approx(expected_ndcg)


def test_sample_summary_reports_peak_steady_and_retained_values():
    from eval.memory_profile_benchmark import ProcessSample, summarize_samples

    samples = [
        ProcessSample(0.0, "startup", 100, 400, 80, 20, 5, 1, 10, 15, 0),
        ProcessSample(1.0, "workload", 300, 900, 250, 24, 7, 2, 50, 70, 20),
        ProcessSample(2.0, "post_workload", 180, 650, 140, 22, 6, 1, 35, 55, 10),
        ProcessSample(3.0, "post_workload", 160, 620, 130, 21, 6, 1, 30, 50, 8),
    ]

    summary = summarize_samples(samples)

    assert summary["peak"]["working_set_bytes"] == 300
    assert summary["peak"]["private_bytes"] == 900
    assert summary["post_workload"]["working_set_bytes"] == 170
    assert summary["post_workload"]["private_bytes"] == 635
    assert summary["post_workload"]["python_heap_bytes"] == 32.5


def test_phase_summary_reports_mean_and_peak_memory():
    from eval.memory_profile_benchmark import ProcessSample, summarize_phases

    samples = [
        ProcessSample(0.0, "warm_idle", 100, 400, 80, 20, 5, 1, 10, 15, 30),
        ProcessSample(1.0, "warm_idle", 140, 500, 100, 22, 7, 3, 30, 45, 50),
        ProcessSample(2.0, "ingestion", 300, 900, 250, 24, 8, 4, 60, 75, 90),
    ]

    phases = summarize_phases(samples)

    assert phases["warm_idle"]["sample_count"] == 2
    assert phases["warm_idle"]["mean"]["working_set_bytes"] == 120
    assert phases["warm_idle"]["peak"]["private_bytes"] == 500
    assert phases["ingestion"]["peak"]["python_heap_bytes"] == 60


def test_public_report_redacts_benchmark_token(tmp_path):
    from eval.memory_profile_benchmark import sanitize_for_report

    payload = {
        "token": "synthetic-only-token",
        "nested": {"authorization": "Bearer synthetic-only-token"},
        "path": Path(tmp_path) / "data",
    }

    sanitized = sanitize_for_report(payload, secrets=["synthetic-only-token"])

    rendered = repr(sanitized)
    assert "synthetic-only-token" not in rendered
    assert "[REDACTED]" in rendered
    assert sanitized["path"] == str(Path(tmp_path) / "data")


def test_benchmark_server_request_annotation_is_globally_resolvable():
    import eval.memory_profile_server as benchmark_server

    # With postponed annotations, FastAPI resolves Request in module globals.
    # A function-local import turns it into a required query parameter instead.
    assert benchmark_server.Request.__name__ == "Request"


def test_ranked_id_extraction_supports_public_search_rows():
    from eval.memory_profile_benchmark import _extract_ranked_ids

    response = {
        "success": True,
        "data": [
            {"id": "public-row-id", "memory": "plain memory content"},
            {"memory": {"id": "nested-row-id", "content": "legacy nested shape"}},
        ],
    }

    assert _extract_ranked_ids(response) == ["public-row-id", "nested-row-id"]


def test_resource_snapshot_reports_bm25_size_and_lazy_integrity_state():
    from eval.memory_profile_server import _resource_state
    from muninn.retrieval.bm25 import BM25Index

    bm25 = BM25Index()
    for index in range(3):
        bm25.add(f"memory-{index}", f"synthetic document {index}")
    server = SimpleNamespace(
        memory=SimpleNamespace(
            _initialized=True,
            _embed_model=None,
            _reranker=None,
            _conflict_detector=None,
            _consolidation=SimpleNamespace(_conflict_detector=None),
            _vectors=object(),
            _graph=object(),
            _metadata=object(),
            _feedback_multiplier_cache={},
            _bm25=bm25,
        )
    )

    resources = _resource_state(server)

    assert resources["bm25_document_count"] == 3
    assert resources["consolidation_conflict_detector_loaded"] is False


def test_benchmark_defaults_to_separate_thirty_minute_workload_and_idle_soaks(tmp_path):
    from eval.memory_profile_benchmark import DEFAULT_SOAK_SECONDS, build_parser

    args = build_parser().parse_args(["--output", str(tmp_path / "report.json")])

    assert args.soak_seconds == DEFAULT_SOAK_SECONDS
    assert args.idle_soak_seconds == DEFAULT_SOAK_SECONDS
