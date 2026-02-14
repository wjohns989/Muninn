"""Tests for muninn.core.config — Configuration management."""

import os
import pytest
from muninn.core.config import (
    MuninnConfig,
    EmbeddingConfig,
    VectorConfig,
    GraphConfig,
    MetadataConfig,
    ExtractionConfig,
    GoalCompassConfig,
    RetrievalFeedbackConfig,
    RerankerConfig,
    ConsolidationConfig,
    ServerConfig,
)


class TestMuninnConfigDefaults:
    def test_from_env_defaults(self):
        config = MuninnConfig.from_env()
        assert config.embedding.model == "nomic-embed-text"
        assert config.embedding.dimensions == 768
        assert config.vector.collection == "muninn_memories"
        assert config.server.port == 42069
        assert config.server.host == "127.0.0.1"

    def test_ensure_directories(self, tmp_path):
        config = MuninnConfig(
            data_dir=str(tmp_path / "muninn_data"),
            embedding=EmbeddingConfig(),
            vector=VectorConfig(path=str(tmp_path / "muninn_data" / "qdrant_v8")),
            graph=GraphConfig(path=str(tmp_path / "muninn_data" / "kuzu_v12")),
            metadata=MetadataConfig(path=str(tmp_path / "muninn_data" / "metadata.db")),
            extraction=ExtractionConfig(),
            reranker=RerankerConfig(),
            consolidation=ConsolidationConfig(),
            server=ServerConfig(),
        )
        config.ensure_directories()
        # ensure_directories creates data_dir and parents of store paths
        assert (tmp_path / "muninn_data").exists()


class TestEmbeddingConfig:
    def test_defaults(self):
        cfg = EmbeddingConfig()
        assert cfg.model == "nomic-embed-text"
        assert cfg.dimensions == 768
        assert cfg.provider == "ollama"
        assert cfg.ollama_url == "http://localhost:11434"

    def test_custom(self):
        cfg = EmbeddingConfig(model="all-MiniLM-L6-v2", dimensions=384, provider="fastembed")
        assert cfg.model == "all-MiniLM-L6-v2"
        assert cfg.dimensions == 384
        assert cfg.provider == "fastembed"


class TestVectorConfig:
    def test_defaults(self):
        cfg = VectorConfig()
        assert cfg.collection == "muninn_memories"
        assert cfg.on_disk is True
        assert cfg.dimensions == 768

    def test_custom(self):
        cfg = VectorConfig(collection="test_col", on_disk=False, dimensions=384)
        assert cfg.collection == "test_col"
        assert cfg.on_disk is False
        assert cfg.dimensions == 384


class TestServerConfig:
    def test_defaults(self):
        cfg = ServerConfig()
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 42069
        assert cfg.log_level == "info"

    def test_custom(self):
        cfg = ServerConfig(host="0.0.0.0", port=8000, log_level="debug")
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8000
        assert cfg.log_level == "debug"


class TestConsolidationConfig:
    def test_defaults(self):
        cfg = ConsolidationConfig()
        assert cfg.enabled is True
        assert cfg.interval_hours == 6.0
        assert cfg.decay_threshold == 0.1
        assert cfg.merge_similarity == 0.92
        assert cfg.promote_access_count == 5
        assert cfg.working_memory_ttl_hours == 24.0

    def test_custom(self):
        cfg = ConsolidationConfig(
            enabled=False,
            interval_hours=12.0,
            decay_threshold=0.2,
            merge_similarity=0.85,
        )
        assert cfg.enabled is False
        assert cfg.interval_hours == 12.0
        assert cfg.merge_similarity == 0.85


class TestExtractionConfig:
    def test_defaults(self):
        cfg = ExtractionConfig()
        assert cfg.enable_xlam is True
        assert cfg.xlam_model == "xLAM"
        assert cfg.enable_ollama_fallback is True

    def test_custom(self):
        cfg = ExtractionConfig(enable_xlam=False, ollama_model="phi3")
        assert cfg.enable_xlam is False
        assert cfg.ollama_model == "phi3"


class TestRerankerConfig:
    def test_defaults(self):
        cfg = RerankerConfig()
        assert cfg.enabled is True
        assert cfg.model == "jinaai/jina-reranker-v1-tiny-en"


class TestGoalCompassConfig:
    def test_defaults(self):
        cfg = GoalCompassConfig()
        assert cfg.drift_threshold == 0.55
        assert cfg.signal_weight == 0.65
        assert cfg.reminder_max_chars == 240


class TestRetrievalFeedbackConfig:
    def test_defaults(self):
        cfg = RetrievalFeedbackConfig()
        assert cfg.enabled is False
        assert cfg.lookback_days == 30
        assert cfg.min_total_signal_weight == 3.0
        assert cfg.estimator == "weighted_mean"
        assert cfg.propensity_floor == 0.05
        assert cfg.min_effective_samples == 2.0
        assert cfg.default_sampling_prob == 1.0
        assert cfg.cache_ttl_seconds == 30
        assert cfg.multiplier_floor == 0.75
        assert cfg.multiplier_ceiling == 1.25


class TestConfigFromEnv:
    def test_env_override_port(self, monkeypatch):
        monkeypatch.setenv("MUNINN_PORT", "9999")
        config = MuninnConfig.from_env()
        assert config.server.port == 9999

    def test_env_override_data_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MUNINN_DATA_DIR", str(tmp_path))
        config = MuninnConfig.from_env()
        assert str(tmp_path) in config.vector.path

    def test_env_override_embedding_model(self, monkeypatch):
        monkeypatch.setenv("MUNINN_EMBEDDING_MODEL", "custom-model")
        config = MuninnConfig.from_env()
        assert config.embedding.model == "custom-model"

    def test_env_override_host(self, monkeypatch):
        monkeypatch.setenv("MUNINN_HOST", "0.0.0.0")
        config = MuninnConfig.from_env()
        assert config.server.host == "0.0.0.0"

    def test_env_disable_reranker(self, monkeypatch):
        monkeypatch.setenv("MUNINN_RERANKER_ENABLED", "false")
        config = MuninnConfig.from_env()
        assert config.reranker.enabled is False

    def test_env_disable_consolidation(self, monkeypatch):
        monkeypatch.setenv("MUNINN_CONSOLIDATION_ENABLED", "false")
        config = MuninnConfig.from_env()
        assert config.consolidation.enabled is False

    def test_env_override_goal_compass(self, monkeypatch):
        monkeypatch.setenv("MUNINN_GOAL_DRIFT_THRESHOLD", "0.61")
        monkeypatch.setenv("MUNINN_GOAL_SIGNAL_WEIGHT", "0.72")
        monkeypatch.setenv("MUNINN_GOAL_REMINDER_MAX_CHARS", "180")
        config = MuninnConfig.from_env()
        assert config.goal_compass.drift_threshold == 0.61
        assert config.goal_compass.signal_weight == 0.72
        assert config.goal_compass.reminder_max_chars == 180

    def test_env_override_retrieval_feedback(self, monkeypatch):
        monkeypatch.setenv("MUNINN_RETRIEVAL_FEEDBACK_ENABLED", "true")
        monkeypatch.setenv("MUNINN_RETRIEVAL_FEEDBACK_LOOKBACK_DAYS", "14")
        monkeypatch.setenv("MUNINN_RETRIEVAL_FEEDBACK_MIN_TOTAL_WEIGHT", "5.5")
        monkeypatch.setenv("MUNINN_RETRIEVAL_FEEDBACK_ESTIMATOR", "snips")
        monkeypatch.setenv("MUNINN_RETRIEVAL_FEEDBACK_PROPENSITY_FLOOR", "0.08")
        monkeypatch.setenv("MUNINN_RETRIEVAL_FEEDBACK_MIN_EFFECTIVE_SAMPLES", "4.0")
        monkeypatch.setenv("MUNINN_RETRIEVAL_FEEDBACK_DEFAULT_SAMPLING_PROB", "0.9")
        monkeypatch.setenv("MUNINN_RETRIEVAL_FEEDBACK_CACHE_TTL", "45")
        monkeypatch.setenv("MUNINN_RETRIEVAL_FEEDBACK_FLOOR", "0.8")
        monkeypatch.setenv("MUNINN_RETRIEVAL_FEEDBACK_CEILING", "1.2")
        config = MuninnConfig.from_env()
        assert config.retrieval_feedback.enabled is True
        assert config.retrieval_feedback.lookback_days == 14
        assert config.retrieval_feedback.min_total_signal_weight == 5.5
        assert config.retrieval_feedback.estimator == "snips"
        assert config.retrieval_feedback.propensity_floor == 0.08
        assert config.retrieval_feedback.min_effective_samples == 4.0
        assert config.retrieval_feedback.default_sampling_prob == 0.9
        assert config.retrieval_feedback.cache_ttl_seconds == 45
        assert config.retrieval_feedback.multiplier_floor == 0.8
        assert config.retrieval_feedback.multiplier_ceiling == 1.2
