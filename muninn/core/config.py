"""
Muninn Configuration
--------------------
Centralized configuration management for all Muninn components.
Loads from environment variables and YAML config files.

v3.1.0: Uses platform abstraction for cross-platform path resolution.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from muninn.platform import get_data_dir

logger = logging.getLogger("Muninn.Config")

# Default data directory — now cross-platform via platform.py
DEFAULT_DATA_DIR = str(get_data_dir())


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    provider: str = "ollama"
    model: str = "nomic-embed-text"
    dimensions: int = 768
    ollama_url: str = "http://localhost:11434"


class VectorConfig(BaseModel):
    """Qdrant vector store configuration."""
    path: str = os.path.join(DEFAULT_DATA_DIR, "qdrant_v8")
    collection: str = "muninn_memories"
    on_disk: bool = True
    dimensions: int = 768


class GraphConfig(BaseModel):
    """Kuzu graph store configuration."""
    path: str = os.path.join(DEFAULT_DATA_DIR, "kuzu_v12")


class MetadataConfig(BaseModel):
    """SQLite metadata store configuration."""
    path: str = os.path.join(DEFAULT_DATA_DIR, "metadata.db")


class ExtractionConfig(BaseModel):
    """Extraction pipeline configuration."""
    enable_xlam: bool = True
    xlam_url: str = "http://localhost:8001/v1"
    xlam_model: str = "xLAM"
    enable_ollama_fallback: bool = True
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    # Instructor-based extraction (v3.1.0)
    enable_instructor: bool = True
    instructor_provider: str = "ollama"  # "ollama" | "xlam" | "openai" | "custom"
    instructor_base_url: str = "http://localhost:11434/v1"
    instructor_model: str = "llama3.2:3b"
    instructor_api_key: str = "not-needed"


class ConflictDetectionConfig(BaseModel):
    """NLI-based conflict detection configuration (v3.2.0)."""
    model_name: str = "cross-encoder/nli-deberta-v3-small"
    contradiction_threshold: float = 0.7
    similarity_prefilter: float = 0.6


class SemanticDedupConfig(BaseModel):
    """Semantic deduplication configuration (v3.2.0)."""
    threshold: float = 0.95
    content_overlap_threshold: float = 0.8


class GoalCompassConfig(BaseModel):
    """Goal compass configuration (v3.1.2)."""
    drift_threshold: float = 0.55
    signal_weight: float = 0.65
    reminder_max_chars: int = 240


class RetrievalFeedbackConfig(BaseModel):
    """Retrieval feedback calibration configuration (v3.2.0)."""
    enabled: bool = False
    lookback_days: int = 30
    min_total_signal_weight: float = 3.0
    estimator: str = "weighted_mean"
    propensity_floor: float = 0.05
    min_effective_samples: float = 2.0
    default_sampling_prob: float = 1.0
    cache_ttl_seconds: int = 30
    multiplier_floor: float = 0.75
    multiplier_ceiling: float = 1.25


class RerankerConfig(BaseModel):
    """Reranker configuration."""
    enabled: bool = True
    model: str = "jinaai/jina-reranker-v1-tiny-en"


class ConsolidationConfig(BaseModel):
    """Consolidation daemon configuration."""
    enabled: bool = True
    interval_hours: float = 6.0
    decay_threshold: float = 0.1
    merge_similarity: float = 0.92
    promote_access_count: int = 5
    working_memory_ttl_hours: float = 24.0


class ServerConfig(BaseModel):
    """FastAPI server configuration."""
    host: str = "127.0.0.1"
    port: int = 42069
    log_level: str = "info"


class MuninnConfig(BaseModel):
    """Root configuration for the entire Muninn system."""
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector: VectorConfig = Field(default_factory=VectorConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    conflict_detection: ConflictDetectionConfig = Field(default_factory=ConflictDetectionConfig)
    semantic_dedup: SemanticDedupConfig = Field(default_factory=SemanticDedupConfig)
    goal_compass: GoalCompassConfig = Field(default_factory=GoalCompassConfig)
    retrieval_feedback: RetrievalFeedbackConfig = Field(default_factory=RetrievalFeedbackConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    data_dir: str = DEFAULT_DATA_DIR

    @classmethod
    def from_env(cls) -> "MuninnConfig":
        """
        Load configuration from environment variables.

        Environment variables override defaults:
        - MUNINN_DATA_DIR: Base data directory
        - MUNINN_HOST / MUNINN_PORT: Server binding
        - MUNINN_EMBEDDING_MODEL: Embedding model name
        - MUNINN_EMBEDDING_DIMS: Embedding dimensions
        - MUNINN_OLLAMA_URL: Ollama server URL
        - MUNINN_XLAM_URL: xLAM server URL
        - MUNINN_XLAM_ENABLED: Enable/disable xLAM extraction
        - MUNINN_RERANKER_ENABLED: Enable/disable reranker
        - MUNINN_CONSOLIDATION_ENABLED: Enable/disable consolidation
        - MUNINN_CONSOLIDATION_INTERVAL: Hours between consolidation cycles
        """
        data_dir = os.environ.get("MUNINN_DATA_DIR", DEFAULT_DATA_DIR)
        ollama_url = os.environ.get("MUNINN_OLLAMA_URL", "http://localhost:11434")
        embedding_model = os.environ.get("MUNINN_EMBEDDING_MODEL", "nomic-embed-text")
        embedding_dims = int(os.environ.get("MUNINN_EMBEDDING_DIMS", "768"))

        config = cls(
            data_dir=data_dir,
            embedding=EmbeddingConfig(
                model=embedding_model,
                dimensions=embedding_dims,
                ollama_url=ollama_url,
            ),
            vector=VectorConfig(
                path=os.path.join(data_dir, "qdrant_v8"),
                dimensions=embedding_dims,
            ),
            graph=GraphConfig(
                path=os.path.join(data_dir, "kuzu_v12"),
            ),
            metadata=MetadataConfig(
                path=os.path.join(data_dir, "metadata.db"),
            ),
            extraction=ExtractionConfig(
                enable_xlam=os.environ.get("MUNINN_XLAM_ENABLED", "true").lower() == "true",
                xlam_url=os.environ.get("MUNINN_XLAM_URL", "http://localhost:8001/v1"),
                enable_ollama_fallback=True,
                ollama_url=ollama_url,
                # Instructor extraction (v3.1.0)
                enable_instructor=os.environ.get("MUNINN_INSTRUCTOR_ENABLED", "true").lower() == "true",
                instructor_provider=os.environ.get("MUNINN_INSTRUCTOR_PROVIDER", "ollama"),
                instructor_base_url=os.environ.get("MUNINN_INSTRUCTOR_URL", f"{ollama_url}/v1"),
                instructor_model=os.environ.get("MUNINN_INSTRUCTOR_MODEL", "llama3.2:3b"),
                instructor_api_key=os.environ.get("MUNINN_INSTRUCTOR_API_KEY", "not-needed"),
            ),
            reranker=RerankerConfig(
                enabled=os.environ.get("MUNINN_RERANKER_ENABLED", "true").lower() == "true",
            ),
            consolidation=ConsolidationConfig(
                enabled=os.environ.get("MUNINN_CONSOLIDATION_ENABLED", "true").lower() == "true",
                interval_hours=float(os.environ.get("MUNINN_CONSOLIDATION_INTERVAL", "6.0")),
            ),
            conflict_detection=ConflictDetectionConfig(
                model_name=os.environ.get(
                    "MUNINN_CONFLICT_MODEL", "cross-encoder/nli-deberta-v3-small"
                ),
                contradiction_threshold=float(
                    os.environ.get("MUNINN_CONFLICT_THRESHOLD", "0.7")
                ),
                similarity_prefilter=float(
                    os.environ.get("MUNINN_CONFLICT_PREFILTER", "0.6")
                ),
            ),
            semantic_dedup=SemanticDedupConfig(
                threshold=float(os.environ.get("MUNINN_DEDUP_THRESHOLD", "0.95")),
                content_overlap_threshold=float(
                    os.environ.get("MUNINN_DEDUP_OVERLAP", "0.8")
                ),
            ),
            goal_compass=GoalCompassConfig(
                drift_threshold=float(
                    os.environ.get("MUNINN_GOAL_DRIFT_THRESHOLD", "0.55")
                ),
                signal_weight=float(
                    os.environ.get("MUNINN_GOAL_SIGNAL_WEIGHT", "0.65")
                ),
                reminder_max_chars=int(
                    os.environ.get("MUNINN_GOAL_REMINDER_MAX_CHARS", "240")
                ),
            ),
            retrieval_feedback=RetrievalFeedbackConfig(
                enabled=os.environ.get("MUNINN_RETRIEVAL_FEEDBACK_ENABLED", "false").lower() == "true",
                lookback_days=int(os.environ.get("MUNINN_RETRIEVAL_FEEDBACK_LOOKBACK_DAYS", "30")),
                min_total_signal_weight=float(
                    os.environ.get("MUNINN_RETRIEVAL_FEEDBACK_MIN_TOTAL_WEIGHT", "3.0")
                ),
                estimator=os.environ.get("MUNINN_RETRIEVAL_FEEDBACK_ESTIMATOR", "weighted_mean"),
                propensity_floor=float(
                    os.environ.get("MUNINN_RETRIEVAL_FEEDBACK_PROPENSITY_FLOOR", "0.05")
                ),
                min_effective_samples=float(
                    os.environ.get("MUNINN_RETRIEVAL_FEEDBACK_MIN_EFFECTIVE_SAMPLES", "2.0")
                ),
                default_sampling_prob=float(
                    os.environ.get("MUNINN_RETRIEVAL_FEEDBACK_DEFAULT_SAMPLING_PROB", "1.0")
                ),
                cache_ttl_seconds=int(
                    os.environ.get("MUNINN_RETRIEVAL_FEEDBACK_CACHE_TTL", "30")
                ),
                multiplier_floor=float(
                    os.environ.get("MUNINN_RETRIEVAL_FEEDBACK_FLOOR", "0.75")
                ),
                multiplier_ceiling=float(
                    os.environ.get("MUNINN_RETRIEVAL_FEEDBACK_CEILING", "1.25")
                ),
            ),
            server=ServerConfig(
                host=os.environ.get("MUNINN_HOST", "127.0.0.1"),
                port=int(os.environ.get("MUNINN_PORT", "42069")),
                log_level=os.environ.get("MUNINN_LOG_LEVEL", "info"),
            ),
        )

        return config

    @classmethod
    def from_yaml(cls, path: str) -> "MuninnConfig":
        """Load configuration from a YAML file."""
        try:
            import yaml
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            return cls(**data)
        except ImportError:
            logger.warning("PyYAML not installed — cannot load YAML config")
            return cls.from_env()
        except FileNotFoundError:
            logger.warning("Config file not found: %s — using defaults", path)
            return cls.from_env()

    def ensure_directories(self) -> None:
        """Create data directories if they don't exist."""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.vector.path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.graph.path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.metadata.path).parent.mkdir(parents=True, exist_ok=True)
        logger.info("Data directory: %s", self.data_dir)
