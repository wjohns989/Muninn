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
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from muninn.core.feature_flags import FeatureFlags
from muninn.platform import get_data_dir

logger = logging.getLogger("Muninn.Config")

# Default data directory — now cross-platform via platform.py
DEFAULT_DATA_DIR = str(get_data_dir())
DEFAULT_LOW_LATENCY_MODEL = "llama3.1:latest"
DEFAULT_BALANCED_MODEL = "qwen2.5:7b"
DEFAULT_HIGH_REASONING_MODEL = "qwen2.5-coder:14b"
SUPPORTED_MODEL_PROFILES = ("low_latency", "balanced", "high_reasoning")


def _parse_optional_float_env(name: str) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = float(raw)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        logger.warning(
            "Invalid %s value '%s'; expected positive float. Ignoring.",
            name,
            raw,
        )
        return None


def _select_profile_models_for_vram(vram_budget_gb: Optional[float]) -> Dict[str, str]:
    """
    Select profile model defaults for an approximate VRAM budget.

    This policy keeps low-latency memory operations responsive under constrained
    developer GPUs while preserving a higher-capability path when headroom exists.
    """
    if vram_budget_gb is None:
        return {
            "low_latency": DEFAULT_LOW_LATENCY_MODEL,
            "balanced": DEFAULT_BALANCED_MODEL,
            "high_reasoning": DEFAULT_HIGH_REASONING_MODEL,
        }

    if vram_budget_gb < 6:
        return {
            "low_latency": "llama3.2:1b",
            "balanced": "qwen3:1.7b",
            "high_reasoning": "qwen3:4b",
        }
    if vram_budget_gb < 10:
        return {
            "low_latency": DEFAULT_LOW_LATENCY_MODEL,
            "balanced": "qwen3:4b",
            "high_reasoning": "qwen3:8b",
        }
    if vram_budget_gb < 18:
        return {
            "low_latency": DEFAULT_LOW_LATENCY_MODEL,
            "balanced": DEFAULT_BALANCED_MODEL,
            "high_reasoning": DEFAULT_HIGH_REASONING_MODEL,
        }
    if vram_budget_gb < 28:
        return {
            "low_latency": DEFAULT_LOW_LATENCY_MODEL,
            "balanced": DEFAULT_BALANCED_MODEL,
            "high_reasoning": "qwen3:30b",
        }
    return {
        "low_latency": DEFAULT_LOW_LATENCY_MODEL,
        "balanced": DEFAULT_BALANCED_MODEL,
        "high_reasoning": "qwen3:32b",
    }


def _normalize_model_profile(profile: Optional[str], default: str) -> str:
    candidate = (profile or "").strip()
    if candidate in SUPPORTED_MODEL_PROFILES:
        return candidate
    if candidate:
        logger.warning(
            "Unsupported model profile '%s'; expected one of %s. Falling back to '%s'.",
            candidate,
            SUPPORTED_MODEL_PROFILES,
            default,
        )
    return default


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
    ollama_model: str = "llama3.2:3b"  # low-latency baseline
    model_profile: str = "balanced"  # low_latency | balanced | high_reasoning
    runtime_model_profile: str = "low_latency"
    ingestion_model_profile: str = "balanced"
    legacy_ingestion_model_profile: str = "balanced"
    ollama_balanced_model: str = DEFAULT_BALANCED_MODEL
    ollama_high_reasoning_model: str = DEFAULT_HIGH_REASONING_MODEL
    vram_budget_gb: Optional[float] = None
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


class IngestionConfig(BaseModel):
    """Multi-source ingestion configuration (v3.3.0)."""
    max_file_size_bytes: int = 5 * 1024 * 1024
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 150
    min_chunk_chars: int = 120
    allowed_roots: List[str] = Field(default_factory=list)


class LegacyDiscoveryConfig(BaseModel):
    """Legacy source discovery configuration (v3.18.1)."""
    enabled: bool = True
    interval_hours: float = 1.0


class MemoryChainsConfig(BaseModel):
    """Memory chain detection/retrieval configuration (v3.3.0)."""
    detection_threshold: float = 0.6
    max_hours_apart: float = 168.0
    max_links_per_memory: int = 4
    candidate_scan_limit: int = 80
    retrieval_signal_weight: float = 0.6
    retrieval_expansion_limit: int = 20
    retrieval_seed_limit: int = 6


class AdvancedConfig(BaseModel):
    """Advanced differentiating features (Phase 6+)."""
    enable_colbert: bool = False
    colbert_dim: int = 128
    enable_temporal_kg: bool = False
    # Phase 13 (v3.10.0): Native ColBERT multi-vector via Qdrant MultiVectorConfig
    enable_colbert_multivec: bool = False
    colbert_multivec_collection: str = "muninn_colbert_multivec"


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
    
    # Phase 9: Maintenance & Integrity (v3.6.0)
    colbert_drift_threshold: float = 0.15
    quantization_threshold_points: int = 10000
    integrity_contradiction_threshold: float = 0.7


class ServerConfig(BaseModel):
    """FastAPI server configuration."""
    host: str = "127.0.0.1"
    port: int = 42069
    log_level: str = "info"
    auth_token: Optional[str] = None


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
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    legacy_discovery: LegacyDiscoveryConfig = Field(default_factory=LegacyDiscoveryConfig)
    memory_chains: MemoryChainsConfig = Field(default_factory=MemoryChainsConfig)
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)
    feature_flags: FeatureFlags = Field(default_factory=FeatureFlags)
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
        vram_budget_gb = _parse_optional_float_env("MUNINN_VRAM_BUDGET_GB")
        profile_models = _select_profile_models_for_vram(vram_budget_gb)
        default_model_profile = _normalize_model_profile(
            os.environ.get("MUNINN_MODEL_PROFILE"),
            "balanced",
        )
        runtime_model_profile = _normalize_model_profile(
            os.environ.get("MUNINN_RUNTIME_MODEL_PROFILE"),
            "low_latency",
        )
        ingestion_model_profile = _normalize_model_profile(
            os.environ.get("MUNINN_INGESTION_MODEL_PROFILE"),
            "balanced",
        )
        legacy_ingestion_model_profile = _normalize_model_profile(
            os.environ.get("MUNINN_LEGACY_INGESTION_MODEL_PROFILE"),
            ingestion_model_profile,
        )
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
                xlam_model=os.environ.get("MUNINN_XLAM_MODEL", "xLAM"),
                enable_ollama_fallback=True,
                ollama_url=ollama_url,
                ollama_model=os.environ.get("MUNINN_OLLAMA_MODEL", profile_models["low_latency"]),
                model_profile=default_model_profile,
                runtime_model_profile=runtime_model_profile,
                ingestion_model_profile=ingestion_model_profile,
                legacy_ingestion_model_profile=legacy_ingestion_model_profile,
                ollama_balanced_model=os.environ.get(
                    "MUNINN_OLLAMA_BALANCED_MODEL", profile_models["balanced"]
                ),
                ollama_high_reasoning_model=os.environ.get(
                    "MUNINN_OLLAMA_HIGH_REASONING_MODEL", profile_models["high_reasoning"]
                ),
                vram_budget_gb=vram_budget_gb,
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
            ingestion=IngestionConfig(
                max_file_size_bytes=int(
                    os.environ.get("MUNINN_INGESTION_MAX_FILE_BYTES", str(5 * 1024 * 1024))
                ),
                chunk_size_chars=int(
                    os.environ.get("MUNINN_INGESTION_CHUNK_SIZE_CHARS", "1200")
                ),
                chunk_overlap_chars=int(
                    os.environ.get("MUNINN_INGESTION_CHUNK_OVERLAP_CHARS", "150")
                ),
                min_chunk_chars=int(
                    os.environ.get("MUNINN_INGESTION_MIN_CHUNK_CHARS", "120")
                ),
                allowed_roots=[
                    part.strip()
                    for part in os.environ.get("MUNINN_INGESTION_ALLOWED_ROOTS", "").split(os.pathsep)
                    if part.strip()
                ],
            ),
            legacy_discovery=LegacyDiscoveryConfig(
                enabled=os.environ.get("MUNINN_LEGACY_DISCOVERY_ENABLED", "true").lower() == "true",
                interval_hours=float(os.environ.get("MUNINN_LEGACY_DISCOVERY_INTERVAL", "1.0")),
            ),
            memory_chains=MemoryChainsConfig(
                detection_threshold=float(
                    os.environ.get("MUNINN_CHAINS_DETECTION_THRESHOLD", "0.6")
                ),
                max_hours_apart=float(
                    os.environ.get("MUNINN_CHAINS_MAX_HOURS_APART", "168.0")
                ),
                max_links_per_memory=int(
                    os.environ.get("MUNINN_CHAINS_MAX_LINKS_PER_MEMORY", "4")
                ),
                candidate_scan_limit=int(
                    os.environ.get("MUNINN_CHAINS_CANDIDATE_SCAN_LIMIT", "80")
                ),
                retrieval_signal_weight=float(
                    os.environ.get("MUNINN_CHAINS_SIGNAL_WEIGHT", "0.6")
                ),
                retrieval_expansion_limit=int(
                    os.environ.get("MUNINN_CHAINS_EXPANSION_LIMIT", "20")
                ),
                retrieval_seed_limit=int(
                    os.environ.get("MUNINN_CHAINS_SEED_LIMIT", "6")
                ),
            ),
            advanced=AdvancedConfig(
                enable_colbert=os.environ.get("MUNINN_COLBERT_ENABLED", "false").lower() == "true",
                colbert_dim=int(os.environ.get("MUNINN_COLBERT_DIM", "128")),
                enable_temporal_kg=os.environ.get("MUNINN_TEMPORAL_KG_ENABLED", "false").lower() == "true",
                enable_colbert_multivec=os.environ.get("MUNINN_COLBERT_MULTIVEC", "0").lower() in ("1", "true", "yes", "on"),
                colbert_multivec_collection=os.environ.get(
                    "MUNINN_COLBERT_MULTIVEC_COLLECTION", "muninn_colbert_multivec"
                ),
            ),
            feature_flags=FeatureFlags.from_env(),
            server=ServerConfig(
                host=os.environ.get("MUNINN_HOST", "127.0.0.1"),
                port=int(os.environ.get("MUNINN_PORT", "42069")),
                log_level=os.environ.get("MUNINN_LOG_LEVEL", "info"),
                auth_token=os.environ.get("MUNINN_SERVER_AUTH_TOKEN"),
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