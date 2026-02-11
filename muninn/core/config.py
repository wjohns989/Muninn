"""
Muninn Configuration
--------------------
Centralized configuration management for all Muninn components.
Loads from environment variables and YAML config files.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

logger = logging.getLogger("Muninn.Config")

# Default data directory
DEFAULT_DATA_DIR = os.path.join(os.path.expanduser("~"), ".muninn", "data")


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
            ),
            reranker=RerankerConfig(
                enabled=os.environ.get("MUNINN_RERANKER_ENABLED", "true").lower() == "true",
            ),
            consolidation=ConsolidationConfig(
                enabled=os.environ.get("MUNINN_CONSOLIDATION_ENABLED", "true").lower() == "true",
                interval_hours=float(os.environ.get("MUNINN_CONSOLIDATION_INTERVAL", "6.0")),
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
