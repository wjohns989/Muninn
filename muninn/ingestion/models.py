"""
Data models for multi-source ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class IngestionChunk:
    source_path: str
    source_type: str
    content: str
    chunk_index: int
    chunk_count: int
    source_sha256: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionSourceResult:
    source_path: str
    source_type: str
    status: str
    chunks: List[IngestionChunk] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    skipped_reason: str = ""


@dataclass
class IngestionReport:
    total_sources: int
    processed_sources: int
    skipped_sources: int
    total_chunks: int
    source_results: List[IngestionSourceResult] = field(default_factory=list)
