"""
Multi-source ingestion package.
"""

from muninn.ingestion.discovery import DiscoveredLegacySource, discover_legacy_sources
from muninn.ingestion.models import IngestionChunk, IngestionReport, IngestionSourceResult
from muninn.ingestion.pipeline import IngestionPipeline
from muninn.ingestion.periodic import PeriodicIngestionScheduler, PeriodicIngestionSettings

__all__ = [
    "IngestionChunk",
    "IngestionSourceResult",
    "IngestionReport",
    "IngestionPipeline",
    "DiscoveredLegacySource",
    "discover_legacy_sources",
    "PeriodicIngestionSettings",
    "PeriodicIngestionScheduler",
]
