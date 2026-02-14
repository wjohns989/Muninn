"""
Heuristic memory-chain detector (Phase 3A).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set

from muninn.core.types import MemoryRecord

DEFAULT_CAUSAL_MARKERS = (
    "because",
    "therefore",
    "so that",
    "led to",
    "resulted in",
    "caused",
    "blocked by",
    "fixed by",
)


@dataclass(frozen=True)
class MemoryChainLink:
    predecessor_id: str
    successor_id: str
    relation_type: str
    confidence: float
    reason: str
    shared_entities: List[str]
    hours_apart: float


class MemoryChainDetector:
    """
    Detect PRECEDES / CAUSES links between a new memory and recent scoped memories.

    The scoring model is intentionally simple, deterministic, and bounded:
    - temporal proximity (closer in time is stronger),
    - entity overlap,
    - causal markers in the successor text,
    - same project/namespace bonus.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.6,
        max_hours_apart: float = 168.0,
        max_links_per_memory: int = 4,
        causal_markers: Sequence[str] = DEFAULT_CAUSAL_MARKERS,
    ):
        self.threshold = max(0.0, min(1.0, float(threshold)))
        self.max_hours_apart = max(1.0, float(max_hours_apart))
        self.max_links_per_memory = max(1, int(max_links_per_memory))
        self.causal_markers = tuple(marker.lower() for marker in causal_markers if marker)

    @staticmethod
    def _normalize_entity_names(entity_names: Iterable[str]) -> Set[str]:
        normalized: Set[str] = set()
        for name in entity_names:
            if not isinstance(name, str):
                continue
            trimmed = name.strip().lower()
            if trimmed:
                normalized.add(trimmed)
        return normalized

    @staticmethod
    def _extract_candidate_entities(candidate: MemoryRecord) -> Set[str]:
        metadata = candidate.metadata or {}
        raw = metadata.get("entity_names", [])
        if isinstance(raw, list):
            return MemoryChainDetector._normalize_entity_names(str(item) for item in raw)
        return set()

    def _causal_strength(self, content: str) -> float:
        lower = content.lower()
        marker_hits = sum(1 for marker in self.causal_markers if marker in lower)
        if marker_hits <= 0:
            return 0.0
        return min(1.0, 0.5 + (0.15 * marker_hits))

    def detect_links(
        self,
        *,
        successor_record: MemoryRecord,
        successor_content: str,
        successor_entity_names: Iterable[str],
        candidate_records: Sequence[MemoryRecord],
    ) -> List[MemoryChainLink]:
        """
        Return best chain links from candidates into the successor record.
        """
        successor_entities = self._normalize_entity_names(successor_entity_names)
        if not successor_entities:
            return []

        candidate_links: List[MemoryChainLink] = []
        successor_ts = float(successor_record.created_at)
        successor_scope = (
            str(successor_record.project),
            str(successor_record.namespace),
        )
        causal_strength = self._causal_strength(successor_content)

        for candidate in candidate_records:
            if candidate.id == successor_record.id:
                continue
            if candidate.created_at > successor_ts:
                continue

            candidate_scope = (str(candidate.project), str(candidate.namespace))
            candidate_entities = self._extract_candidate_entities(candidate)
            if not candidate_entities:
                continue

            shared = sorted(successor_entities.intersection(candidate_entities))
            if not shared:
                continue

            hours_apart = max(0.0, (successor_ts - float(candidate.created_at)) / 3600.0)
            if hours_apart > self.max_hours_apart:
                continue

            temporal_score = max(0.0, 1.0 - (hours_apart / self.max_hours_apart))
            overlap_score = min(1.0, len(shared) / 3.0)
            scope_score = 1.0 if candidate_scope == successor_scope else 0.0

            confidence = (
                0.45 * temporal_score
                + 0.35 * overlap_score
                + 0.15 * causal_strength
                + 0.05 * scope_score
            )
            if confidence < self.threshold:
                continue

            relation_type = "CAUSES" if causal_strength >= 0.5 else "PRECEDES"
            reason = (
                f"temporal={temporal_score:.3f};overlap={overlap_score:.3f};"
                f"causal={causal_strength:.3f};scope={scope_score:.3f}"
            )
            candidate_links.append(
                MemoryChainLink(
                    predecessor_id=candidate.id,
                    successor_id=successor_record.id,
                    relation_type=relation_type,
                    confidence=min(1.0, max(0.0, confidence)),
                    reason=reason,
                    shared_entities=shared,
                    hours_apart=hours_apart,
                )
            )

        candidate_links.sort(key=lambda link: (-link.confidence, link.hours_apart, link.predecessor_id))
        return candidate_links[: self.max_links_per_memory]

