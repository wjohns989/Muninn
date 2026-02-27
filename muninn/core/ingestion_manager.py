"""
Muninn Ingestion Manager
-------------------------
Handles the logic for adding and updating memories, including extraction,
embedding, deduplication, and conflict detection.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple

from muninn.core.types import (
    MemoryRecord, MemoryType, Provenance, ExtractionResult,
)
from muninn.scoring.importance import calculate_importance, calculate_novelty

logger = logging.getLogger("Muninn.Ingestion")

class IngestionManager:
    """
    Orchestrates the ingestion pipeline for a single memory record.
    """
    def __init__(self, memory):
        self.memory = memory
        self.config = memory.config
        self._otel = memory._otel

    async def process_add(
        self,
        content: str,
        user_id: str,
        agent_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
        namespace: str,
        memory_type: MemoryType,
        provenance: Provenance,
        scope: str = "project",
        media_type: str = "text",
    ) -> Dict[str, Any]:
        """
        Execute the full ingestion pipeline: extract -> embed -> dedup -> conflict -> score -> store.
        """
        self._otel.add_event(
            "muninn.add.request",
            {"content_preview": self._otel.maybe_content(content)},
        )

        scoped_metadata = {**(metadata or {}), "user_id": user_id}
        project_value = scoped_metadata.get("project")
        project = str(project_value) if project_value else "global"
        branch = scoped_metadata.get("branch")
        if branch is not None:
            branch = str(branch)

        scope_filters = {"namespace": namespace, "user_id": user_id}
        if project_value:
            scope_filters["project"] = project

        extraction_profile = str(
            scoped_metadata.get("operator_model_profile")
            or self.config.extraction.runtime_model_profile
            or self.config.extraction.model_profile
        )
        skip_extraction = bool(scoped_metadata.get("muninn_skip_extraction", False))
        extraction_timeout_value = scoped_metadata.get("muninn_extraction_timeout_seconds")
        extraction_timeout_seconds: Optional[float] = None
        if extraction_timeout_value is not None:
            try:
                parsed_timeout = float(extraction_timeout_value)
            except (TypeError, ValueError):
                parsed_timeout = 0.0
            if parsed_timeout > 0:
                extraction_timeout_seconds = parsed_timeout
        
        # 1. Extraction
        if skip_extraction:
            extraction = ExtractionResult()
            entity_names = []
        else:
            with self._otel.span("muninn.ingestion.extract", {"model_profile": extraction_profile}):
                if extraction_timeout_seconds is not None:
                    try:
                        extraction = await asyncio.wait_for(
                            self.memory._extract_with_profile(
                                content,
                                model_profile=extraction_profile,
                            ),
                            timeout=extraction_timeout_seconds,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Extraction timed out after %.3fs; continuing without extracted entities",
                            extraction_timeout_seconds,
                        )
                        extraction = ExtractionResult()
                        scoped_metadata["muninn_extraction_timed_out"] = True
                else:
                    extraction = await self.memory._extract_with_profile(
                        content,
                        model_profile=extraction_profile,
                    )
                entity_names = self.memory._extract_entity_names(extraction)
                if entity_names:
                    scoped_metadata["entity_names"] = entity_names
        
        # 2. Embedding
        with self._otel.span("muninn.ingestion.embed"):
            try:
                embedding = await asyncio.wait_for(
                    self.memory._embed(content),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning("Embedding timed out after 30s; using zero-vector fallback")
                embedding = [0.0] * self.config.vector.dimensions
                scoped_metadata["muninn_embedding_timed_out"] = True

        record = MemoryRecord(
            content=content,
            memory_type=memory_type,
            provenance=provenance,
            source_agent=agent_id or "unknown",
            project=project,
            branch=branch,
            namespace=namespace,
            metadata=scoped_metadata,
            novelty_score=0.0,
            scope=scope,
            media_type=media_type,
        )

        # 3. Semantic Deduplication
        vectors_count = await asyncio.to_thread(self.memory._vectors.count)
        if self.memory._dedup and vectors_count > 0:
            from muninn.dedup.semantic_dedup import DedupStrategy

            with self._otel.span("muninn.ingestion.dedup"):
                dedup_result = await asyncio.to_thread(
                    self.memory._dedup.check_duplicate,
                    embedding=embedding,
                    content=content,
                    vector_store=self.memory._vectors,
                    metadata_store=self.memory._metadata,
                    filters=scope_filters,
                )
            if dedup_result and dedup_result.is_duplicate:
                if dedup_result.strategy == DedupStrategy.SKIP:
                    logger.info("Dedup SKIP: duplicate of %s", dedup_result.existing_memory_id)
                    return {
                        "id": None,
                        "content": content,
                        "event": "DEDUP_SKIP",
                        "dedup": dedup_result.model_dump(),
                    }
                # Strategy: UPDATE_EXISTING will be handled after gathering all necessary ADD data
                # to support fallback in case of scope mismatch.

        # 4. Conflict Detection
        conflict_info = None
        if self.memory._conflict_detector and vectors_count > 0:
            with self._otel.span("muninn.ingestion.conflict_detection"):
                try:
                    similar_for_conflict = await asyncio.to_thread(
                        self.memory._vectors.search,
                        embedding,
                        limit=5,
                        score_threshold=self.config.conflict_detection.similarity_prefilter,
                        filters=scope_filters,
                    )
                    if similar_for_conflict:
                        candidate_ids = [mid for mid, _score in similar_for_conflict]
                        all_candidates = await asyncio.to_thread(self.memory._metadata.get_by_ids, candidate_ids)
                        candidate_records = [
                            candidate
                            for candidate in all_candidates
                            if self.memory._record_matches_scope(candidate, namespace, user_id)
                        ]
                        if candidate_records:
                            conflicts = await asyncio.to_thread(
                                self.memory._conflict_detector.detect_conflicts, content, candidate_records
                            )
                            if conflicts and self.memory._conflict_resolver:
                                conflicts.sort(key=lambda c: c.contradiction_score, reverse=True)
                                conflict = conflicts[0]
                                resolution = await self.memory._conflict_resolver.resolve(
                                    conflict,
                                    new_record=record,
                                    new_embedding=embedding,
                                    user_id=user_id,
                                )
                                conflict_info = {
                                    "conflict_detected": True,
                                    "resolution": resolution,
                                    "conflict_details": conflict.model_dump(),
                                }
                                if resolution.get("skip_new_storage"):
                                    return {
                                        "id": None,
                                        "content": content,
                                        "event": f"CONFLICT_{resolution['resolution'].upper()}",
                                        "conflict": conflict_info,
                                    }
                except Exception as e:
                    logger.warning("Conflict detection failed: %s", e)

        # 5. Scoring
        max_similarity = 0.0
        if vectors_count > 0:
            with self._otel.span("muninn.ingestion.novelty_search"):
                try:
                    similar = await asyncio.to_thread(
                        self.memory._vectors.search, embedding, limit=5, filters=scope_filters
                    )
                    if similar:
                        max_similarity = similar[0][1]
                except Exception:
                    pass

        record.novelty_score = calculate_novelty(max_similarity)
        centrality = 0.1 if extraction.entities else 0.0
        
        with self._otel.span("muninn.ingestion.calculate_importance"):
            record.importance = calculate_importance(
                record,
                max_similarity=max_similarity,
                centrality=centrality,
            )

        # 6. Final Decision
        # If we had an UPDATE_EXISTING strategy from dedup, return it now with full data.
        if 'dedup_result' in locals() and dedup_result and dedup_result.is_duplicate:
             return {
                "event": "DEDUP_SIGNAL_UPDATE",
                "dedup": dedup_result,
                "embedding": embedding,
                "record": record,
                "extraction": extraction,
                "entity_names": entity_names,
                "conflict_info": conflict_info,
            }

        return {
            "event": "PROCESS_COMPLETE",
            "record": record,
            "extraction": extraction,
            "embedding": embedding,
            "entity_names": entity_names,
            "conflict_info": conflict_info,
            "dedup_result": None # Placeholder
        }