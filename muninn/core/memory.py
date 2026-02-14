"""
Muninn Memory Engine
---------------------
Central memory management class that replaces mem0.Memory.

Composes all subsystems:
- SQLite metadata store (records, importance, access patterns)
- Qdrant vector store (semantic similarity)
- Kuzu graph store (entity/relation knowledge graph)
- BM25 index (keyword precision)
- Extraction pipeline (entity/relation extraction)
- Hybrid retrieval engine (multi-signal + RRF + reranking)
- Importance scoring (multi-factor)
- Consolidation daemon (background maintenance)

API is designed to be a drop-in replacement for mem0.Memory
while providing significantly richer capabilities.
"""

import asyncio
import hashlib
import json
import uuid
import time
import logging
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from muninn.core.types import (
    MemoryRecord, MemoryType, Provenance, SearchResult,
    ExtractionResult, Entity, Relation,
)
from muninn.core.config import MuninnConfig
from muninn.store.sqlite_metadata import SQLiteMetadataStore
from muninn.store.vector_store import VectorStore
from muninn.store.graph_store import GraphStore
from muninn.retrieval.bm25 import BM25Index
from muninn.retrieval.reranker import Reranker
from muninn.retrieval.hybrid import HybridRetriever
from muninn.extraction.pipeline import ExtractionPipeline
from muninn.scoring.importance import calculate_importance, calculate_novelty
from muninn.consolidation.daemon import ConsolidationDaemon
from muninn.core.feature_flags import get_flags
from muninn.goal import GoalCompass
from muninn.observability import OTelGenAITracer
from muninn.chains import MemoryChainDetector
from muninn.ingestion import IngestionPipeline, discover_legacy_sources as discover_legacy_sources_catalog
from muninn.ingestion.parser import infer_source_type

logger = logging.getLogger("Muninn")


class MuninnMemory:
    """
    Muninn persistent memory engine — local-first, assistant-agnostic.

    This is the primary interface for all memory operations. It replaces
    mem0.Memory with a richer, fully local architecture featuring:

    - Multi-type memory hierarchy (Working → Episodic → Semantic → Procedural)
    - Multi-signal hybrid retrieval with RRF fusion
    - Cross-encoder reranking for precision
    - Tiered entity extraction (rules → xLAM → Ollama)
    - Graph-based knowledge linking (Kuzu)
    - Importance scoring with exponential decay
    - Background consolidation (merge, promote, replay)
    - Bi-temporal timestamps for event vs. ingestion time

    Usage:
        config = MuninnConfig.from_env()
        memory = MuninnMemory(config)
        await memory.initialize()

        # Add a memory
        result = await memory.add("User prefers dark mode", user_id="user1")

        # Search
        results = await memory.search("What theme does the user prefer?")

        # Cleanup
        await memory.shutdown()
    """

    def __init__(self, config: Optional[MuninnConfig] = None):
        self.config = config or MuninnConfig.from_env()
        self.config.ensure_directories()

        # Stores
        self._metadata: Optional[SQLiteMetadataStore] = None
        self._vectors: Optional[VectorStore] = None
        self._graph: Optional[GraphStore] = None
        self._bm25: Optional[BM25Index] = None

        # Engines
        self._extraction: Optional[ExtractionPipeline] = None
        self._retriever: Optional[HybridRetriever] = None
        self._reranker: Optional[Reranker] = None
        self._consolidation: Optional[ConsolidationDaemon] = None
        self._goal_compass: Optional[GoalCompass] = None
        self._otel = OTelGenAITracer(enabled=False)
        self._ingestion: Optional[IngestionPipeline] = None
        self._chain_detector: Optional[MemoryChainDetector] = None

        # Phase 2 engines (v3.2.0)
        self._dedup = None           # SemanticDedup
        self._conflict_detector = None  # ConflictDetector
        self._conflict_resolver = None  # ConflictResolver

        # Embedding
        self._embed_model = None
        self._initialized = False
        self._user_scope_migration_complete = False
        self._feedback_multiplier_cache: Dict[Tuple[str, str, str], Tuple[float, Dict[str, float]]] = {}

    async def initialize(self) -> None:
        """
        Initialize all subsystems. Must be called before any operations.
        """
        if self._initialized:
            return

        logger.info("Initializing Muninn memory engine...")
        t0 = time.time()

        # Initialize stores
        self._metadata = SQLiteMetadataStore(self.config.metadata.path)
        self._vectors = VectorStore(
            data_path=self.config.vector.path,
            collection_name=self.config.vector.collection,
            embedding_dims=self.config.vector.dimensions,
        )
        self._graph = GraphStore(self.config.graph.path)
        self._bm25 = BM25Index()

        # Initialize embedding model
        self._init_embedding()

        # Initialize reranker
        if self.config.reranker.enabled:
            self._reranker = Reranker(model_name=self.config.reranker.model)
        else:
            self._reranker = None

        # Initialize extraction pipeline
        self._extraction = ExtractionPipeline(
            xlam_url=self.config.extraction.xlam_url if self.config.extraction.enable_xlam else None,
            ollama_url=self.config.extraction.ollama_url if self.config.extraction.enable_ollama_fallback else None,
            instructor_base_url=(
                self.config.extraction.instructor_base_url
                if self.config.extraction.enable_instructor
                else None
            ),
            instructor_model=self.config.extraction.instructor_model,
            instructor_api_key=self.config.extraction.instructor_api_key,
        )

        # Initialize hybrid retriever
        self._retriever = HybridRetriever(
            metadata_store=self._metadata,
            vector_store=self._vectors,
            graph_store=self._graph,
            bm25_index=self._bm25,
            reranker=self._reranker,
            embed_fn=self._embed,
            telemetry=self._otel,
            chain_signal_weight=self.config.memory_chains.retrieval_signal_weight,
            chain_expansion_limit=self.config.memory_chains.retrieval_expansion_limit,
            chain_max_seed_memories=self.config.memory_chains.retrieval_seed_limit,
        )

        # Initialize consolidation daemon
        self._consolidation = ConsolidationDaemon(
            config=self.config.consolidation,
            metadata=self._metadata,
            vectors=self._vectors,
            graph=self._graph,
            bm25=self._bm25,
            embed_fn=self._embed,
        )

        # Initialize Phase 2 engines (v3.2.0) — gated by feature flags
        flags = get_flags()
        self._otel = OTelGenAITracer(enabled=flags.is_enabled("otel_genai"))
        if self._retriever is not None:
            self._retriever._telemetry = self._otel

        if flags.is_enabled("goal_compass"):
            self._goal_compass = GoalCompass(
                metadata_store=self._metadata,
                embed_fn=self._embed,
                drift_threshold=self.config.goal_compass.drift_threshold,
                signal_weight=self.config.goal_compass.signal_weight,
                reminder_max_chars=self.config.goal_compass.reminder_max_chars,
            )
            logger.info(
                "Goal compass enabled (drift_threshold=%.3f, signal_weight=%.3f)",
                self.config.goal_compass.drift_threshold,
                self.config.goal_compass.signal_weight,
            )

        if flags.is_enabled("multi_source_ingestion"):
            self._ingestion = IngestionPipeline(
                max_file_size_bytes=self.config.ingestion.max_file_size_bytes,
                chunk_size_chars=self.config.ingestion.chunk_size_chars,
                chunk_overlap_chars=self.config.ingestion.chunk_overlap_chars,
                min_chunk_chars=self.config.ingestion.min_chunk_chars,
                allowed_roots=self.config.ingestion.allowed_roots,
            )
            logger.info(
                "Multi-source ingestion enabled (max_file_size_bytes=%d, chunk=%d/%d)",
                self.config.ingestion.max_file_size_bytes,
                self.config.ingestion.chunk_size_chars,
                self.config.ingestion.chunk_overlap_chars,
            )

        if flags.is_enabled("memory_chains"):
            self._chain_detector = MemoryChainDetector(
                threshold=self.config.memory_chains.detection_threshold,
                max_hours_apart=self.config.memory_chains.max_hours_apart,
                max_links_per_memory=self.config.memory_chains.max_links_per_memory,
            )
            logger.info(
                "Memory chains enabled (threshold=%.3f, max_hours=%.1f, max_links=%d)",
                self.config.memory_chains.detection_threshold,
                self.config.memory_chains.max_hours_apart,
                self.config.memory_chains.max_links_per_memory,
            )

        if flags.is_enabled("semantic_dedup"):
            from muninn.dedup.semantic_dedup import SemanticDedup
            self._dedup = SemanticDedup(
                threshold=self.config.semantic_dedup.threshold,
                content_overlap_threshold=self.config.semantic_dedup.content_overlap_threshold,
            )
            logger.info("Semantic dedup enabled (threshold=%.3f)", self.config.semantic_dedup.threshold)

        if flags.is_enabled("conflict_detection"):
            try:
                from muninn.conflict.detector import ConflictDetector
                from muninn.conflict.resolver import ConflictResolver
                self._conflict_detector = ConflictDetector(
                    model_name=self.config.conflict_detection.model_name,
                    contradiction_threshold=self.config.conflict_detection.contradiction_threshold,
                    similarity_prefilter=self.config.conflict_detection.similarity_prefilter,
                )
                if self._conflict_detector.is_available:
                    self._conflict_resolver = ConflictResolver(
                        metadata_store=self._metadata,
                        vector_store=self._vectors,
                        graph_store=self._graph,
                        bm25_index=self._bm25,
                        embed_fn=self._embed,
                    )
                    logger.info("Conflict detection enabled (model=%s)", self.config.conflict_detection.model_name)
                else:
                    logger.info("Conflict detection flag ON but NLI model unavailable (install transformers+torch)")
                    self._conflict_detector = None
            except Exception as e:
                logger.warning("Conflict detection initialization failed: %s", e)
                self._conflict_detector = None
                self._conflict_resolver = None

        # Controlled migration for legacy records that predate user_id metadata scope.
        migration_complete = self._metadata.get_meta("user_scope_migration_complete", "0") == "1"
        if not migration_complete:
            migration_stats = self._run_user_scope_migration(batch_size=500, max_batches=5)
            migration_complete = bool(migration_stats["complete"])
            logger.info(
                "User scope migration progress: updated=%d retried=%d remaining_failures=%d complete=%s",
                migration_stats["updated"],
                migration_stats["retried"],
                migration_stats["remaining_failures"],
                migration_complete,
            )
        self._user_scope_migration_complete = migration_complete

        # Rebuild BM25 index from existing memories
        await self._rebuild_bm25()

        # Start consolidation daemon
        await self._consolidation.start()

        self._initialized = True
        elapsed = time.time() - t0
        count = self._metadata.count()
        logger.info("Muninn initialized: %d memories loaded in %.2fs", count, elapsed)

    async def shutdown(self) -> None:
        """Gracefully shut down all subsystems."""
        if self._consolidation:
            await self._consolidation.stop()
        logger.info("Muninn shut down")
        self._initialized = False
        self._user_scope_migration_complete = False
        self._feedback_multiplier_cache.clear()

    # ==========================================
    # Core Memory Operations (mem0-compatible API)
    # ==========================================

    def _record_matches_scope(self, record: MemoryRecord, namespace: str, user_id: str) -> bool:
        """Return True when a record belongs to the provided namespace/user scope."""
        if record.namespace != namespace:
            return False

        metadata = record.metadata or {}
        record_user_id = metadata.get("user_id")
        return record_user_id == user_id

    @staticmethod
    def _extract_entity_names(extraction: ExtractionResult) -> List[str]:
        """Derive unique entity names from extraction output, preserving order."""
        names: List[str] = []
        seen = set()
        for entity in extraction.entities:
            raw_name = getattr(entity, "name", None)
            if not isinstance(raw_name, str):
                continue
            clean = raw_name.strip()
            if not clean:
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            names.append(clean)
        return names

    def _upsert_memory_chain_links(
        self,
        *,
        successor_record: MemoryRecord,
        successor_content: str,
        successor_entity_names: List[str],
    ) -> int:
        """Detect and persist memory-chain links for a memory record."""
        if (
            self._chain_detector is None
            or self._metadata is None
            or self._graph is None
            or not successor_entity_names
        ):
            return 0

        metadata = successor_record.metadata or {}
        user_id = str(metadata.get("user_id") or "global_user")

        try:
            candidate_records = self._metadata.get_all(
                limit=self.config.memory_chains.candidate_scan_limit,
                project=successor_record.project,
                namespace=successor_record.namespace,
                user_id=user_id,
            )
            links = self._chain_detector.detect_links(
                successor_record=successor_record,
                successor_content=successor_content,
                successor_entity_names=successor_entity_names,
                candidate_records=candidate_records,
            )

            persisted = 0
            for link in links:
                created = self._graph.add_chain_link(
                    predecessor_id=link.predecessor_id,
                    successor_id=link.successor_id,
                    relation_type=link.relation_type,
                    confidence=link.confidence,
                    reason=link.reason,
                    shared_entities=link.shared_entities,
                    hours_apart=link.hours_apart,
                )
                if created:
                    persisted += 1
            return persisted
        except Exception as e:
            logger.warning("Memory-chain linking failed (non-fatal): %s", e)
            return 0

    async def add(
        self,
        content: str,
        user_id: str = "global_user",
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: str = "global",
        memory_type: MemoryType = MemoryType.EPISODIC,
        provenance: Provenance = Provenance.AUTO_EXTRACTED,
    ) -> Dict[str, Any]:
        """
        Add a new memory.

        Args:
            content: The memory content text.
            user_id: User identifier.
            agent_id: Agent/assistant identifier.
            metadata: Optional key-value metadata.
            namespace: Memory namespace for isolation.
            memory_type: Type of memory (default: episodic).
            provenance: How this memory was created.

        Returns:
            Dict with 'id', 'content', 'importance', and extraction results.
        """
        self._check_initialized()
        with self._otel.span(
            "muninn.memory.add",
            {
                "gen_ai.operation.name": "memory.add",
                "gen_ai.system": "muninn",
                "muninn.namespace": namespace,
                "muninn.user_id": user_id,
                "muninn.project": (metadata or {}).get("project", "global"),
            },
        ):
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

            extraction = await self._extract(content)
            entity_names = self._extract_entity_names(extraction)
            if entity_names:
                scoped_metadata["entity_names"] = entity_names
            embedding = self._embed(content)

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
            )

            dedup_result = None
            if self._dedup and self._vectors.count() > 0:
                from muninn.dedup.semantic_dedup import DedupStrategy

                dedup_result = self._dedup.check_duplicate(
                    embedding=embedding,
                    content=content,
                    vector_store=self._vectors,
                    metadata_store=self._metadata,
                    filters=scope_filters,
                )
                if dedup_result and dedup_result.is_duplicate:
                    if dedup_result.strategy == DedupStrategy.SKIP:
                        logger.info(
                            "Dedup SKIP: duplicate of %s (sim=%.3f)",
                            dedup_result.existing_memory_id,
                            dedup_result.similarity,
                        )
                        return {
                            "id": None,
                            "content": content,
                            "event": "DEDUP_SKIP",
                            "dedup": dedup_result.model_dump(),
                        }
                    if dedup_result.strategy == DedupStrategy.UPDATE_EXISTING:
                        existing = self._metadata.get(dedup_result.existing_memory_id)
                        if existing:
                            if not self._record_matches_scope(existing, namespace, user_id):
                                logger.warning(
                                    "Dedup UPDATE_EXISTING scope mismatch: memory_id=%s namespace=%s user_id=%s",
                                    dedup_result.existing_memory_id,
                                    existing.namespace,
                                    (existing.metadata or {}).get("user_id"),
                                )
                            else:
                                merged_content = self._dedup.merge_content(content, existing.content)
                                self._metadata.update(
                                    dedup_result.existing_memory_id,
                                    content=merged_content,
                                )
                                merged_embedding = self._embed(merged_content)
                                self._vectors.upsert(
                                    memory_id=dedup_result.existing_memory_id,
                                    embedding=merged_embedding,
                                    metadata={
                                        "content": merged_content[:500],
                                        "memory_type": existing.memory_type.value,
                                        "namespace": namespace,
                                        "importance": existing.importance,
                                        "user_id": user_id,
                                        "project": existing.project,
                                        "branch": existing.branch,
                                    },
                                )
                                self._bm25.add(dedup_result.existing_memory_id, merged_content)
                                logger.info(
                                    "Dedup UPDATE_EXISTING: merged into %s",
                                    dedup_result.existing_memory_id,
                                )
                                return {
                                    "id": dedup_result.existing_memory_id,
                                    "content": merged_content,
                                    "event": "DEDUP_MERGED",
                                    "dedup": dedup_result.model_dump(),
                                }

            conflict_info = None
            if self._conflict_detector and self._vectors.count() > 0:
                try:
                    similar_for_conflict = self._vectors.search(
                        embedding,
                        limit=5,
                        score_threshold=self.config.conflict_detection.similarity_prefilter,
                        filters=scope_filters,
                    )
                    if similar_for_conflict:
                        candidate_ids = [mid for mid, _score in similar_for_conflict]
                        candidate_records = [
                            candidate
                            for candidate in self._metadata.get_by_ids(candidate_ids)
                            if self._record_matches_scope(candidate, namespace, user_id)
                        ]
                        if candidate_records:
                            conflicts = self._conflict_detector.detect_conflicts(content, candidate_records)
                            if conflicts and self._conflict_resolver:
                                conflicts.sort(key=lambda c: c.contradiction_score, reverse=True)
                                conflict = conflicts[0]
                                resolution = self._conflict_resolver.resolve(
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
                                    logger.info(
                                        "Conflict resolution: %s — skipping new storage",
                                        resolution.get("resolution"),
                                    )
                                    return {
                                        "id": None,
                                        "content": content,
                                        "event": f"CONFLICT_{resolution['resolution'].upper()}",
                                        "conflict": conflict_info,
                                    }
                except Exception as e:
                    logger.warning("Conflict detection failed (non-fatal): %s", e)

            max_similarity = 0.0
            if self._vectors.count() > 0:
                try:
                    similar = self._vectors.search(embedding, limit=5, filters=scope_filters)
                    if similar:
                        max_similarity = similar[0][1]
                except Exception:
                    pass

            record.novelty_score = calculate_novelty(max_similarity)

            centrality = 0.1 if extraction.entities else 0.0
            importance = calculate_importance(
                record,
                max_similarity=max_similarity,
                centrality=centrality,
            )
            record.importance = importance

            self._metadata.add(record)
            self._vectors.upsert(
                memory_id=record.id,
                embedding=embedding,
                metadata={
                    "content": content[:500],
                    "memory_type": memory_type.value,
                    "namespace": namespace,
                    "importance": importance,
                    "user_id": user_id,
                    "project": project,
                    "branch": branch,
                },
            )

            self._graph.add_memory_node(record.id, extraction.summary or content[:200])
            for entity in extraction.entities:
                self._graph.add_entity(entity.name, entity.entity_type)
                self._graph.link_memory_to_entity(record.id, entity.name, "mentions")
            for relation in extraction.relations:
                self._graph.add_entity(relation.subject, "concept")
                self._graph.add_entity(relation.object, "concept")
                self._graph.add_relation(
                    relation.subject,
                    relation.predicate,
                    relation.object,
                    record.id,
                    relation.confidence,
                )

            chain_links_created = self._upsert_memory_chain_links(
                successor_record=record,
                successor_content=content,
                successor_entity_names=entity_names,
            )

            self._bm25.add(record.id, content)
            logger.info(
                "Added memory %s (importance=%.3f, entities=%d, relations=%d, chains=%d)",
                record.id,
                importance,
                len(extraction.entities),
                len(extraction.relations),
                chain_links_created,
            )

            result = {
                "id": record.id,
                "content": content,
                "importance": importance,
                "memory_type": memory_type.value,
                "entities": [e.model_dump() for e in extraction.entities],
                "relations": [r.model_dump() for r in extraction.relations],
                "chain_links_created": chain_links_created,
                "event": "ADD",
            }
            if conflict_info:
                result["conflict"] = conflict_info
            if dedup_result:
                result["dedup"] = dedup_result.model_dump()
            if self._goal_compass is not None and project:
                drift = await self._goal_compass.evaluate_drift(
                    text=content,
                    user_id=user_id,
                    namespace=namespace,
                    project=project,
                )
                if drift is not None:
                    result["goal_alignment"] = drift

            self._otel.add_event(
                "muninn.add.result",
                {"memory_id": record.id, "importance": importance},
            )
            return result

    async def search(
        self,
        query: str,
        user_id: str = "global_user",
        agent_id: Optional[str] = None,
        limit: int = 10,
        rerank: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        namespaces: Optional[List[str]] = None,
        explain: bool = False,
        project: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search memories using hybrid multi-signal retrieval.

        Args:
            query: Search query text.
            user_id: User filter.
            agent_id: Agent filter.
            limit: Maximum results.
            rerank: Whether to apply cross-encoder reranking.
            filters: Additional metadata filters.
            namespaces: Namespace filter list.
            explain: When True, include RecallTrace per result (v3.1.0).

        Returns:
            List of memory dicts with scores.
        """
        self._check_initialized()

        with self._otel.span(
            "muninn.memory.search",
            {
                "gen_ai.operation.name": "memory.search",
                "gen_ai.system": "muninn",
                "muninn.user_id": user_id,
                "muninn.limit": limit,
            },
        ):
            self._otel.add_event(
                "muninn.search.request",
                {"query_preview": self._otel.maybe_content(query)},
            )
            effective_filters = dict(filters or {})
            if project and "project" not in effective_filters:
                effective_filters["project"] = project

            resolved_namespace = "global"
            if namespaces and len(namespaces) == 1:
                resolved_namespace = namespaces[0]
            elif effective_filters.get("namespace"):
                resolved_namespace = str(effective_filters["namespace"])

            resolved_project = str(effective_filters.get("project") or project or "global")
            goal_embedding = None
            goal_alignment = None
            feedback_signal_multipliers = None
            if self._goal_compass is not None and resolved_project:
                goal = await self._goal_compass.get_goal(
                    user_id=user_id,
                    namespace=resolved_namespace,
                    project=resolved_project,
                )
                if goal:
                    goal_embedding = goal.get("goal_embedding")
                    goal_alignment = await self._goal_compass.evaluate_drift(
                        text=query,
                        user_id=user_id,
                        namespace=resolved_namespace,
                        project=resolved_project,
                    )
            flags = get_flags()
            if (
                flags.is_enabled("retrieval_feedback")
                and self.config.retrieval_feedback.enabled
                and resolved_project
            ):
                feedback_signal_multipliers = self._get_feedback_signal_multipliers_cached(
                    user_id=user_id,
                    namespace=resolved_namespace,
                    project=resolved_project,
                )

            results = await self._retriever.search(
                query=query,
                limit=limit,
                user_id=user_id,
                filters=effective_filters,
                rerank=rerank,
                namespaces=namespaces,
                explain=explain,
                goal_embedding=goal_embedding,
                goal_signal_weight=self.config.goal_compass.signal_weight,
                feedback_signal_multipliers=feedback_signal_multipliers,
            )

            output = []
            for r in results:
                item = {
                    "id": r.memory.id,
                    "memory": r.memory.content,
                    "score": r.score,
                    "source": r.source,
                    "memory_type": r.memory.memory_type.value,
                    "importance": r.memory.importance,
                    "created_at": r.memory.created_at,
                    "metadata": r.memory.metadata,
                }
                if explain and r.trace is not None:
                    item["trace"] = r.trace.model_dump()
                if goal_alignment is not None:
                    item["goal_similarity"] = goal_alignment["similarity"]
                    if goal_alignment["is_drift"]:
                        item["goal_drift_warning"] = goal_alignment["reminder"]
                output.append(item)
            self._otel.add_event("muninn.search.result", {"result_count": len(output)})
            return output

    async def get_all(
        self,
        user_id: str = "global_user",
        agent_id: Optional[str] = None,
        limit: int = 100,
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all memories, optionally filtered.

        Returns:
            List of memory dicts ordered by importance.
        """
        self._check_initialized()

        records = self._metadata.get_all(
            limit=limit,
            namespace=namespace,
            user_id=user_id,
        )

        return [
            {
                "id": r.id,
                "memory": r.content,
                "memory_type": r.memory_type.value,
                "importance": r.importance,
                "created_at": r.created_at,
                "access_count": r.access_count,
                "metadata": r.metadata,
            }
            for r in records
        ]

    def _get_feedback_signal_multipliers_cached(
        self,
        *,
        user_id: str,
        namespace: str,
        project: str,
    ) -> Dict[str, float]:
        """Fetch scoped feedback multipliers with short TTL cache."""
        cache_key = (user_id, namespace, project)
        now = time.time()
        cached = self._feedback_multiplier_cache.get(cache_key)
        if cached and cached[0] > now:
            return dict(cached[1])

        multipliers = self._metadata.get_feedback_signal_multipliers(
            user_id=user_id,
            namespace=namespace,
            project=project,
            lookback_days=self.config.retrieval_feedback.lookback_days,
            min_total_signal_weight=self.config.retrieval_feedback.min_total_signal_weight,
            estimator=self.config.retrieval_feedback.estimator,
            propensity_floor=self.config.retrieval_feedback.propensity_floor,
            min_effective_samples=self.config.retrieval_feedback.min_effective_samples,
            default_sampling_prob=self.config.retrieval_feedback.default_sampling_prob,
            floor=self.config.retrieval_feedback.multiplier_floor,
            ceiling=self.config.retrieval_feedback.multiplier_ceiling,
        )
        ttl = max(1, int(self.config.retrieval_feedback.cache_ttl_seconds))
        self._feedback_multiplier_cache[cache_key] = (now + ttl, dict(multipliers))
        return multipliers

    async def record_retrieval_feedback(
        self,
        *,
        query: str,
        memory_id: str,
        outcome: float,
        user_id: str = "global_user",
        namespace: str = "global",
        project: str = "global",
        rank: Optional[int] = None,
        sampling_prob: Optional[float] = None,
        signals: Optional[Dict[str, float]] = None,
        source: str = "manual",
    ) -> Dict[str, Any]:
        """Persist retrieval feedback for adaptive weighting calibration."""
        self._check_initialized()
        feedback_id = self._metadata.add_retrieval_feedback(
            user_id=user_id,
            namespace=namespace,
            project=project,
            query_text=query,
            memory_id=memory_id,
            outcome=outcome,
            rank=rank,
            sampling_prob=sampling_prob,
            signals=signals or {},
            source=source,
        )
        self._feedback_multiplier_cache.pop((user_id, namespace, project), None)
        return {
            "feedback_id": feedback_id,
            "query": query,
            "memory_id": memory_id,
            "outcome": max(0.0, min(1.0, float(outcome))),
            "project": project,
            "namespace": namespace,
            "rank": int(rank) if isinstance(rank, int) and rank > 0 else None,
            "sampling_prob": (
                max(0.0, min(1.0, float(sampling_prob)))
                if isinstance(sampling_prob, (int, float))
                else None
            ),
            "source": source,
        }

    async def set_project_goal(
        self,
        *,
        user_id: str = "global_user",
        namespace: str = "global",
        project: str,
        goal_statement: str,
        constraints: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Set/update a scoped project goal and cache its embedding."""
        self._check_initialized()
        if self._goal_compass is None:
            raise RuntimeError("Goal compass is disabled by feature flag")
        if not goal_statement.strip():
            raise ValueError("goal_statement cannot be empty")
        return await self._goal_compass.set_goal(
            user_id=user_id,
            namespace=namespace,
            project=project,
            goal_statement=goal_statement,
            constraints=constraints or [],
        )

    async def get_project_goal(
        self,
        *,
        user_id: str = "global_user",
        namespace: str = "global",
        project: str,
    ) -> Optional[Dict[str, Any]]:
        """Fetch a scoped project goal if configured."""
        self._check_initialized()
        if self._goal_compass is None:
            return None
        return await self._goal_compass.get_goal(
            user_id=user_id,
            namespace=namespace,
            project=project,
        )

    @staticmethod
    def _handoff_payload_for_checksum(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return canonical payload subset used for checksum generation."""
        return {
            "schema_version": payload.get("schema_version", 1),
            "project": payload.get("project"),
            "namespace": payload.get("namespace"),
            "user_id": payload.get("user_id"),
            "goal": payload.get("goal"),
            "decisions": payload.get("decisions", []),
            "open_questions": payload.get("open_questions", []),
            "memories": payload.get("memories", []),
            "watermark_created_at": payload.get("watermark_created_at"),
        }

    @classmethod
    def _handoff_checksum(cls, payload: Dict[str, Any]) -> str:
        canonical = cls._handoff_payload_for_checksum(payload)
        blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    async def export_handoff(
        self,
        *,
        user_id: str = "global_user",
        namespace: str = "global",
        project: str,
        limit: int = 25,
    ) -> Dict[str, Any]:
        """Export deterministic project handoff bundle for cross-assistant continuity."""
        self._check_initialized()
        goal = await self.get_project_goal(user_id=user_id, namespace=namespace, project=project)

        records = self._metadata.get_all(
            limit=5000,
            project=project,
            namespace=namespace,
            user_id=user_id,
        )

        now = time.time()

        def _handoff_rank(record: MemoryRecord) -> float:
            age_seconds = max(0.0, now - record.created_at)
            recency = 1.0 / (1.0 + (age_seconds / 86400.0))
            return float(record.importance) * recency

        ranked_records = sorted(records, key=_handoff_rank, reverse=True)
        top_records = ranked_records[: max(1, min(limit, 500))]

        decisions = []
        open_questions = []
        memories = []
        for record in top_records:
            metadata = record.metadata or {}
            memories.append(
                {
                    "id": record.id,
                    "content": record.content,
                    "importance": record.importance,
                    "memory_type": record.memory_type.value,
                    "created_at": record.created_at,
                    "metadata": metadata,
                }
            )
            if metadata.get("kind") == "decision":
                decisions.append(
                    {
                        "id": metadata.get("decision_id", record.id),
                        "text": record.content,
                        "status": metadata.get("status", "accepted"),
                    }
                )
            if metadata.get("kind") == "open_question" or record.content.strip().endswith("?"):
                open_questions.append(record.content.strip())

        payload = {
            "schema_version": 1,
            "project": project,
            "namespace": namespace,
            "user_id": user_id,
            "exported_at": time.time(),
            "goal": goal,
            "decisions": decisions,
            "open_questions": open_questions,
            "memories": memories,
            "watermark_created_at": max((r.created_at for r in top_records), default=0.0),
        }
        checksum = self._handoff_checksum(payload)
        payload["checksum"] = f"sha256:{checksum}"
        payload["event_id"] = f"handoff:{project}:{namespace}:{checksum[:16]}"
        return payload

    async def import_handoff(
        self,
        *,
        bundle: Dict[str, Any],
        user_id: str = "global_user",
        namespace: str = "global",
        project: str,
        source: str = "handoff_import",
    ) -> Dict[str, Any]:
        """Import handoff bundle with checksum verification and idempotent replay."""
        self._check_initialized()
        if not isinstance(bundle, dict):
            raise ValueError("bundle must be an object")

        payload = dict(bundle)
        expected_checksum = payload.get("checksum", "")
        if not expected_checksum or not isinstance(expected_checksum, str):
            raise ValueError("bundle.checksum is required")

        computed = self._handoff_checksum(payload)
        checksum_ok = expected_checksum == f"sha256:{computed}"
        if not checksum_ok:
            raise ValueError("bundle checksum verification failed")

        event_id = payload.get("event_id") or f"handoff:{project}:{namespace}:{computed[:16]}"
        if self._metadata.has_handoff_event(event_id):
            return {
                "event": "HANDOFF_DUPLICATE",
                "event_id": event_id,
                "imported": 0,
                "skipped": len(payload.get("memories", [])),
                "checksum_verified": True,
            }

        imported = 0
        skipped = 0

        goal = payload.get("goal")
        if isinstance(goal, dict) and goal.get("goal_statement"):
            await self.set_project_goal(
                user_id=user_id,
                namespace=namespace,
                project=project,
                goal_statement=str(goal.get("goal_statement")),
                constraints=[str(item) for item in goal.get("constraints", [])],
            )

        memories = payload.get("memories", [])
        for item in memories:
            if not isinstance(item, dict):
                skipped += 1
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                skipped += 1
                continue

            merged_meta = dict(item.get("metadata") or {})
            merged_meta.setdefault("project", project)
            merged_meta.setdefault("kind", "handoff_memory")
            merged_meta["handoff_event_id"] = event_id
            merged_meta["handoff_source"] = source

            add_result = await self.add(
                content=content,
                user_id=user_id,
                namespace=namespace,
                metadata=merged_meta,
                provenance=Provenance.INGESTED,
            )
            if add_result.get("event") in {"DEDUP_SKIP", "CONFLICT_SKIP"}:
                skipped += 1
            else:
                imported += 1

        self._metadata.record_handoff_event(event_id=event_id, source=source)
        return {
            "event": "HANDOFF_IMPORTED",
            "event_id": event_id,
            "imported": imported,
            "skipped": skipped,
            "checksum_verified": True,
        }

    def _require_ingestion_pipeline(self) -> IngestionPipeline:
        flags = get_flags()
        flags.require("multi_source_ingestion")
        if self._ingestion is None:
            raise RuntimeError("Ingestion pipeline is unavailable")
        return self._ingestion

    def _normalize_discovery_roots(
        self,
        *,
        ingestion: IngestionPipeline,
        roots: Optional[List[str]],
    ) -> List[str]:
        normalized_roots: List[str] = []
        for root in roots or []:
            resolved = ingestion.ensure_allowed_path(root)
            normalized_roots.append(str(resolved))
        return normalized_roots

    async def _persist_ingestion_report(
        self,
        *,
        report,
        user_id: str,
        namespace: str,
        base_metadata: Dict[str, Any],
        source_context_by_path: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        source_payloads: List[Dict[str, Any]] = []
        added_memories = 0
        skipped_chunks = 0
        failed_chunks = 0
        source_context_by_path = source_context_by_path or {}

        for source_result in report.source_results:
            source_record: Dict[str, Any] = {
                "source_path": source_result.source_path,
                "source_type": source_result.source_type,
                "status": source_result.status,
                "errors": list(source_result.errors),
                "skipped_reason": source_result.skipped_reason,
                "chunks_discovered": len(source_result.chunks),
                "chunks_added": 0,
                "chunks_skipped": 0,
                "chunks_failed": 0,
            }
            if source_result.status != "processed":
                source_payloads.append(source_record)
                continue

            source_context = source_context_by_path.get(source_result.source_path, {})
            for chunk in source_result.chunks:
                chunk_metadata = dict(base_metadata)
                chunk_metadata.update(source_context)
                chunk_metadata.update(chunk.metadata)
                try:
                    add_result = await self.add(
                        content=chunk.content,
                        user_id=user_id,
                        namespace=namespace,
                        metadata=chunk_metadata,
                        provenance=Provenance.INGESTED,
                    )
                except Exception as exc:
                    failed_chunks += 1
                    source_record["chunks_failed"] += 1
                    source_record["errors"].append(
                        f"chunk[{chunk.chunk_index}] add failed: {exc}"
                    )
                    continue

                if add_result.get("event") in {"DEDUP_SKIP", "CONFLICT_SKIP"}:
                    skipped_chunks += 1
                    source_record["chunks_skipped"] += 1
                else:
                    added_memories += 1
                    source_record["chunks_added"] += 1

            source_payloads.append(source_record)

        return {
            "total_sources": report.total_sources,
            "processed_sources": report.processed_sources,
            "skipped_sources": report.skipped_sources,
            "total_chunks_discovered": report.total_chunks,
            "added_memories": added_memories,
            "skipped_chunks": skipped_chunks,
            "failed_chunks": failed_chunks,
            "source_results": source_payloads,
        }

    async def ingest_sources(
        self,
        *,
        sources: List[str],
        user_id: str = "global_user",
        namespace: str = "global",
        project: str = "global",
        metadata: Optional[Dict[str, Any]] = None,
        recursive: bool = False,
        chronological_order: str = "none",
        max_file_size_bytes: Optional[int] = None,
        chunk_size_chars: Optional[int] = None,
        chunk_overlap_chars: Optional[int] = None,
        min_chunk_chars: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Ingest multiple file sources with fail-open behavior.

        Each source is parsed independently. Parser/source failures are recorded
        and ingestion continues for remaining sources.
        """
        self._check_initialized()
        if not sources:
            raise ValueError("sources must be a non-empty list")

        ingestion = self._require_ingestion_pipeline()

        report = ingestion.ingest(
            sources,
            recursive=recursive,
            chronological_order=chronological_order,
            max_file_size_bytes=max_file_size_bytes,
            chunk_size_chars=chunk_size_chars,
            chunk_overlap_chars=chunk_overlap_chars,
            min_chunk_chars=min_chunk_chars,
        )

        base_metadata = dict(metadata or {})
        base_metadata.setdefault("project", project)
        base_metadata.setdefault("kind", "ingested_source_chunk")
        result = await self._persist_ingestion_report(
            report=report,
            user_id=user_id,
            namespace=namespace,
            base_metadata=base_metadata,
        )
        result["event"] = "INGEST_COMPLETED"
        return result

    async def discover_legacy_sources(
        self,
        *,
        roots: Optional[List[str]] = None,
        providers: Optional[List[str]] = None,
        include_unsupported: bool = False,
        max_results_per_provider: int = 100,
    ) -> Dict[str, Any]:
        self._check_initialized()
        ingestion = self._require_ingestion_pipeline()
        normalized_roots = self._normalize_discovery_roots(
            ingestion=ingestion,
            roots=roots,
        )

        discovered = discover_legacy_sources_catalog(
            roots=normalized_roots,
            include_unsupported=include_unsupported,
            max_results_per_provider=max_results_per_provider,
        )
        discovered = [
            item
            for item in discovered
            if ingestion.is_path_allowed(Path(str(item.get("path", ""))))
        ]
        if providers:
            allowed = {p.strip().lower() for p in providers if p and p.strip()}
            discovered = [
                item
                for item in discovered
                if str(item.get("provider", "")).lower() in allowed
            ]

        provider_counts: Dict[str, int] = {}
        parser_supported = 0
        for item in discovered:
            provider = str(item.get("provider", "unknown"))
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
            if item.get("parser_supported"):
                parser_supported += 1

        return {
            "event": "LEGACY_DISCOVERY_COMPLETED",
            "total_discovered": len(discovered),
            "parser_supported": parser_supported,
            "parser_unsupported": len(discovered) - parser_supported,
            "provider_counts": provider_counts,
            "sources": discovered,
        }

    async def ingest_legacy_sources(
        self,
        *,
        selected_source_ids: Optional[List[str]] = None,
        selected_paths: Optional[List[str]] = None,
        roots: Optional[List[str]] = None,
        providers: Optional[List[str]] = None,
        include_unsupported: bool = False,
        max_results_per_provider: int = 100,
        user_id: str = "global_user",
        namespace: str = "global",
        project: str = "global",
        metadata: Optional[Dict[str, Any]] = None,
        recursive: bool = False,
        chronological_order: str = "none",
        max_file_size_bytes: Optional[int] = None,
        chunk_size_chars: Optional[int] = None,
        chunk_overlap_chars: Optional[int] = None,
        min_chunk_chars: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._check_initialized()
        ingestion = self._require_ingestion_pipeline()
        normalized_roots = self._normalize_discovery_roots(
            ingestion=ingestion,
            roots=roots,
        )

        catalog = discover_legacy_sources_catalog(
            roots=normalized_roots,
            include_unsupported=True,
            max_results_per_provider=max_results_per_provider,
        )
        catalog = [
            item
            for item in catalog
            if ingestion.is_path_allowed(Path(str(item.get("path", ""))))
        ]
        if providers:
            allowed = {p.strip().lower() for p in providers if p and p.strip()}
            catalog = [
                item
                for item in catalog
                if str(item.get("provider", "")).lower() in allowed
            ]

        by_id = {str(item["source_id"]): item for item in catalog}
        by_path = {str(item["path"]): item for item in catalog}
        missing_source_ids: List[str] = []
        selected: List[Dict[str, Any]] = []
        seen_paths: set[str] = set()

        for source_id in selected_source_ids or []:
            item = by_id.get(source_id)
            if item is None:
                missing_source_ids.append(source_id)
                continue
            path = str(item["path"])
            if path in seen_paths:
                continue
            selected.append(item)
            seen_paths.add(path)

        for path in selected_paths or []:
            norm = str(ingestion.ensure_allowed_path(path))
            item = by_path.get(norm)
            if item is None:
                source_type = infer_source_type(Path(norm))
                parser_supported = source_type != "unsupported"
                size_bytes = 0
                try:
                    size_bytes = Path(norm).stat().st_size
                except OSError:
                    pass
                item = {
                    "source_id": f"manual:{hashlib.sha1(norm.encode('utf-8', errors='replace')).hexdigest()[:16]}",
                    "provider": "manual_selection",
                    "category": "assistant_chat",
                    "path": norm,
                    "source_type": source_type,
                    "parser_supported": parser_supported,
                    "confidence": "manual",
                    "size_bytes": size_bytes,
                    "notes": "Explicit path selected by user",
                }
            if item["path"] in seen_paths:
                continue
            selected.append(item)
            seen_paths.add(str(item["path"]))

        if not selected:
            raise ValueError("No legacy sources selected. Provide selected_source_ids and/or selected_paths.")

        supported_selected = [
            item
            for item in selected
            if item.get("parser_supported") is True
        ]
        unsupported_selected = [
            item
            for item in selected
            if item.get("parser_supported") is not True
        ]
        unsupported_payload = unsupported_selected if include_unsupported else []

        sources = [str(item["path"]) for item in supported_selected]

        report = ingestion.ingest(
            sources,
            recursive=recursive,
            chronological_order=chronological_order,
            max_file_size_bytes=max_file_size_bytes,
            chunk_size_chars=chunk_size_chars,
            chunk_overlap_chars=chunk_overlap_chars,
            min_chunk_chars=min_chunk_chars,
        )

        source_context_by_path: Dict[str, Dict[str, Any]] = {}
        for item in supported_selected:
            source_context_by_path[str(item["path"])] = {
                "legacy_import": True,
                "legacy_source_id": item.get("source_id"),
                "legacy_source_provider": item.get("provider"),
                "legacy_source_category": item.get("category"),
                "legacy_source_confidence": item.get("confidence"),
                "legacy_source_notes": item.get("notes"),
            }

        base_metadata = dict(metadata or {})
        base_metadata.setdefault("project", project)
        base_metadata.setdefault("kind", "legacy_ingested_source_chunk")
        result = await self._persist_ingestion_report(
            report=report,
            user_id=user_id,
            namespace=namespace,
            base_metadata=base_metadata,
            source_context_by_path=source_context_by_path,
        )
        result.update(
            {
                "event": "LEGACY_INGEST_COMPLETED",
                "discovery_candidates": len(catalog),
                "selected_sources": len(selected),
                "selected_supported_sources": len(supported_selected),
                "selected_unsupported_sources": len(unsupported_selected),
                "missing_source_ids": missing_source_ids,
                "unsupported_sources": unsupported_payload,
            }
        )
        return result

    async def update(self, memory_id: str, data: str) -> Dict[str, Any]:
        """
        Update a memory's content.

        Args:
            memory_id: ID of the memory to update.
            data: New content text.

        Returns:
            Updated memory dict.
        """
        self._check_initialized()

        record = self._metadata.get(memory_id)
        if not record:
            return {"error": f"Memory {memory_id} not found"}

        old_content = record.content
        record.content = data

        # Re-extract entities
        extraction = await self._extract(data)
        entity_names = self._extract_entity_names(extraction)
        updated_metadata = dict(record.metadata or {})
        if entity_names:
            updated_metadata["entity_names"] = entity_names
        else:
            updated_metadata.pop("entity_names", None)
        record.metadata = updated_metadata

        # Re-embed
        embedding = self._embed(data)

        # Update stores
        self._metadata.update(record.id, content=record.content, metadata=record.metadata)
        self._vectors.upsert(
            memory_id=record.id,
            embedding=embedding,
            metadata={
                "content": data[:500],
                "memory_type": record.memory_type.value,
                "namespace": record.namespace,
                "importance": record.importance,
                "user_id": record.metadata.get("user_id", "global_user"),
                "project": record.project,
                "branch": record.branch,
            },
        )

        # Update graph
        self._graph.delete_memory_references(record.id)
        self._graph.add_memory_node(record.id, extraction.summary or data[:200])
        for entity in extraction.entities:
            self._graph.add_entity(entity.name, entity.entity_type)
            self._graph.link_memory_to_entity(record.id, entity.name, "mentions")
        for relation in extraction.relations:
            self._graph.add_entity(relation.subject, "concept")
            self._graph.add_entity(relation.object, "concept")
            self._graph.add_relation(
                relation.subject, relation.predicate,
                relation.object, record.id, relation.confidence,
            )

        chain_links_created = self._upsert_memory_chain_links(
            successor_record=record,
            successor_content=data,
            successor_entity_names=entity_names,
        )

        # Update BM25
        self._bm25.add(record.id, data)

        logger.info("Updated memory %s (chains=%d)", memory_id, chain_links_created)

        return {
            "id": record.id,
            "content": data,
            "previous_content": old_content,
            "chain_links_created": chain_links_created,
            "event": "UPDATE",
        }

    async def delete(self, memory_id: str) -> Dict[str, str]:
        """Delete a memory from all stores."""
        self._check_initialized()

        self._metadata.delete(memory_id)
        self._vectors.delete(memory_id)
        self._graph.delete_memory_references(memory_id)
        self._bm25.remove(memory_id)

        logger.info("Deleted memory %s", memory_id)
        return {"id": memory_id, "event": "DELETE"}

    async def delete_all(self, user_id: str = "global_user") -> Dict[str, Any]:
        """Delete all memories for the given user.

        Scoped by ``user_id`` so that one tenant cannot wipe another's data.
        Individual vectors and BM25 entries are removed per-record to keep
        the stores consistent without a full collection nuke.
        """
        self._check_initialized()

        # 1. Collect IDs to delete from vector / BM25 stores
        records = self._metadata.get_all(user_id=user_id, limit=100_000)
        memory_ids = [r.id for r in records]

        # 2. User-scoped deletion in SQLite
        count = self._metadata.delete_all(user_id=user_id)

        # 3. Remove matching vectors individually (best-effort)
        for mid in memory_ids:
            try:
                self._vectors.delete(mid)
            except Exception:
                logger.debug("Vector delete skipped for %s", mid)

        # 4. Remove matching BM25 documents individually
        for mid in memory_ids:
            self._bm25.remove(mid)

        # 5. Clean up graph references
        for mid in memory_ids:
            try:
                self._graph.delete_memory_references(mid)
            except Exception:
                logger.debug("Graph cleanup skipped for %s", mid)

        logger.info("Deleted %d memories for user %s", count, user_id)
        return {"event": "DELETE_ALL", "user_id": user_id, "deleted_count": count}

    async def health(self) -> Dict[str, Any]:
        """Return system health status."""
        self._check_initialized()

        return {
            "status": "ok",
            "memory_count": self._metadata.count(),
            "vector_count": self._vectors.count(),
            "graph_nodes": len(self._graph.get_all_entities()),
            "bm25_size": self._bm25.size,
            "reranker": "active" if (self._reranker and self._reranker.is_available) else "inactive",
            "consolidation": self._consolidation.status if self._consolidation else {"running": False},
            "backend": "muninn-native",
        }

    # ==========================================
    # Internal Methods
    # ==========================================

    def _init_embedding(self) -> None:
        """Initialize the embedding model."""
        try:
            from fastembed import TextEmbedding
            self._embed_model = TextEmbedding(
                model_name=f"nomic-ai/{self.config.embedding.model}-v1.5"
                if "nomic" in self.config.embedding.model
                else self.config.embedding.model
            )
            logger.info("Embedding model loaded: fastembed/%s", self.config.embedding.model)
        except ImportError:
            logger.info("fastembed not available — falling back to Ollama embeddings")
            self._embed_model = None
        except Exception as e:
            logger.warning("FastEmbed init failed: %s — falling back to Ollama", e)
            self._embed_model = None

    def _embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if self._embed_model is not None:
            # fastembed
            embeddings = list(self._embed_model.embed([text]))
            return embeddings[0].tolist()
        else:
            # Ollama fallback
            return self._ollama_embed(text)

    def _ollama_embed(self, text: str) -> List[float]:
        """Generate embedding via Ollama API."""
        import httpx
        try:
            response = httpx.post(
                f"{self.config.embedding.ollama_url}/api/embeddings",
                json={
                    "model": self.config.embedding.model,
                    "prompt": text,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            logger.error("Ollama embedding failed: %s", e)
            # Return zero vector as absolute fallback
            return [0.0] * self.config.embedding.dimensions

    async def _extract(self, content: str) -> ExtractionResult:
        """Run extraction pipeline on content."""
        if self._extraction:
            return self._extraction.extract(content)
        return ExtractionResult()

    async def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from all metadata records."""
        records = self._metadata.get_all(limit=10000)
        documents = {r.id: r.content for r in records}
        self._bm25.rebuild(documents)

    def _run_user_scope_migration(
        self,
        default_user_id: str = "global_user",
        batch_size: int = 500,
        max_batches: int = 5,
    ) -> Dict[str, int]:
        """Run a bounded migration pass and retry ledger for legacy user scope backfill."""
        updated = 0
        retried = 0

        # Retry previous vector payload failures first.
        retry_ids = self._metadata.get_user_scope_backfill_failures(limit=batch_size)
        if retry_ids:
            for memory_id in retry_ids:
                try:
                    self._vectors.set_payload(memory_id, {"user_id": default_user_id})
                    self._metadata.clear_user_scope_backfill_failure(memory_id)
                    retried += 1
                except Exception as e:
                    self._metadata.record_user_scope_backfill_failure(memory_id, str(e))

        for _ in range(max_batches):
            records = self._metadata.get_missing_user_id_records(limit=batch_size)
            if not records:
                break

            for record in records:
                updated_metadata = {**record.metadata, "user_id": default_user_id}
                self._metadata.update(record.id, metadata=updated_metadata)

                try:
                    self._vectors.set_payload(record.id, {"user_id": default_user_id})
                    self._metadata.clear_user_scope_backfill_failure(record.id)
                except Exception as e:
                    self._metadata.record_user_scope_backfill_failure(record.id, str(e))

                updated += 1

        # Determine completion based on remaining records missing user_id and retry failures.
        remaining_missing = self._metadata.count_missing_user_id()

        remaining_failures = self._metadata.count_user_scope_backfill_failures()
        complete = remaining_missing == 0 and remaining_failures == 0
        self._metadata.set_meta("user_scope_migration_complete", "1" if complete else "0")

        return {
            "updated": updated,
            "retried": retried,
            "remaining_missing": remaining_missing,
            "remaining_failures": remaining_failures,
            "complete": int(complete),
        }

    def _check_initialized(self) -> None:
        """Raise if not initialized."""
        if not self._initialized:
            raise RuntimeError("MuninnMemory not initialized. Call await memory.initialize() first.")

    @property
    def count(self) -> int:
        """Total number of memories."""
        if self._metadata:
            return self._metadata.count()
        return 0
