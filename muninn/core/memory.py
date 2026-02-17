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
- Bi-temporal timestamps for event vs. ingestion time

API is designed to be be a drop-in replacement for mem0.Memory
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
from muninn.core.config import MuninnConfig, SUPPORTED_MODEL_PROFILES
from muninn.store.sqlite_metadata import SQLiteMetadataStore
from muninn.store.vector_store import VectorStore
from muninn.store.graph_store import GraphStore
from muninn.retrieval.bm25 import BM25Index
from muninn.retrieval.reranker import Reranker
from muninn.retrieval.hybrid import HybridRetriever
from muninn.extraction.pipeline import ExtractionPipeline
from muninn.scoring.importance import calculate_importance, calculate_novelty
from muninn.consolidation.daemon import ConsolidationDaemon
from muninn.goal import GoalCompass
from muninn.observability import OTelGenAITracer
from muninn.chains import MemoryChainDetector
from muninn.ingestion import IngestionPipeline, discover_legacy_sources as discover_legacy_sources_catalog
from muninn.ingestion.parser import infer_source_type
from muninn.core.ingestion_manager import IngestionManager
from muninn.advanced.temporal_kg import TemporalKnowledgeGraph
from muninn.advanced.cross_agent import FederationManager
from muninn.core.feature_flags import get_flags

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

        # Managers
        self._ingestion_manager: Optional[IngestionManager] = None
        self._temporal_kg: Optional[TemporalKnowledgeGraph] = None
        self._federation: Optional[FederationManager] = None

        # Locking
        self._write_lock = asyncio.Lock()

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
            xlam_model=self.config.extraction.xlam_model,
            ollama_url=self.config.extraction.ollama_url if self.config.extraction.enable_ollama_fallback else None,
            ollama_model=self.config.extraction.ollama_model,
            ollama_balanced_model=self.config.extraction.ollama_balanced_model,
            ollama_high_reasoning_model=self.config.extraction.ollama_high_reasoning_model,
            model_profile=self.config.extraction.model_profile,
            instructor_base_url=(
                self.config.extraction.instructor_base_url
                if self.config.extraction.enable_instructor
                else None
            ),
            instructor_model=self.config.extraction.instructor_model,
            instructor_api_key=self.config.extraction.instructor_api_key,
        )

        # Initialize Phase 2 engines (v3.2.0) — gated by feature flags
        flags = get_flags()
        self._otel = OTelGenAITracer(enabled=flags.is_enabled("otel_genai"))

        # Initialize ColBERT (Phase 6)
        if flags.is_enabled("colbert"):
            from muninn.retrieval.colbert_index import ColBERTIndexer
            self._colbert_indexer = ColBERTIndexer(
                vector_store=self._vectors,
                collection_name="muninn_colbert_tokens"
            )
            logger.info("ColBERT indexing enabled")
        else:
            self._colbert_indexer = None

        # Initialize hybrid retriever
        self._retriever = HybridRetriever(
            metadata_store=self._metadata,
            vector_store=self._vectors,
            graph_store=self._graph,
            bm25_index=self._bm25,
            reranker=self._reranker,
            colbert_indexer=self._colbert_indexer,
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
            colbert_indexer=self._colbert_indexer,
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

        # Initialize ingestion manager
        self._ingestion_manager = IngestionManager(self)

        # Phase 6: Advanced Features
        if self.config.advanced.enable_temporal_kg:
            self._temporal_kg = TemporalKnowledgeGraph(self._graph)
            self._temporal_kg.initialize_schema()
            logger.info("Temporal Knowledge Graph enabled")

        # Federation is always available (no heavy deps), just needs init
        self._federation = FederationManager(self)

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
        Add a new memory. Delegation to IngestionManager.
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
            # Process via IngestionManager
            processed = await self._ingestion_manager.process_add(
                content=content,
                user_id=user_id,
                agent_id=agent_id,
                metadata=metadata,
                namespace=namespace,
                memory_type=memory_type,
                provenance=provenance,
            )

            # Handle terminal early returns (DEDUP_SKIP, CONFLICT_SKIP)
            # Note: DEDUP_SIGNAL_UPDATE is handled below as it may fall through to ADD.
            if (
                processed.get("id") is None 
                and "event" in processed 
                and processed["event"] not in ("PROCESS_COMPLETE", "DEDUP_SIGNAL_UPDATE")
            ):
                return processed

            # Handle Dedup Update
            if processed.get("event") == "DEDUP_SIGNAL_UPDATE":
                dedup_result = processed["dedup"]
                embedding = processed["embedding"]
                record = processed["record"]
                
                merged_successfully = False
                async with self._write_lock:
                    existing = await asyncio.to_thread(self._metadata.get, dedup_result.existing_memory_id)
                    if existing and self._record_matches_scope(existing, namespace, user_id):
                        merged_content = self._dedup.merge_content(content, existing.content)
                        await asyncio.gather(
                            asyncio.to_thread(self._metadata.update, dedup_result.existing_memory_id, content=merged_content),
                            asyncio.to_thread(
                                self._vectors.upsert,
                                memory_id=dedup_result.existing_memory_id,
                                embedding=await self._embed(merged_content),
                                metadata={
                                    "content": merged_content[:500],
                                    "memory_type": existing.memory_type.value,
                                    "namespace": namespace,
                                    "importance": existing.importance,
                                    "user_id": user_id,
                                    "project": existing.project,
                                    "branch": existing.branch,
                                },
                            ),
                            asyncio.to_thread(self._bm25.add, dedup_result.existing_memory_id, merged_content)
                        )
                        merged_successfully = True
                
                if merged_successfully:
                    return {
                        "id": dedup_result.existing_memory_id,
                        "content": merged_content,
                        "event": "DEDUP_MERGED",
                        "dedup": dedup_result.model_dump(),
                    }
                # Fall through to normal persistence if scope mismatch

            # Handle Persistence for PROCESS_COMPLETE (or Dedup Fallback)
            record = processed["record"]
            extraction = processed["extraction"]
            embedding = processed["embedding"]
            entity_names = processed["entity_names"]
            conflict_info = processed["conflict_info"]

            # Acquire write lock only for the persistence phase
            async with self._write_lock:
                def _write_metadata():
                    self._metadata.add(record)

                def _write_vectors():
                    self._vectors.upsert(
                        memory_id=record.id,
                        embedding=embedding,
                        metadata={
                            "content": content[:500],
                            "memory_type": memory_type.value,
                            "namespace": namespace,
                            "importance": record.importance,
                            "user_id": user_id,
                            "project": record.project,
                            "branch": record.branch,
                        },
                    )

                def _write_graph():
                    self._graph.add_memory_node(
                        record.id, 
                        extraction.summary or content[:200],
                        user_id=user_id,
                        namespace=namespace
                    )
                    for entity in extraction.entities:
                        self._graph.add_entity(entity.name, entity.entity_type, user_id, namespace)
                        self._graph.link_memory_to_entity(record.id, entity.name, "mentions", user_id, namespace)
                    for relation in extraction.relations:
                        self._graph.add_entity(relation.subject, "concept", user_id, namespace)
                        self._graph.add_entity(relation.object, "concept", user_id, namespace)
                        self._graph.create_relation(
                            relation.subject,
                            relation.predicate,
                            relation.object,
                            record.id,
                            relation.confidence,
                            user_id=user_id,
                            namespace=namespace,
                        )

                def _write_bm25():
                    self._bm25.add(record.id, content)

                def _write_colbert():
                    if self._colbert_indexer:
                        self._colbert_indexer.index_text(record.id, content)

                await asyncio.gather(
                    asyncio.to_thread(_write_metadata),
                    asyncio.to_thread(_write_vectors),
                    asyncio.to_thread(_write_graph),
                    asyncio.to_thread(_write_bm25),
                    asyncio.to_thread(_write_colbert),
                )

                chain_links_created = await asyncio.to_thread(
                    self._upsert_memory_chain_links,
                    successor_record=record,
                    successor_content=content,
                    successor_entity_names=entity_names,
                )

            result = {
                "id": record.id,
                "content": content,
                "importance": record.importance,
                "memory_type": memory_type.value,
                "entities": [e.model_dump() for e in extraction.entities],
                "relations": [r.model_dump() for r in extraction.relations],
                "chain_links_created": chain_links_created,
                "event": "ADD",
            }
            if conflict_info:
                result["conflict"] = conflict_info
            
            if self._goal_compass is not None and record.project:
                drift = await self._goal_compass.evaluate_drift(
                    text=content,
                    user_id=user_id,
                    namespace=namespace,
                    project=record.project,
                )
                if drift is not None:
                    result["goal_alignment"] = drift

            self._otel.add_event(
                "muninn.add.result",
                {"memory_id": record.id, "importance": record.importance},
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
        project: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all memories, optionally filtered.

        Returns:
            List of memory dicts ordered by importance.
        """
        self._check_initialized()

        records = await asyncio.to_thread(
            self._metadata.get_all,
            limit=limit,
            namespace=namespace,
            user_id=user_id,
            project=project,
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
        async with self._write_lock:
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

    @classmethod
    def _merge_profile_patch(
        cls,
        base: Dict[str, Any],
        patch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Recursively merge profile patch data into an existing profile object.

        Dict values are deep-merged; non-dict values replace the target key.
        """
        merged = dict(base)
        for key, value in patch.items():
            if (
                isinstance(value, dict)
                and isinstance(merged.get(key), dict)
            ):
                merged[key] = cls._merge_profile_patch(
                    merged[key],  # type: ignore[arg-type]
                    value,
                )
            else:
                merged[key] = value
        return merged

    async def set_user_profile(
        self,
        *,
        profile: Dict[str, Any],
        user_id: str = "global_user",
        merge: bool = True,
        source: str = "runtime_api",
    ) -> Dict[str, Any]:
        """Set/update editable user profile and global context data."""
        self._check_initialized()
        if not isinstance(profile, dict):
            raise ValueError("profile must be a JSON object")

        async with self._write_lock:
            existing = self._metadata.get_user_profile(user_id=user_id)
            current_profile = (
                dict(existing.get("profile", {}))
                if existing and isinstance(existing.get("profile"), dict)
                else {}
            )
            next_profile = (
                self._merge_profile_patch(current_profile, profile)
                if merge
                else dict(profile)
            )

            self._metadata.set_user_profile(
                user_id=user_id,
                profile=next_profile,
                source=source,
            )
        stored = self._metadata.get_user_profile(user_id=user_id)
        return {
            "event": "USER_PROFILE_UPDATED",
            "user_id": user_id,
            "merge": bool(merge),
            "profile": stored.get("profile", {}) if stored else {},
            "source": stored.get("source") if stored else source,
            "updated_at": stored.get("updated_at") if stored else None,
        }

    async def get_user_profile(
        self,
        *,
        user_id: str = "global_user",
    ) -> Dict[str, Any]:
        """Fetch editable user profile and global context data for a user."""
        self._check_initialized()
        profile = self._metadata.get_user_profile(user_id=user_id)
        if profile is None:
            return {
                "event": "USER_PROFILE_EMPTY",
                "user_id": user_id,
                "profile": {},
                "source": None,
                "updated_at": None,
            }
        return {
            "event": "USER_PROFILE_LOADED",
            "user_id": user_id,
            "profile": profile.get("profile", {}),
            "source": profile.get("source"),
            "updated_at": profile.get("updated_at"),
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
        
        async with self._write_lock:
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

        async with self._write_lock:
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

        # Use a semaphore to bound concurrency for chunk processing (extractions/embeddings).
        # This prevents overwhelming local LLM endpoints or thread pools.
        semaphore = asyncio.Semaphore(8)

        async def _add_chunk_task(chunk, source_context, record_ref):
            nonlocal added_memories, skipped_chunks, failed_chunks
            chunk_metadata = dict(base_metadata)
            chunk_metadata.update(source_context)
            chunk_metadata.update(chunk.metadata)
            
            async with semaphore:
                try:
                    add_result = await self.add(
                        content=chunk.content,
                        user_id=user_id,
                        namespace=namespace,
                        metadata=chunk_metadata,
                        provenance=Provenance.INGESTED,
                    )
                    
                    if add_result.get("event") in {"DEDUP_SKIP", "CONFLICT_SKIP"}:
                        skipped_chunks += 1
                        record_ref["chunks_skipped"] += 1
                    else:
                        added_memories += 1
                        record_ref["chunks_added"] += 1
                except Exception as exc:
                    failed_chunks += 1
                    record_ref["chunks_failed"] += 1
                    record_ref["errors"].append(
                        f"chunk[{chunk.chunk_index}] add failed: {exc}"
                    )

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
            source_payloads.append(source_record)
            
            if source_result.status != "processed":
                continue

            source_context = source_context_by_path.get(source_result.source_path, {})
            
            # Create tasks for all chunks in this source
            tasks = [
                _add_chunk_task(chunk, source_context, source_record)
                for chunk in source_result.chunks
            ]
            if tasks:
                await asyncio.gather(*tasks)

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

        report = await asyncio.to_thread(
            ingestion.get_report if hasattr(ingestion, 'get_report') else ingestion.ingest,
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
        base_metadata.setdefault(
            "operator_model_profile",
            self.config.extraction.ingestion_model_profile,
        )
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
                resolved_path = Path(norm)
                source_type = infer_source_type(Path(norm))
                parser_supported = source_type != "unsupported"
                size_bytes = 0
                modified_at_epoch = None
                modified_at_iso = None
                try:
                    stat = resolved_path.stat()
                    size_bytes = stat.st_size
                    modified_at_epoch = float(stat.st_mtime)
                    modified_at_iso = time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ",
                        time.gmtime(modified_at_epoch),
                    )
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
                    "parent_path": str(resolved_path.parent),
                    "path_depth": len(resolved_path.parts),
                    "modified_at_epoch": modified_at_epoch,
                    "modified_at_iso": modified_at_iso,
                    "relative_path_hint": resolved_path.name,
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
                "legacy_source_parent_path": item.get("parent_path"),
                "legacy_source_path_depth": item.get("path_depth"),
                "legacy_source_relative_path": item.get("relative_path_hint"),
                "legacy_source_modified_at_epoch": item.get("modified_at_epoch"),
                "legacy_source_modified_at_iso": item.get("modified_at_iso"),
                "legacy_contextualization_mode": "chronological_hierarchy",
            }

        base_metadata = dict(metadata or {})
        base_metadata.setdefault("project", project)
        base_metadata.setdefault("kind", "legacy_ingested_source_chunk")
        base_metadata.setdefault(
            "operator_model_profile",
            self.config.extraction.legacy_ingestion_model_profile,
        )
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

        record = await asyncio.to_thread(self._metadata.get, memory_id)
        if not record:
            return {"error": f"Memory {memory_id} not found"}

        old_content = record.content
        record.content = data

        # Re-extract entities
        extraction = await self._extract_with_profile(
            data,
            model_profile=self.config.extraction.runtime_model_profile,
        )
        entity_names = self._extract_entity_names(extraction)
        updated_metadata = dict(record.metadata or {})
        if entity_names:
            updated_metadata["entity_names"] = entity_names
        else:
            updated_metadata.pop("entity_names", None)
        record.metadata = updated_metadata

        # Re-embed
        embedding = await self._embed(data)

        # Update stores
        async with self._write_lock:
            def _update_metadata():
                self._metadata.update(record.id, content=record.content, metadata=record.metadata)

            def _update_vectors():
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

            def _update_graph():
                uid = record.metadata.get("user_id", "global")
                ns = record.namespace
                self._graph.delete_memory_references(record.id)
                self._graph.add_memory_node(
                    record.id, extraction.summary or data[:200],
                    user_id=uid, namespace=ns,
                )
                for entity in extraction.entities:
                    self._graph.add_entity(entity.name, entity.entity_type, uid, ns)
                    self._graph.link_memory_to_entity(record.id, entity.name, "mentions", uid, ns)
                for relation in extraction.relations:
                    self._graph.add_entity(relation.subject, "concept", uid, ns)
                    self._graph.add_entity(relation.object, "concept", uid, ns)
                    self._graph.create_relation(
                        relation.subject,
                        relation.predicate,
                        relation.object,
                        record.id,
                        relation.confidence,
                        user_id=uid,
                        namespace=ns,
                    )

            def _update_bm25():
                self._bm25.add(record.id, data)

            await asyncio.gather(
                asyncio.to_thread(_update_metadata),
                asyncio.to_thread(_update_vectors),
                asyncio.to_thread(_update_graph),
                asyncio.to_thread(_update_bm25),
            )

            chain_links_created = await asyncio.to_thread(
                self._upsert_memory_chain_links,
                successor_record=record,
                successor_content=data,
                successor_entity_names=entity_names,
            )

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

        async with self._write_lock:
            await asyncio.gather(
                asyncio.to_thread(self._metadata.delete, memory_id),
                asyncio.to_thread(self._vectors.delete, memory_id),
                asyncio.to_thread(self._graph.delete_memory_references, memory_id),
                asyncio.to_thread(self._bm25.remove, memory_id),
            )

        logger.info("Deleted memory %s", memory_id)
        return {"id": memory_id, "event": "DELETE"}

    async def delete_all(self, user_id: str = "global_user") -> Dict[str, Any]:
        """Delete all memories for a user.

        Scoped by ``user_id`` so that one tenant cannot wipe another's data.
        Individual vectors and BM25 entries are removed per-record to keep
        the stores consistent without a full collection nuke.
        """
        self._check_initialized()

        async with self._write_lock:
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

        def _get_graph_node_count():
            return len(self._graph.get_all_entities())

        (
            memory_count,
            vector_count,
            graph_nodes,
        ) = await asyncio.gather(
            asyncio.to_thread(self._metadata.count),
            asyncio.to_thread(self._vectors.count),
            asyncio.to_thread(_get_graph_node_count),
        )

        return {
            "status": "ok",
            "memory_count": memory_count,
            "vector_count": vector_count,
            "graph_nodes": graph_nodes,
            "bm25_size": self._bm25.size,
            "reranker": "active" if (self._reranker and self._reranker.is_available) else "inactive",
            "consolidation": self._consolidation.status if self._consolidation else {"running": False},
            "backend": "muninn-native",
        }

    async def get_model_profiles(self) -> Dict[str, Any]:
        """Return active extraction profile policy and supported profile values."""
        self._check_initialized()

        extraction = self.config.extraction
        return {
            "supported_profiles": list(SUPPORTED_MODEL_PROFILES),
            "active": {
                "model_profile": extraction.model_profile,
                "runtime_model_profile": extraction.runtime_model_profile,
                "ingestion_model_profile": extraction.ingestion_model_profile,
                "legacy_ingestion_model_profile": extraction.legacy_ingestion_model_profile,
            },
            "models": {
                "low_latency": extraction.ollama_model,
                "balanced": extraction.ollama_balanced_model,
                "high_reasoning": extraction.ollama_high_reasoning_model,
            },
            "vram_budget_gb": extraction.vram_budget_gb,
        }

    async def set_model_profiles(
        self,
        *,
        model_profile: Optional[str] = None,
        runtime_model_profile: Optional[str] = None,
        ingestion_model_profile: Optional[str] = None,
        legacy_ingestion_model_profile: Optional[str] = None,
        source: str = "runtime_api",
    ) -> Dict[str, Any]:
        """
        Update active extraction profile policy at runtime.

        Notes:
        - Values must be one of SUPPORTED_MODEL_PROFILES.
        - This mutates in-memory policy for the current server process.
        """
        self._check_initialized()

        extraction = self.config.extraction
        requested = {
            "model_profile": model_profile,
            "runtime_model_profile": runtime_model_profile,
            "ingestion_model_profile": ingestion_model_profile,
            "legacy_ingestion_model_profile": legacy_ingestion_model_profile,
        }
        updates: Dict[str, Dict[str, str]] = {}

        for field_name, raw_value in requested.items():
            if raw_value is None:
                continue
            candidate = raw_value.strip()
            if candidate not in SUPPORTED_MODEL_PROFILES:
                raise ValueError(
                    f"Unsupported {field_name} '{raw_value}'. "
                    f"Expected one of {SUPPORTED_MODEL_PROFILES}."
                )

            current = str(getattr(extraction, field_name))
            if current != candidate:
                setattr(extraction, field_name, candidate)
                updates[field_name] = {"from": current, "to": candidate}

        # Keep extraction pipeline default route aligned with configured base profile.
        if self._extraction is not None:
            self._extraction.model_profile = extraction.model_profile

        event = "MODEL_PROFILE_POLICY_UPDATED" if updates else "MODEL_PROFILE_POLICY_UNCHANGED"
        policy = await self.get_model_profiles()
        audit_event_id: Optional[int] = None
        if updates:
            async with self._write_lock:
                audit_event_id = self._metadata.record_profile_policy_event(
                    source=source,
                    updates=updates,
                    policy=policy,
                )
        return {
            "event": event,
            "updates": updates,
            "policy": policy,
            "audit_event_id": audit_event_id,
        }

    async def get_model_profile_events(self, *, limit: int = 25) -> Dict[str, Any]:
        """Return recent runtime profile-policy mutation events."""
        self._check_initialized()
        events = self._metadata.get_profile_policy_events(limit=limit)
        return {
            "event": "MODEL_PROFILE_EVENTS",
            "events": events,
            "count": len(events),
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

    async def _embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if self._embed_model is not None:
            # fastembed is CPU-bound, run in thread
            def _run_fastembed():
                embeddings = list(self._embed_model.embed([text]))
                return embeddings[0].tolist()
            return await asyncio.to_thread(_run_fastembed)
        else:
            # Ollama fallback (already using httpx but wrapped in sync func, let's offload it or make it async if possible)
            # _ollama_embed uses httpx.post synchronously.
            return await asyncio.to_thread(self._ollama_embed, text)

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

    async def _extract(self, content: str, model_profile: Optional[str] = None) -> ExtractionResult:
        """Run extraction pipeline on content."""
        if self._extraction:
            return await asyncio.to_thread(
                self._extraction.extract,
                content,
                model_profile=model_profile,
            )
        return ExtractionResult()

    async def _extract_with_profile(
        self,
        content: str,
        model_profile: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Invoke extraction with graceful backward compatibility for tests/mocks.

        Some tests monkeypatch `self._extract` with a legacy one-argument
        coroutine. If keyword profile routing is unsupported by that stub,
        retry without `model_profile`.
        """
        try:
            return await self._extract(content, model_profile=model_profile)
        except TypeError as e:
            if "model_profile" not in str(e):
                raise
            return await self._extract(content)

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

    async def get_temporal_knowledge(
        self,
        timestamp: Optional[float] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Query the Temporal Knowledge Graph for facts valid at a specific time."""
        self._check_initialized()
        if not self._temporal_kg:
            return []
        
        ts = float(timestamp) if timestamp is not None else time.time()
        # This is a read operation, usually fast, but we can offload if needed.
        # Kuzu reads are blocking, so offload to thread.
        return await asyncio.to_thread(self._temporal_kg.query_valid_at, ts, limit)

    async def get_federation_manager(self) -> FederationManager:
        self._check_initialized()
        return self._federation