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
import uuid
import time
import logging
from typing import List, Optional, Dict, Any
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

        # Phase 2 engines (v3.2.0)
        self._dedup = None           # SemanticDedup
        self._conflict_detector = None  # ConflictDetector
        self._conflict_resolver = None  # ConflictResolver

        # Embedding
        self._embed_model = None
        self._initialized = False

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
        )

        # Initialize hybrid retriever
        self._retriever = HybridRetriever(
            metadata_store=self._metadata,
            vector_store=self._vectors,
            graph_store=self._graph,
            bm25_index=self._bm25,
            reranker=self._reranker,
            embed_fn=self._embed,
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
                    )
                    logger.info("Conflict detection enabled (model=%s)", self.config.conflict_detection.model_name)
                else:
                    logger.info("Conflict detection flag ON but NLI model unavailable (install transformers+torch)")
                    self._conflict_detector = None
            except Exception as e:
                logger.warning("Conflict detection initialization failed: %s", e)
                self._conflict_detector = None
                self._conflict_resolver = None

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

    # ==========================================
    # Core Memory Operations (mem0-compatible API)
    # ==========================================

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

        # Extract entities and relations
        extraction = await self._extract(content)

        # Generate embedding
        embedding = self._embed(content)

        # --- Phase 2: Semantic Deduplication (v3.2.0) ---
        dedup_result = None
        if self._dedup and self._vectors.count() > 0:
            from muninn.dedup.semantic_dedup import DedupStrategy
            dedup_result = self._dedup.check_duplicate(
                embedding=embedding,
                content=content,
                vector_store=self._vectors,
                metadata_store=self._metadata,
            )
            if dedup_result and dedup_result.is_duplicate:
                if dedup_result.strategy == DedupStrategy.SKIP:
                    logger.info("Dedup SKIP: duplicate of %s (sim=%.3f)",
                                dedup_result.existing_memory_id, dedup_result.similarity)
                    return {
                        "id": None,
                        "content": content,
                        "event": "DEDUP_SKIP",
                        "dedup": dedup_result.model_dump(),
                    }
                elif dedup_result.strategy == DedupStrategy.UPDATE_EXISTING:
                    merged_content = self._dedup.merge_content(content, "")
                    existing = self._metadata.get(dedup_result.existing_memory_id)
                    if existing:
                        merged_content = self._dedup.merge_content(content, existing.content)
                        self._metadata.update(
                            dedup_result.existing_memory_id,
                            content=merged_content,
                        )
                        # Re-embed merged content
                        merged_embedding = self._embed(merged_content)
                        self._vectors.upsert(
                            memory_id=dedup_result.existing_memory_id,
                            embedding=merged_embedding,
                            metadata={
                                "content": merged_content[:500],
                                "memory_type": existing.memory_type.value,
                                "namespace": existing.namespace,
                                "importance": existing.importance,
                            },
                        )
                        self._bm25.add(dedup_result.existing_memory_id, merged_content)
                        logger.info("Dedup UPDATE_EXISTING: merged into %s",
                                    dedup_result.existing_memory_id)
                        return {
                            "id": dedup_result.existing_memory_id,
                            "content": merged_content,
                            "event": "DEDUP_MERGED",
                            "dedup": dedup_result.model_dump(),
                        }

        # --- Phase 2: Conflict Detection (v3.2.0) ---
        conflict_info = None
        if self._conflict_detector and self._vectors.count() > 0:
            try:
                # Pre-filter by vector similarity to find candidates
                similar_for_conflict = self._vectors.search(
                    embedding,
                    limit=5,
                    score_threshold=self.config.conflict_detection.similarity_prefilter,
                )
                if similar_for_conflict:
                    candidate_ids = [mid for mid, _score in similar_for_conflict]
                    candidate_records = self._metadata.get_by_ids(candidate_ids)
                    if candidate_records:
                        conflicts = self._conflict_detector.detect_conflicts(content, candidate_records)
                        if conflicts and self._conflict_resolver:
                            # Resolve the first (highest-scoring) conflict
                            conflict = conflicts[0]
                            resolution = self._conflict_resolver.resolve(conflict)
                            conflict_info = {
                                "conflict_detected": True,
                                "resolution": resolution,
                                "conflict_details": conflict.model_dump(),
                            }
                            # If resolution says skip new storage, return early
                            if resolution.get("skip_new_storage"):
                                logger.info("Conflict resolution: %s — skipping new storage",
                                            resolution.get("resolution"))
                                return {
                                    "id": None,
                                    "content": content,
                                    "event": f"CONFLICT_{resolution['resolution'].upper()}",
                                    "conflict": conflict_info,
                                }
            except Exception as e:
                logger.warning("Conflict detection failed (non-fatal): %s", e)

        # Calculate novelty by checking similarity to existing memories
        max_similarity = 0.0
        if self._vectors.count() > 0:
            try:
                similar = self._vectors.search(embedding, limit=5)
                if similar:
                    max_similarity = similar[0][1]  # Highest score
            except Exception:
                pass

        # Calculate initial importance
        record = MemoryRecord(
            content=content,
            memory_type=memory_type,
            provenance=provenance,
            source_agent=agent_id or "unknown",
            namespace=namespace,
            metadata=metadata or {},
            novelty_score=calculate_novelty(max_similarity),
        )

        # Get centrality from graph (if entities exist)
        centrality = 0.0
        if extraction.entities:
            # We'll compute centrality after adding to graph
            centrality = 0.1  # Baseline for having entities

        importance = calculate_importance(
            record,
            max_similarity=max_similarity,
            centrality=centrality,
        )
        record.importance = importance

        # Store in metadata (SQLite)
        self._metadata.add(record)

        # Store embedding in vector store (Qdrant)
        self._vectors.upsert(
            memory_id=record.id,
            embedding=embedding,
            metadata={
                "content": content[:500],
                "memory_type": memory_type.value,
                "namespace": namespace,
                "importance": importance,
                "user_id": user_id,
            },
        )

        # Store in graph (Kuzu)
        self._graph.add_memory_node(record.id, extraction.summary or content[:200])
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

        # Add to BM25 index
        self._bm25.add(record.id, content)

        logger.info("Added memory %s (importance=%.3f, entities=%d, relations=%d)",
                     record.id, importance, len(extraction.entities), len(extraction.relations))

        result = {
            "id": record.id,
            "content": content,
            "importance": importance,
            "memory_type": memory_type.value,
            "entities": [e.model_dump() for e in extraction.entities],
            "relations": [r.model_dump() for r in extraction.relations],
            "event": "ADD",
        }

        # Attach Phase 2 metadata when present
        if conflict_info:
            result["conflict"] = conflict_info
        if dedup_result:
            result["dedup"] = dedup_result.model_dump()

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

        results = await self._retriever.search(
            query=query,
            limit=limit,
            user_id=user_id,
            filters=filters,
            rerank=rerank,
            namespaces=namespaces,
            explain=explain,
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
            output.append(item)
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

        # Re-embed
        embedding = self._embed(data)

        # Update stores
        self._metadata.update(record)
        self._vectors.upsert(
            memory_id=record.id,
            embedding=embedding,
            metadata={
                "content": data[:500],
                "memory_type": record.memory_type.value,
                "namespace": record.namespace,
                "importance": record.importance,
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

        # Update BM25
        self._bm25.add(record.id, data)

        logger.info("Updated memory %s", memory_id)

        return {
            "id": record.id,
            "content": data,
            "previous_content": old_content,
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

    async def delete_all(self, user_id: str = "global_user") -> Dict[str, str]:
        """Delete all memories."""
        self._check_initialized()

        self._metadata.delete_all()
        self._vectors.delete_all()
        self._bm25.clear()

        logger.info("Deleted all memories for user %s", user_id)
        return {"event": "DELETE_ALL", "user_id": user_id}

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
