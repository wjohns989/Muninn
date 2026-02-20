"""
Muninn Consolidation Daemon
----------------------------
Background consolidation process inspired by neuroscience sleep consolidation
and biological immune system memory maturation.

Runs periodically (default: every 6 hours) through 5 phases:
1. DECAY    — Recalculate importance, soft-delete low-value memories
2. MERGE    — Find and merge near-duplicate episodic memories
3. PROMOTE  — Promote frequently-accessed memories to higher types
4. REPLAY   — Re-embed high-importance memories with latest model
5. STATISTICS — Update system-wide metrics and health
"""

import asyncio
import time
import logging
from typing import Optional

from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException

from muninn.core.config import ConsolidationConfig
from muninn.store.sqlite_metadata import SQLiteMetadataStore
from muninn.store.vector_store import VectorStore
from muninn.store.graph_store import GraphStore
from muninn.retrieval.bm25 import BM25Index
from muninn.scoring.importance import calculate_importance, batch_update_importance
from muninn.consolidation.merge import find_merge_candidates, merge_memories
from muninn.consolidation.promote import find_promotion_candidates, promote_memory
from muninn.core.types import MemoryType

logger = logging.getLogger("Muninn.Consolidation")


class ConsolidationDaemon:
    """
    Background daemon that runs periodic consolidation cycles.

    Each cycle processes through 5 phases to maintain memory health:
    decay → merge → promote → replay → statistics.
    """

    def __init__(
        self,
        config: ConsolidationConfig,
        metadata: SQLiteMetadataStore,
        vectors: VectorStore,
        graph: GraphStore,
        bm25: BM25Index,
        embed_fn=None,
        colbert_indexer=None,
    ):
        self.config = config
        self.metadata = metadata
        self.vectors = vectors
        self.graph = graph
        self.bm25 = bm25
        self._embed_fn = embed_fn
        self.colbert_indexer = colbert_indexer
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_cycle: Optional[float] = None
        self._cycle_count = 0
        
        # Phase 9 integrity components (v3.6.0)
        from muninn.conflict.detector import ConflictDetector
        from muninn.conflict.resolver import ConflictResolver
        self._conflict_detector = ConflictDetector(
            contradiction_threshold=self.config.integrity_contradiction_threshold
        )
        self._conflict_resolver = ConflictResolver(
            metadata_store=self.metadata,
            vector_store=self.vectors,
            graph_store=self.graph,
            bm25_index=self.bm25,
            embed_fn=self._embed_fn
        )

    async def start(self) -> None:
        """Start the consolidation daemon."""
        if not self.config.enabled:
            logger.info("Consolidation daemon disabled")
            return

        if self._running:
            logger.warning("Consolidation daemon already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Consolidation daemon started (interval=%.1fh)", self.config.interval_hours)

    async def stop(self) -> None:
        """Stop the consolidation daemon."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Consolidation daemon stopped")

    async def run_cycle(self) -> dict:
        """
        Execute a single consolidation cycle (all 5 phases).

        Returns:
            Dict with phase results and timing.
        """
        t0 = time.time()
        self._cycle_count += 1
        logger.info("=== Consolidation cycle #%d starting ===", self._cycle_count)

        results = {
            "cycle": self._cycle_count,
            "phases": {},
        }

        try:
            # Phase 1: DECAY
            decay_result = await self._phase_decay()
            results["phases"]["decay"] = decay_result

            # Phase 2: MERGE
            merge_result = await self._phase_merge()
            results["phases"]["merge"] = merge_result

            # Phase 3: PROMOTE
            promote_result = await self._phase_promote()
            results["phases"]["promote"] = promote_result

            # Phase 4: REPLAY
            replay_result = await self._phase_replay()
            results["phases"]["replay"] = replay_result

            # Phase 5: STATISTICS
            stats_result = await self._phase_statistics()
            results["phases"]["statistics"] = stats_result

            # Phase 6: MAINTENANCE (ColBERT)
            maintenance_result = await self._phase_maintenance()
            results["phases"]["maintenance"] = maintenance_result

            # Phase 7: OPTIMIZATION (Storage)
            optimization_result = await self._phase_optimization()
            results["phases"]["optimization"] = optimization_result

            # Phase 8: INTEGRITY (NLI Conflict Detection)
            integrity_result = await self._phase_integrity()
            results["phases"]["integrity"] = integrity_result

        except Exception as e:
            logger.error("Consolidation cycle failed: %s", e, exc_info=True)
            results["error"] = str(e)

        elapsed = time.time() - t0
        results["elapsed_seconds"] = round(elapsed, 2)
        self._last_cycle = time.time()

        logger.info("=== Consolidation cycle #%d completed in %.2fs ===",
                     self._cycle_count, elapsed)
        return results

    async def _run_loop(self) -> None:
        """Main daemon loop."""
        interval_seconds = self.config.interval_hours * 3600

        while self._running:
            try:
                await self.run_cycle()
            except Exception as e:
                logger.error("Consolidation cycle error: %s", e, exc_info=True)

            # Sleep until next cycle
            try:
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break

    # --- Phase Implementations ---

    async def _phase_decay(self) -> dict:
        """
        Phase 1: DECAY
        - Recalculate importance for all memories
        - Soft-delete memories below threshold
        - Expire working memories past TTL
        """
        t0 = time.time()
        records = self.metadata.get_for_consolidation(limit=500)
        decayed = 0
        expired = 0
        updated = 0

        # Batch fetch centrality for all records (P1 Performance Optimization)
        record_ids = [r.id for r in records]
        centrality_map = self.graph.get_memory_node_degrees_batch(record_ids)

        for record in records:
            # Get centrality from pre-fetched map
            centrality = centrality_map.get(record.id, 0.0)

            # Get max similarity for novelty calculation
            max_sim = 0.0  # Would need vector lookup — simplified for now

            # Recalculate importance
            new_importance = calculate_importance(
                record,
                max_similarity=max_sim,
                centrality=centrality,
            )

            if new_importance != record.importance:
                record.importance = new_importance
                self.metadata.update(record)
                updated += 1

            # Soft-delete below threshold
            if new_importance < self.config.decay_threshold:
                self.metadata.delete(record.id)
                self.vectors.delete([record.id])
                self.graph.delete_memory_references(record.id)
                self.bm25.remove(record.id)
                decayed += 1

            # Expire working memories past TTL
            if record.memory_type == MemoryType.WORKING:
                ttl_seconds = self.config.working_memory_ttl_hours * 3600
                if (time.time() - record.created_at) > ttl_seconds:
                    self.metadata.delete(record.id)
                    self.vectors.delete([record.id])
                    expired += 1

        elapsed = time.time() - t0
        result = {"updated": updated, "decayed": decayed, "expired": expired,
                  "elapsed": round(elapsed, 2)}
        logger.info("Phase DECAY: %s", result)
        return result

    async def _phase_merge(self) -> dict:
        """
        Phase 2: MERGE
        - Find near-duplicate episodic memories
        - Merge content and metadata
        - Remove absorbed duplicates
        """
        t0 = time.time()
        # Process in batches to maintain isolation context
        records = self.metadata.get_for_consolidation(limit=500)
        episodic = [r for r in records if r.memory_type == MemoryType.EPISODIC]

        if not episodic:
            return {"merged": 0, "elapsed": 0.0}

        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        # Find candidates using vector similarity
        def vector_search_fn(vector_id, user_id=None, namespace=None):
            try:
                # v3.8.0 Isolation: We need the actual vector to search
                # But since merge happens in background, we rely on find_merge_candidates
                # to respect the passed records. 
                # Optimization: We use the daemon's vector search with strict filters.
                vec = self.vectors.get_vectors([vector_id]).get(vector_id)
                if not vec:
                    return []
                
                must_conditions = []
                if user_id:
                    must_conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
                if namespace:
                    must_conditions.append(FieldCondition(key="namespace", match=MatchValue(value=namespace)))
                
                search_filter = Filter(must=must_conditions) if must_conditions else None
                
                results = self.vectors.search(
                    query_embedding=vec,
                    limit=5,
                    filters=search_filter
                )
                return results
            except Exception as e:
                logger.debug(f"Merge vector search failed: {e}")
                return []

        # Find merge candidates across the batch
        candidates = find_merge_candidates(episodic, vector_search_fn)
        merged_count = 0

        for primary_id, secondary_id, _ in candidates:
            primary = self.metadata.get(primary_id)
            secondary = self.metadata.get(secondary_id)
            
            if not primary or not secondary:
                continue
            
            # CRITICAL SAFETY check: Never merge across namespaces
            primary_uid = (primary.metadata or {}).get("user_id")
            secondary_uid = (secondary.metadata or {}).get("user_id")
            if primary_uid != secondary_uid or primary.namespace != secondary.namespace:
                logger.warning(
                    "BLOCKED cross-namespace merge attempt: %s (%s/%s) vs %s (%s/%s)",
                    primary.id, primary_uid, primary.namespace,
                    secondary.id, secondary_uid, secondary.namespace
                )
                continue

            merged = merge_memories(primary, secondary)

            # Update stores
            self.metadata.update(merged)
            self.metadata.delete(secondary.id)
            self.vectors.delete([secondary.id])
            self.graph.delete_memory_references(secondary.id)
            # Graph update: primary node summary changes
            merged_uid = (merged.metadata or {}).get("user_id")
            self.graph.add_memory_node(
                merged.id,
                merged.content[:500],
                user_id=merged_uid,
                namespace=merged.namespace
            )
            
            merged_count += 1

        elapsed = time.time() - t0
        result = {"merged": merged_count, "candidates_checked": len(episodic),
                  "elapsed": round(elapsed, 2)}
        logger.info("Phase MERGE: %s", result)
        return result

    async def _phase_promote(self) -> dict:
        """
        Phase 3: PROMOTE
        - Find memories eligible for type promotion
        - Promote episodic → semantic → procedural
        """
        t0 = time.time()
        records = self.metadata.get_for_consolidation(limit=500)

        candidates = find_promotion_candidates(records)
        promoted = 0

        for mem_id, new_type in candidates:
            record = self.metadata.get(mem_id)
            if record:
                updated = promote_memory(record, new_type)
                self.metadata.update(updated)
                promoted += 1

        elapsed = time.time() - t0
        result = {"promoted": promoted, "elapsed": round(elapsed, 2)}
        logger.info("Phase PROMOTE: %s", result)
        return result

    async def _phase_replay(self) -> dict:
        """
        Phase 4: REPLAY (Hippocampal Replay)
        - Re-embed high-importance memories that haven't been re-embedded recently
        - Ensures embedding drift doesn't degrade retrieval quality

        This is a lightweight phase — only processes top-K important memories
        per cycle to avoid overloading the embedding service.
        """
        t0 = time.time()
        re_embedded = 0

        if self._embed_fn is None:
            return {"re_embedded": 0, "reason": "no_embed_fn", "elapsed": 0.0}

        # Get high-importance memories for replay
        records = self.metadata.get_for_consolidation(limit=100)
        high_importance = [r for r in records if r.importance > 0.7][:20]

        for record in high_importance:
            try:
                # Re-embed the content
                embedding = self._embed_fn(record.content)
                if hasattr(embedding, "__await__"):
                    embedding = await embedding

                # Upsert to vector store
                self.vectors.upsert(
                    doc_id=record.id,
                    vector=embedding,
                    payload={
                        "content": record.content[:500],
                        "memory_type": record.memory_type.value,
                        "namespace": record.namespace,
                        "importance": record.importance,
                    },
                )
                re_embedded += 1
            except Exception as e:
                logger.warning("Replay re-embed failed for %s: %s", record.id, e)

        elapsed = time.time() - t0
        result = {"re_embedded": re_embedded, "elapsed": round(elapsed, 2)}
        logger.info("Phase REPLAY: %s", result)
        return result

    async def _phase_statistics(self) -> dict:
        """
        Phase 5: STATISTICS
        - Count memories by type
        - Calculate average importance
        - Report system health metrics
        """
        t0 = time.time()

        total = self.metadata.count()
        vector_count = self.vectors.count()
        entity_count = len(self.graph.get_all_entities())

        stats = {
            "total_memories": total,
            "vector_count": vector_count,
            "entity_count": entity_count,
            "cycle_number": self._cycle_count,
            "elapsed": round(time.time() - t0, 2),
        }

        logger.info("Phase STATISTICS: %s", stats)
        return stats

    async def _phase_maintenance(self) -> dict:
        """
        Phase 6: MAINTENANCE (ColBERT)
        - Monitor centroid drift
        - Trigger re-clustering if threshold exceeded
        """
        if self.colbert_indexer is None:
            return {"status": "skipped", "reason": "no_colbert_indexer", "elapsed": 0.0}
            
        t0 = time.time()
        drift = 0.0
        re_clustered = False
        
        try:
            # Sample from the ColBERT token collection — centroid relevance must
            # be measured in token-embedding space, not in main memory embedding space.
            sample_size = 2000
            client = self.colbert_indexer.vectors._get_client()
            token_collection = self.colbert_indexer.collection_name

            # Check if the ColBERT token collection has enough points to warrant drift check
            try:
                colbert_count = client.count(collection_name=token_collection).count
            except (UnexpectedResponse, ResponseHandlingException):
                # Collection does not exist yet (HTTP 404) or qdrant is temporarily
                # unavailable — skip drift check silently; it is not a fatal error.
                colbert_count = 0
            if colbert_count < 100:
                return {"status": "skipped", "reason": "too_few_colbert_tokens", "count": colbert_count}

            scroll_result = client.scroll(
                collection_name=token_collection,
                limit=sample_size,
                with_vectors=True
            )
            points = scroll_result[0] if scroll_result else []
            
            if len(points) >= 100:
                import numpy as np
                sample_vectors = np.array([p.vector for p in points if p.vector is not None]).astype(np.float32)
                
                if len(sample_vectors) > 0:
                    drift = self.colbert_indexer.check_centroid_relevance(sample_vectors)
                    logger.info("Phase MAINTENANCE: ColBERT drift detected: %.4f", drift)
                    
                    if drift > self.config.colbert_drift_threshold:
                        logger.warning("Phase MAINTENANCE: Drift %.4f exceeds threshold %.4f. Re-clustering...", 
                                       drift, self.config.colbert_drift_threshold)
                        re_clustered = self.colbert_indexer.recluster_centroids(sample_vectors)
        except Exception as e:
            logger.error("Phase MAINTENANCE: ColBERT maintenance failed: %s", e)

        elapsed = time.time() - t0
        result = {"drift": round(drift, 4), "re-clustered": re_clustered, "elapsed": round(elapsed, 2)}
        logger.info("Phase MAINTENANCE: %s", result)
        return result

    async def _phase_optimization(self) -> dict:
        """
        Phase 7: OPTIMIZATION (Storage)
        - Tune collection settings based on size
        - Enable/update Scalar Quantization
        """
        t0 = time.time()
        optimized = False
        
        try:
            count = self.vectors.count()
            if count >= self.config.quantization_threshold_points:
                # v3.6.1 Refinement: Skip if quantization already active
                client = self.vectors._get_client()
                collection_info = client.get_collection(self.vectors.collection_name)
                if collection_info.config.quantization_config:
                    return {"status": "skipped", "reason": "already_quantized", "elapsed": 0.0}

                from qdrant_client import models
                logger.info("Phase OPTIMIZATION: Collection size (%d) qualifies for Scalar Quantization tuning.", count)
                optimized = await self.vectors.update_collection_quantization(
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True
                        )
                    )
                )
        except Exception as e:
            logger.error("Phase OPTIMIZATION: Storage optimization failed: %s", e)

        elapsed = time.time() - t0
        result = {"optimized": optimized, "elapsed": round(elapsed, 2)}
        logger.info("Phase OPTIMIZATION: %s", result)
        return result

    async def _phase_integrity(self) -> dict:
        """
        Phase 8: INTEGRITY (NLI Conflict Detection)
        - Audit high-importance recent memories for contradictions
        - Resolve detected conflicts automatically
        """
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        
        if not self._conflict_detector.is_available:
            return {"status": "skipped", "reason": "nli_model_unavailable", "elapsed": 0.0}
            
        t0 = time.time()
        audited = 0
        conflicts_resolved = 0
        
        try:
            # Audit top-K important recent memories
            records = self.metadata.get_for_consolidation(limit=50)
            if not records:
                return {"audited": 0, "conflicts_resolved": 0, "elapsed": round(time.time() - t0, 2)}

            # v3.6.2 Phase Optimization: Batch vector and metadata retrieval
            record_ids = [r.id for r in records]
            vector_map = self.vectors.get_vectors(record_ids)
            
            # neighbor_map: record_id -> List[neighbor_id]
            neighbor_map = {}
            all_neighbor_ids = set()
            
            for record in records:
                vec = vector_map.get(record.id)
                if not vec:
                    continue
                
                # v3.6.2/v3.8.0 Security Fix: Enforce user and namespace scoping in semantic search
                # candidates must belong to the same user AND namespace as the record being audited
                record_user_id = (record.metadata or {}).get("user_id")
                must_conditions = []
                if record_user_id:
                    must_conditions.append(FieldCondition(key="user_id", match=MatchValue(value=record_user_id)))
                if record.namespace:
                    must_conditions.append(FieldCondition(key="namespace", match=MatchValue(value=record.namespace)))
                
                search_filter = Filter(must=must_conditions) if must_conditions else None

                # Search Top-5 closest neighbors (limit 6 to exclude self)
                similar = self.vectors.search(query_embedding=vec, limit=6, filters=search_filter)
                sim_ids = [s[0] for s in similar if s[0] != record.id]
                neighbor_map[record.id] = sim_ids
                all_neighbor_ids.update(sim_ids)

            # v3.6.3 Fallback: Get random samples for records with no neighbors
            lonely_ids = [r.id for r in records if not neighbor_map.get(r.id)]
            if lonely_ids:
                random_pool = self.metadata.get_random(limit=20)
                random_ids = [r.id for r in random_pool]
                for rid in lonely_ids:
                    # Provide up to 5 random candidates for auditing
                    sample = [sid for sid in random_ids if sid != rid][:5]
                    neighbor_map[rid] = sample
                    all_neighbor_ids.update(sample)

            # Batch fetch all candidate metadata
            candidate_records = self.metadata.get_by_ids(list(all_neighbor_ids))
            cand_map = {c.id: c for c in candidate_records}

            # Final NLI Audit Loop
            for record in records:
                sim_ids = neighbor_map.get(record.id, [])
                candidates = [cand_map[sid] for sid in sim_ids if sid in cand_map]
                
                if not candidates:
                    continue

                conflicts = self._conflict_detector.detect_conflicts(record.content, candidates)
                for conflict in conflicts:
                    self._conflict_resolver.resolve(conflict, new_record=record)
                    conflicts_resolved += 1
                audited += 1
                
        except Exception as e:
            logger.error("Phase INTEGRITY: Integrity audit failed: %s", e)

        elapsed = time.time() - t0
        result = {"audited": audited, "conflicts_resolved": conflicts_resolved, "elapsed": round(elapsed, 2)}
        logger.info("Phase INTEGRITY: %s", result)
        return result

    @property
    def status(self) -> dict:
        """Current daemon status."""
        return {
            "running": self._running,
            "cycle_count": self._cycle_count,
            "last_cycle": self._last_cycle,
            "interval_hours": self.config.interval_hours,
        }