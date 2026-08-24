"""
Vector Clustering Engine
------------------------
Implements a DBSCAN-like density-based spatial clustering algorithm using iterative vector search.
Used by DistillationDaemon to identify semantic clusters of episodic memories.
"""

import logging
from typing import List, Dict, Any, Set, Iterator

from muninn.core.memory import MuninnMemory
from muninn.core.types import MemoryType

logger = logging.getLogger("Muninn.Optimization.Clustering")

class VectorClusterEngine:
    def __init__(self, memory: MuninnMemory):
        self.memory = memory
        self._last_scan_ts = 0.0 # Dirty Mark Optimization (v3.24.1)

    async def _region_query(self, memory_id: str, namespace: str, similarity_threshold: float) -> List[str]:
        vector = self.memory._vectors.get_vector(memory_id)
        if not vector:
            return []

        neighbors = self.memory._vectors.search(
            query_embedding=vector,
            limit=50, # Max neighbors to explore at once
            score_threshold=similarity_threshold,
            filters={
                "memory_type": "episodic",
                "namespace": namespace
            }
        )
        return [mid for mid, _ in neighbors]

    async def find_episodic_clusters(
        self, 
        min_cluster_size: int = 5, 
        similarity_threshold: float = 0.85,
        limit_candidates: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Identify clusters of related episodic memories using a DBSCAN-like algorithm.
        Returns a list of cluster dicts: {'id': '...', 'memory_ids': [...], 'topic': '...'}
        """
        clusters = []
        processed_ids: Set[str] = set()
        noise_ids: Set[str] = set()
        
        # 1. Fetch candidates (Episodic, not archived, since last scan)
        candidates = await self.memory._metadata.get_all(
            memory_type=MemoryType.EPISODIC,
            archived=False,
            created_at_min=self._last_scan_ts, # Only scan new memories
            limit=limit_candidates,
        )
        
        # Update high-water mark for next run
        if candidates:
            self._last_scan_ts = max(c.created_at for c in candidates)

        logger.info(f"Clustering scanning {len(candidates)} new candidates since {self._last_scan_ts}...")

        candidates_dict = {c.id: c for c in candidates}

        for p_id in list(candidates_dict.keys()):
            if p_id in processed_ids:
                continue
            
            p = candidates_dict[p_id]
            if p.archived or p.consolidated:
                processed_ids.add(p_id)
                continue

            processed_ids.add(p_id)

            neighbors = await self._region_query(p_id, p.namespace, similarity_threshold)
            
            valid_neighbors = list(neighbors) # Allow taking from noise

            if len(valid_neighbors) < min_cluster_size:
                noise_ids.add(p_id)
            else:
                # Found a cluster
                cluster_memories = set()
                cluster_memories.add(p_id)
                
                seed_set = list(valid_neighbors)
                seed_set.remove(p_id) if p_id in seed_set else None
                
                while seed_set:
                    q_id = seed_set.pop(0)

                    if q_id in noise_ids:
                        noise_ids.remove(q_id)
                        cluster_memories.add(q_id)
                    elif q_id not in processed_ids:
                        processed_ids.add(q_id)

                        # Double check metadata to ensure not archived
                        # We might not have q_id in candidates_dict if it's older
                        q_record = candidates_dict.get(q_id)
                        if not q_record:
                            q_records = self.memory._metadata.get_by_ids([q_id])
                            if q_records:
                                q_record = q_records[0]

                        if q_record and (q_record.archived or q_record.consolidated):
                            continue

                        cluster_memories.add(q_id)

                        q_namespace = q_record.namespace if q_record else p.namespace
                        q_neighbors = await self._region_query(q_id, q_namespace, similarity_threshold)

                        if len(q_neighbors) >= min_cluster_size:
                            for n in q_neighbors:
                                if n not in processed_ids and n not in seed_set:
                                    seed_set.append(n)

                if len(cluster_memories) >= min_cluster_size:
                    # Form Cluster
                    cluster_id = f"cluster_{p_id[:8]}"
                    topic = f"Cluster around: {p.content[:50]}..."

                    cluster_records = self.memory._metadata.get_by_ids(list(cluster_memories))

                    clusters.append({
                        "id": cluster_id,
                        "memory_ids": list(cluster_memories),
                        "topic": topic,
                        "memories": [r.model_dump() for r in cluster_records],
                        "namespace": p.namespace,
                        "project": p.project
                    })

                    logger.debug(f"Found cluster {cluster_id} size={len(cluster_memories)}")

        return clusters