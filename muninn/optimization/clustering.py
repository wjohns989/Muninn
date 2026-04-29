"""
Vector Clustering Engine
------------------------
Implements 'Leader-Follower' clustering using iterative vector search.
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

    async def find_episodic_clusters(
        self, 
        min_cluster_size: int = 5, 
        similarity_threshold: float = 0.85,
        limit_candidates: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Identify clusters of related episodic memories.
        Returns a list of cluster dicts: {'id': '...', 'memory_ids': [...], 'topic': '...'}
        """
        clusters = []
        processed_ids: Set[str] = set()
        
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

        for leader in candidates:
            if leader.id in processed_ids:
                continue
            
            # Skip if already consolidated/archived (double check)
            if leader.archived or leader.consolidated:
                processed_ids.add(leader.id)
                continue

            # 2. Get Leader Vector
            vector = self.memory._vectors.get_vector(leader.id)
            if not vector:
                continue

            # 3. Find Neighbors (The "Followers")
            neighbors = self.memory._vectors.search(
                query_embedding=vector,
                limit=50, # Max cluster size cap
                score_threshold=similarity_threshold,
                filters={
                    "memory_type": "episodic",
                    # We might want to filter by project/namespace too, 
                    # but cross-project clustering could be interesting?
                    # Safer to restrict to same namespace for now.
                    "namespace": leader.namespace
                }
            )
            
            # Determine density-based core points (DBSCAN approach)
            valid_neighbors = []
            for mid, score in neighbors:
                if mid in processed_ids:
                    continue
                valid_neighbors.append(mid)

            if len(valid_neighbors) >= min_cluster_size:
                # Form Cluster, expanding density reachability
                cluster_member_ids = set(valid_neighbors)
                processed_ids.add(leader.id)
                seeds = set(valid_neighbors)
                seeds.discard(leader.id)

                while seeds:
                    current_id = seeds.pop()
                    if current_id in processed_ids:
                        continue

                    processed_ids.add(current_id)
                    curr_vector = self.memory._vectors.get_vector(current_id)
                    if not curr_vector:
                        continue

                    curr_neighbors = self.memory._vectors.search(
                        query_embedding=curr_vector,
                        limit=50,
                        score_threshold=similarity_threshold,
                        filters={
                            "memory_type": "episodic",
                            "namespace": leader.namespace
                        }
                    )

                    curr_valid_neighbors = []
                    for mid, score in curr_neighbors:
                        curr_valid_neighbors.append(mid)

                    if len(curr_valid_neighbors) >= min_cluster_size:
                        for n_id in curr_valid_neighbors:
                            if n_id not in cluster_member_ids:
                                cluster_member_ids.add(n_id)
                                if n_id not in processed_ids:
                                    seeds.add(n_id)

                cluster_id = f"cluster_{leader.id[:8]}"
                topic = f"Cluster around: {leader.content[:50]}..."
                
                # Fetch full records for the daemon to use
                cluster_records = self.memory._metadata.get_by_ids(list(cluster_member_ids))
                
                clusters.append({
                    "id": cluster_id,
                    "memory_ids": list(cluster_member_ids),
                    "topic": topic,
                    "memories": [r.model_dump() for r in cluster_records],
                    "namespace": leader.namespace,
                    "project": leader.project
                })
                
                # Mark as processed
                processed_ids.update(cluster_member_ids)
                logger.debug(f"Found cluster {cluster_id} size={len(cluster_member_ids)}")
            else:
                # Mark leader as processed (noise)
                processed_ids.add(leader.id)

        return clusters