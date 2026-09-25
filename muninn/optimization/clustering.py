"""
Vector Clustering Engine
------------------------
Implements 'Leader-Follower' clustering using iterative vector search.
Used by DistillationDaemon to identify semantic clusters of episodic memories.
"""

import asyncio
import logging
from typing import Any, Dict, List, Set

from muninn.core.memory import MuninnMemory
from muninn.core.types import MemoryType

logger = logging.getLogger("Muninn.Optimization.Clustering")


class VectorClusterEngine:
    def __init__(self, memory: MuninnMemory):
        self.memory = memory
        self._last_scan_ts = 0.0  # Dirty Mark Optimization (v3.24.1)

    async def find_episodic_clusters(
        self,
        min_cluster_size: int = 5,
        similarity_threshold: float = 0.85,
        limit_candidates: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Identify clusters of related episodic memories.
        Returns a list of cluster dicts: {'id': '...', 'memory_ids': [...], 'topic': '...'}
        """
        clusters = []
        processed_ids: Set[str] = set()

        # 1. Fetch candidates (Episodic, not archived, since last scan)
        candidates = await asyncio.to_thread(
            self.memory._metadata.get_all,
            memory_type=MemoryType.EPISODIC,
            archived=False,
            created_at_min=self._last_scan_ts,
            limit=limit_candidates,
        )

        logger.info(
            "Clustering scanning %d candidates since %.3f",
            len(candidates),
            self._last_scan_ts,
        )
        candidate_vectors = await asyncio.to_thread(
            self.memory._vectors.get_vectors,
            [candidate.id for candidate in candidates],
        )

        for leader in candidates:
            if leader.id in processed_ids:
                continue

            # Skip if already consolidated/archived (double check)
            if leader.archived or leader.consolidated:
                processed_ids.add(leader.id)
                continue

            # 2. Get Leader Vector
            vector = candidate_vectors.get(leader.id)
            if not vector:
                continue

            # 3. Find Neighbors (The "Followers")
            leader_user_id = (leader.metadata or {}).get("user_id", "global_user")
            neighbors = await asyncio.to_thread(
                self.memory._vectors.search,
                query_embedding=vector,
                limit=50,
                score_threshold=similarity_threshold,
                filters={
                    "memory_type": "episodic",
                    "namespace": leader.namespace,
                    "project": leader.project,
                    "user_id": leader_user_id,
                },
            )

            neighbor_ids = [mid for mid, _score in neighbors if mid not in processed_ids]
            neighbor_records = await asyncio.to_thread(
                self.memory._metadata.get_by_ids,
                neighbor_ids,
            )
            records_by_id = {record.id: record for record in neighbor_records}
            scoped_records = [
                records_by_id[memory_id]
                for memory_id in neighbor_ids
                if memory_id in records_by_id
                and not records_by_id[memory_id].archived
                and not records_by_id[memory_id].consolidated
                and records_by_id[memory_id].namespace == leader.namespace
                and records_by_id[memory_id].project == leader.project
                and (records_by_id[memory_id].metadata or {}).get(
                    "user_id", "global_user"
                )
                == leader_user_id
            ]
            valid_neighbors = [record.id for record in scoped_records]

            if len(valid_neighbors) >= min_cluster_size:
                # 4. Form Cluster
                cluster_id = f"cluster_{leader.id[:8]}"
                topic = f"Cluster around: {leader.content[:50]}..."

                clusters.append(
                    {
                        "id": cluster_id,
                        "memory_ids": valid_neighbors,
                        "topic": topic,
                        "memories": [
                            record.model_dump() for record in scoped_records
                        ],
                        "namespace": leader.namespace,
                        "project": leader.project,
                    }
                )

                # Mark as processed
                processed_ids.update(valid_neighbors)
                logger.debug("Found cluster %s size=%d", cluster_id, len(valid_neighbors))
            else:
                # Mark leader as processed (noise)
                processed_ids.add(leader.id)

        if candidates:
            self._last_scan_ts = max(candidate.created_at for candidate in candidates)
        return clusters
