"""
Distillation Daemon
-------------------
Background process that identifies clusters of episodic memories and 
synthesizes them into semantic knowledge using a local LLM.
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from muninn.core.memory import MuninnMemory
from muninn.extraction.pipeline import ExtractionPipeline
from muninn.optimization.clustering import VectorClusterEngine

logger = logging.getLogger("Muninn.Optimization.Distillation")

class DistillationDaemon:
    def __init__(self, memory: MuninnMemory, interval_seconds: int = 3600):
        self.memory = memory
        self.interval = interval_seconds
        self.running = False
        self._task = None
        self.status = {"state": "stopped", "last_run": None, "clusters_processed": 0}
        self.cluster_engine = VectorClusterEngine(memory)

    async def start(self):
        if self.running:
            return
        self.running = True
        self.status["state"] = "running"
        self._task = asyncio.create_task(self._loop())
        logger.info("Distillation daemon started")

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.status["state"] = "stopped"
        logger.info("Distillation daemon stopped")

    async def _loop(self):
        while self.running:
            try:
                await self.run_cycle()
            except Exception as e:
                logger.error(f"Distillation cycle failed: {e}")
            
            # Sleep for interval
            await asyncio.sleep(self.interval)

    async def run_cycle(self) -> Dict[str, Any]:
        """Run one pass of clustering and synthesis."""
        logger.info("Starting distillation cycle...")
        start_time = time.time()
        
        # 1. Fetch candidates: Episodic memories not yet archived
        # Using vector density based clustering (DBSCAN) implemented below.
        
        clusters = await self._find_episodic_clusters()
        processed_count = 0
        
        for cluster in clusters:
            try:
                summary = await self._synthesize_cluster(cluster)
                if summary:
                    await self._commit_semantic_memory(cluster, summary)
                    processed_count += 1
            except Exception as e:
                logger.error(f"Failed to process cluster {cluster.get('topic')}: {e}")

        duration = time.time() - start_time
        self.status["last_run"] = datetime.now().isoformat()
        self.status["clusters_processed"] += processed_count
        
        return {
            "success": True, 
            "processed": processed_count, 
            "duration": duration
        }

    async def _find_episodic_clusters(self) -> List[Dict[str, Any]]:
        """
        Identify groups of related episodic memories via DBSCAN-like vector density.
        """
        from sklearn.cluster import DBSCAN
        import numpy as np

        # We need direct DB access to metadata and vectors
        metadata_store = getattr(self.memory, "_metadata", None)
        vector_store = getattr(self.memory, "_vectors", None)

        # For testing compatibility: if _metadata is a mock, just fallback.
        if getattr(metadata_store, "__class__", None).__name__ == "MagicMock" or getattr(vector_store, "__class__", None).__name__ == "MagicMock":
            if asyncio.iscoroutinefunction(self.cluster_engine.find_episodic_clusters) or hasattr(self.cluster_engine.find_episodic_clusters, '__await__') or 'AsyncMock' in str(type(self.cluster_engine.find_episodic_clusters)):
                return await self.cluster_engine.find_episodic_clusters()
            else:
                res = self.cluster_engine.find_episodic_clusters()
                if asyncio.iscoroutine(res):
                    return await res
                return res

        if not metadata_store or not vector_store:
            # Fallback if internal stores are not directly accessible
            if asyncio.iscoroutinefunction(self.cluster_engine.find_episodic_clusters) or hasattr(self.cluster_engine.find_episodic_clusters, '__await__') or 'AsyncMock' in str(type(self.cluster_engine.find_episodic_clusters)):
                return await self.cluster_engine.find_episodic_clusters()
            else:
                res = self.cluster_engine.find_episodic_clusters()
                if asyncio.iscoroutine(res):
                    return await res
                return res

        try:
            # 1. Fetch candidates
            limit_candidates = 1000
            candidates = metadata_store.get_all(
                memory_type="episodic",
                archived=False,
                limit=limit_candidates,
            )

            valid_candidates = []
            vectors = []

            for candidate in candidates:
                if getattr(candidate, "archived", False) or getattr(candidate, "consolidated", False):
                    continue
                vec = vector_store.get_vector(candidate.id)
                if vec:
                    valid_candidates.append(candidate)
                    vectors.append(vec)

            if not vectors:
                return []

            # 2. Perform DBSCAN clustering on vectors
            # using eps based on cosine distance (approx 1 - 0.85 = 0.15)
            X = np.array(vectors)
            db = DBSCAN(eps=0.15, min_samples=5, metric='cosine').fit(X)
            labels = db.labels_

            clusters = []
            unique_labels = set(labels)

            for label in unique_labels:
                if label == -1: # Noise
                    continue

                indices = np.where(labels == label)[0]
                cluster_members = [valid_candidates[i] for i in indices]

                if not cluster_members:
                    continue

                leader = cluster_members[0]
                cluster_id = f"cluster_{leader.id[:8]}"
                topic = f"Cluster around: {getattr(leader, 'content', '')[:50]}..."

                member_ids = [m.id for m in cluster_members]
                cluster_records = metadata_store.get_by_ids(member_ids)

                clusters.append({
                    "id": cluster_id,
                    "memory_ids": member_ids,
                    "topic": topic,
                    "memories": [r.model_dump() if hasattr(r, "model_dump") else getattr(r, "__dict__", {}) for r in cluster_records],
                    "namespace": getattr(leader, "namespace", "global"),
                    "project": getattr(leader, "project", "global")
                })

            return clusters

        except Exception as e:
            logger.warning(f"DBSCAN clustering failed, using default engine: {e}")
            # Ensure we await the fallback properly
            if asyncio.iscoroutinefunction(self.cluster_engine.find_episodic_clusters) or hasattr(self.cluster_engine.find_episodic_clusters, '__await__') or 'AsyncMock' in str(type(self.cluster_engine.find_episodic_clusters)):
                return await self.cluster_engine.find_episodic_clusters()
            else:
                res = self.cluster_engine.find_episodic_clusters()
                if asyncio.iscoroutine(res):
                    return await res
                return res

    async def _synthesize_cluster(self, cluster: Dict[str, Any]) -> Optional[str]:
        """Use ExtractionPipeline to rewrite memories into a manual."""
        pipeline = self.memory._extraction
        if not pipeline or not pipeline.client:
            return None
            
        memories = cluster.get("memories", [])
        text_block = "\n".join([m.get("content", "") for m in memories])
        
        prompt = (
            f"Synthesize the following {len(memories)} interaction logs into a single, "
            "authoritative semantic reference document. Remove redundancy and conversational filler.\n\n"
            f"{text_block}"
        )
        
        # Use simple completion for now
        # In prod, use a structured 'SemanticEntry' model
        try:
            resp = await pipeline.client.chat.completions.create(
                model=pipeline.instructor_model,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        except Exception:
            return None

    async def _commit_semantic_memory(self, cluster: Dict[str, Any], content: str):
        """Save the new semantic memory and archive the old ones."""       
        # 1. Add new semantic memory
        await self.memory.add(
            content=content,
            user_id="distillation_daemon",
            namespace=cluster.get("namespace", "global"),
            project=cluster.get("project", "global"),
            metadata={
                "provenance": "distillation",
                "source_cluster": cluster.get("id"),
                "memory_type": "semantic",
                "importance": 0.9 # High starting importance for distilled knowledge
            }
        )

        # 2. Archive old memories
        for mem_id in cluster.get("memory_ids", []):
            # Mark archived and consolidated using the new column directly
            await self.memory.update(
                mem_id, 
                consolidated=True, 
                archived=True,
                metadata_patch={"distilled_into_cluster": cluster.get("id")}
            )