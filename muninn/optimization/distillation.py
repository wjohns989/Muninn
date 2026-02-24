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
        # This requires direct DB access or a new method on MuninnMemory.
        # For now, we simulate finding a cluster.
        # TODO: Implement proper clustering via vector density or graph communities.
        
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
        Identify groups of related episodic memories.
        """
        return await self.cluster_engine.find_episodic_clusters()

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
            # Mark archived and consolidated
            await self.memory.update(
                mem_id, 
                consolidated=True, 
                metadata={"archived": True, "distilled_into_cluster": cluster.get("id")}
            )