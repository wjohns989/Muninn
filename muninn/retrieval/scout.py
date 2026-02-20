"""
Muninn Scout — Agentic Context Discovery
----------------------------------------
Intelligent retrieval agent that proactively hunts for hidden context,
multi-hop relationships, and cross-project knowledge.
"""

import logging
import asyncio
import time
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict

from muninn.core.types import SearchResult, MemoryRecord
from muninn.retrieval.hybrid import HybridRetriever
from muninn.core.feature_flags import get_flags

logger = logging.getLogger("Muninn.Scout")

class MuninnScout:
    """
    Intelligent search agent that 'searches out' memory rather than just matching it.
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    async def hunt(
        self,
        query: str,
        limit: int = 10,
        user_id: str = "global_user",
        namespaces: Optional[List[str]] = None,
        depth: int = 2,
        expand_query: bool = True
    ) -> List[SearchResult]:
        """
        Perform a multi-hop agentic search.
        """
        t0 = time.time()
        flags = get_flags()
        
        # 1. Initial Hybrid Search (Broad)
        initial_results = await self.retriever.search(
            query=query,
            limit=limit,
            user_id=user_id,
            namespaces=namespaces,
            explain=True,
            rerank=True
        )
        
        if not initial_results:
            # Try global fallback immediately if project search failed
            initial_results = await self.retriever.search(
                query=query,
                limit=limit,
                user_id=user_id,
                filters={"scope": "global"},
                explain=True
            )

        # 2. Graph-Based Discovery (Hop 1 & 2)
        # Find all entities mentioned in initial results
        seed_ids = [r.memory.id for r in initial_results]
        discovery_ids = set(seed_ids)
        
        related_entities = set()
        for r in initial_results:
            meta = r.memory.metadata or {}
            entities = meta.get("entity_names", [])
            if isinstance(entities, list):
                related_entities.update(entities)

        # multi-hop expansion via graph
        if related_entities and depth > 0:
            logger.info("Scout: Expanding via %d entities", len(related_entities))
            # Find memories related to these entities
            expanded_ids = self.retriever.graph.find_related_memories(
                list(related_entities),
                limit=limit * 2,
                user_id=user_id,
                namespace=namespaces[0] if namespaces and len(namespaces) == 1 else "global"
            )
            discovery_ids.update(expanded_ids)

        # 3. Chain Discovery (Temporal/Causal)
        # Follow PRECEDES/CAUSES edges from initial seeds
        if flags.is_enabled("memory_chains") and seed_ids:
            chain_results = self.retriever.graph.find_chain_related_memories(
                seed_ids, 
                limit=limit
            )
            for mid, _score in chain_results:
                discovery_ids.add(mid)

        # 4. Aggregate & Rerank
        # Fetch all discovered records
        all_ids = list(discovery_ids)
        if not all_ids:
            return initial_results

        # Fetch full records from metadata
        records = self.retriever.metadata.get_by_ids(all_ids)
        record_map = {r.id: r for r in records}

        # Final scoring (Hybrid + Discovery Bonus)
        # We re-score the union using the retriever's logic
        final_results = await self.retriever.search(
            query=query,
            limit=limit,
            user_id=user_id,
            namespaces=namespaces,
            filters={"memory_ids": all_ids}, # We need a way to filter to specific IDs in retriever
            rerank=True
        )

        # 5. Synthesize Discovery Path
        # (This would ideally use an LLM, but we'll build the trace here)
        elapsed = time.time() - t0
        logger.info("Scout: Hunt completed in %.3fs. Found %d candidates.", elapsed, len(all_ids))
        
        return final_results
