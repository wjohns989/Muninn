"""
Foraging Engine
---------------
Implements 'Epistemic Foraging' (Active Inference).
When retrieval is ambiguous, this engine follows graph edges to find 'scent trails'
and executes exploratory queries.
"""

import logging
from typing import List, Dict, Any, Optional

from muninn.core.memory import MuninnMemory
from muninn.scoring.entropy import calculate_shannon_entropy, normalize_entropy

logger = logging.getLogger("Muninn.Optimization.Foraging")

class ForagingEngine:
    def __init__(self, memory: MuninnMemory):
        self.memory = memory

    async def forage(
        self, 
        initial_query: str, 
        initial_results: List[Dict[str, Any]],
        ambiguity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Analyze initial results and actively forage if ambiguous.
        """
        # 1. Check Ambiguity
        scores = [r.get("score", 0.0) for r in initial_results]
        entropy = calculate_shannon_entropy(scores)
        norm_entropy = normalize_entropy(entropy, len(scores))
        
        logger.info(f"Foraging Analysis: Query='{initial_query}' Entropy={norm_entropy:.3f}")
        
        if norm_entropy < ambiguity_threshold:
            return {
                "triggered": False,
                "reason": "low_entropy",
                "entropy": norm_entropy,
                "new_results": []
            }

        # 2. Extract Entities from Top-K
        # We need the memory IDs to check the graph
        top_ids = [r.get("id") for r in initial_results[:3]] # Look at top 3 for scent
        
        # 3. Graph Scent Following
        # Find neighbors of these memories in the Knowledge Graph
        scent_terms = set()
        
        # This requires graph access.
        # Assuming memory._graph has methods to get neighbors.
        # We'll need to extend GraphStore or use existing methods.
        # For now, we'll try to get entities linked to these memories.
        
        try:
            for mid in top_ids:
                # Get entities mentioned in this memory
                # GraphStore.get_memory_entities(mid) -> List[str]
                # We need to check if this method exists or assume schema.
                pass
                # Placeholder: In a real implementation, we query Kuzu
                # MATCH (m:Memory)-[:MENTIONS]->(e:Entity) WHERE m.id = $mid RETURN e.name
        except Exception as e:
            logger.warning(f"Graph forage failed: {e}")

        # 4. Generate Exploratory Queries
        # If we found related entities, search for them.
        # For this MVP, we will return the "suggestion" to forage.
        
        return {
            "triggered": True,
            "entropy": norm_entropy,
            "scent_trails": list(scent_terms),
            "suggestion": "Ambiguity detected. Consider searching for related entities."
        }
