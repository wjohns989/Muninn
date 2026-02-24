"""
Memory Surgeon
--------------
Implements 'Generative Feedback' loops.
Finds erroneous memories and rewrites them based on user correction.
"""

import logging
from typing import Dict, Any, Optional
from muninn.core.memory import MuninnMemory

logger = logging.getLogger("Muninn.Optimization.Surgeon")

class MemorySurgeon:
    def __init__(self, memory: MuninnMemory):
        self.memory = memory

    async def correct_memory(self, target_id: str, correction: str) -> bool:
        """
        Apply a correction to a specific memory ID.
        This is a 'hard' fix: it rewrites the content.
        """
        # 1. Fetch original
        records = self.memory._metadata.get_by_ids([target_id])
        if not records:
            logger.warning(f"Surgeon could not find memory {target_id}")
            return False
        
        original = records[0]
        
        # 2. Rewrite content (Generative Step)
        # We assume the correction is the NEW truth.
        # Ideally, we'd use an LLM to merge them ("The user said X is actually Y...").
        # For MVP, we append the correction as a 'Correction Note'.
        
        new_content = f"{original.content}\n\n[CORRECTION]: {correction}"
        
        # 3. Update
        try:
            await self.memory.update(
                target_id, 
                data=new_content
            )
            # Boost score so it sticks
            # self.memory.update_score(target_id, boost=0.5) 
            return True
        except Exception as e:
            logger.error(f"Surgery failed on {target_id}: {e}")
            return False