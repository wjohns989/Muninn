"""
Omission Detector
-----------------
Implements the core 'Gap Analysis' logic for the CoALA reasoning layer.
"""

import logging
from typing import Dict, Any, Optional

from muninn.core.memory import MuninnMemory
from muninn.reasoning.models import GapAnalysis, GAP_ANALYSIS_SYSTEM_PROMPT
from muninn.extraction.pipeline import ExtractionPipeline

logger = logging.getLogger("Muninn.Reasoning")

class OmissionDetector:
    def __init__(self, memory: MuninnMemory):
        self.memory = memory
        # We'll use the memory's existing extraction pipeline if available,
        # otherwise we might need to instantiate one or use a lighter weight method.
        # Accessing private _extraction is a bit dirty, but typical for internal modules.
        # Ideally MuninnMemory exposes extraction or we pass it in.
        # For now, we assume MuninnMemory has it initialized.
        pass

    async def detect_gaps(
        self, 
        query: str, 
        context: Optional[str] = None,
        user_id: str = "global_user",
        limit: int = 10
    ) -> GapAnalysis:
        """
        Analyze a query and context to determine if information is missing.
        """
        # 1. Retrieve relevant memories (The "Recall" step of CoALA)
        # We use 'hunt' to find multi-hop context
        retrieved_memories = await self.memory.hunt(
            query=f"{query} {context or ''}",
            user_id=user_id,
            limit=limit,
            depth=1, # Shallow hunt for speed
            synthesize=False
        )
        
        # 2. Prepare the prompt context
        memory_text = "\n".join([
            f"[{m.get('id', 'unknown')}] {m.get('memory', '')}"
            for m in retrieved_memories
        ])
        
        full_text = (
            f"USER INTENT: {query}\n"
            f"WORKING CONTEXT: {context or 'None'}\n"
            f"RETRIEVED MEMORIES:\n{memory_text}"
        )

        # 3. Reasoning (The "Decide" step of CoALA)
        # We need to call the LLM to fill the GapAnalysis model.
        # This requires access to the ExtractionPipeline's underlying instructor client.
        
        pipeline = self.memory._extraction
        if not pipeline or not pipeline.client:
             # Fallback if no LLM configured: assume insufficient to be safe?
             # Or raise error?
             logger.warning("No extraction pipeline available for reasoning.")
             return GapAnalysis(
                 verdict="insufficient", 
                 missing_info=["Reasoning engine unavailable"], 
                 reasoning="LLM not configured."
             )

        try:
            # We use the raw instructor client to get the GapAnalysis model
            # This bypasses the standard "extract entities" flow.
            # Using 'xlam' or 'ollama' or 'instructor' depending on config.
            # For reasoning, we prefer 'high_reasoning' or 'balanced' profile.
            
            # Using the pipeline's _extract_structured helper if possible, 
            # but it is tailored for ExtractedMemoryFacts.
            # We might need to call pipeline.client.chat.completions.create directly
            # if using Instructor.
            
            resp = await pipeline.client.chat.completions.create(
                model=pipeline.instructor_model,
                messages=[
                    {"role": "system", "content": GAP_ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": full_text}
                ],
                response_model=GapAnalysis,
                max_retries=2,
            )
            return resp

        except Exception as e:
            logger.error(f"Gap analysis failed: {e}")
            return GapAnalysis(
                verdict="insufficient",
                missing_info=["Error during analysis"],
                reasoning=f"Internal error: {str(e)}"
            )