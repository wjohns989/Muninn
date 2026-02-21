"""
Muninn Temporal Synthesis (Phase 22)
------------------------------------
Evaluates semantic contradictions between two facts to determine
temporal precedence and establish Shadow Edges.
"""

import logging
from typing import Optional

from muninn.extraction.models import (
    TemporalContradictionResolution,
    TEMPORAL_SYNTHESIS_PROMPT,
)
from muninn.extraction.instructor_extractor import InstructorExtractor

logger = logging.getLogger("Muninn.Extract.Temporal")


def synthesize_temporal_contradiction(
    extractor: InstructorExtractor,
    fact_a_text: str,
    fact_b_text: str,
) -> Optional[TemporalContradictionResolution]:
    """
    Given two potentially contradictory statements, dispatch an LLM synthesis
    prompt to determine if they truly contradict and which one supersedes the other.

    Args:
        extractor: An initialized InstructorExtractor instance.
        fact_a_text: The older existing memory content.
        fact_b_text: The newer memory content.

    Returns:
        TemporalContradictionResolution object if the LLM successfully responded, else None.
    """
    try:
        # We access the raw instructor client from the InstructorExtractor
        client = extractor._client
        model = extractor.model

        response: TemporalContradictionResolution = client.chat.completions.create(
            model=model,
            response_model=TemporalContradictionResolution,
            messages=[
                {"role": "system", "content": TEMPORAL_SYNTHESIS_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Fact A (Older):\n{fact_a_text}\n\n"
                        f"Fact B (Newer):\n{fact_b_text}\n\n"
                        "Evaluate if there is a strict contradiction. If so, Fact B should "
                        "generally supersede Fact A unless logically impossible."
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=256,
        )
        return response
    except Exception as e:
        logger.error(f"Temporal contradiction synthesis failed: {e}")
        return None
