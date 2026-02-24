"""
Muninn Reasoning Engine
-----------------------
Implements cognitive architecture components for proactive decision support.
Focuses on Omission Filtering (Gap Detection) and Grounding.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class GapAnalysis(BaseModel):
    verdict: str = Field(..., description="'sufficient' or 'insufficient'")
    missing_info: List[str] = Field(default_factory=list, description="List of specific missing information items")
    reasoning: str = Field(..., description="Explanation of the verdict")
    grounding_memory_ids: List[str] = Field(default_factory=list, description="IDs of memories used for this analysis")

class ReasoningRequest(BaseModel):
    query: str
    context: Optional[str] = None
    user_id: str = "global_user"
    limit: int = 10
