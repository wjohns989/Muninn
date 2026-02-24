"""
Muninn Cognitive Optimization
-----------------------------
Modules for "Sleep-Time" maintenance and active reasoning:
- Distillation: Compressing episodic logs into semantic manuals.
- Surgery: Fixing incorrect memories based on feedback.
- Foraging: Active exploration for ambiguous queries.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

class DistillationJob(BaseModel):
    cluster_id: str
    memory_ids: List[str]
    target_topic: str
    status: str = "pending"

class MemoryCorrection(BaseModel):
    target_memory_id: str
    correction_text: str
    user_feedback_score: float = 0.0
