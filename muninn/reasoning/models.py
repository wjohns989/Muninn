"""
Muninn Reasoning Models
-----------------------
Pydantic schemas for structured cognitive reasoning tasks.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

class GapAnalysis(BaseModel):
    """
    Structured output for Omission Filtering / Gap Detection.
    Determines if sufficient information exists to fulfill a user intent.
    """
    verdict: str = Field(
        ..., 
        description="The final judgment: 'sufficient' if all necessary information is present, 'insufficient' if key details are missing."
    )
    missing_info: List[str] = Field(
        default_factory=list, 
        description="A list of specific information items that are missing. E.g., ['production database password', 'AWS region']."
    )
    reasoning: str = Field(
        ..., 
        description="A concise explanation of why the information is sufficient or what specific gaps prevent action."
    )
    grounding_memory_ids: List[str] = Field(
        default_factory=list, 
        description="List of memory IDs that were used to verify the presence of information."
    )

GAP_ANALYSIS_SYSTEM_PROMPT = (
    "You are a meticulous Cognitive Reasoning Engine for an AI agent. "
    "Your goal is to prevent hallucination by strictly verifying if you have "
    "ALL necessary information to complete a requested task.\n\n"
    "Input:\n"
    "1. User Intent: What the user wants to do.\n"
    "2. Context/Working Memory: Current conversation state.\n"
    "3. Retrieved Memories: Relevant long-term knowledge found in the database.\n\n"
    "Task:\n"
    "- Analyze the User Intent requirements (e.g., credentials, paths, IPs, versions).\n"
    "- Check if these requirements are satisfied by the Context or Retrieved Memories.\n"
    "- If ANYTHING is missing, verdict is 'insufficient'. List the missing items explicitly.\n"
    "- If EVERYTHING is present, verdict is 'sufficient'.\n\n"
    "Rules:\n"
    "- Be pessimistic. If you are 90% sure but missing a key parameter, mark it missing.\n"
    "- Do not assume defaults unless standard convention (e.g. localhost:8080).\n"
    "- Do not halllucinate credentials or paths."
)