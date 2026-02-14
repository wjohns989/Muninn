"""
Muninn Goal Compass
-------------------
Project goal persistence and drift detection for cross-assistant continuity.
"""

import math
from typing import Any, Dict, List, Optional

from muninn.store.sqlite_metadata import SQLiteMetadataStore


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity with safe fallback for zero vectors."""
    if not vec_a or not vec_b:
        return 0.0
    length = min(len(vec_a), len(vec_b))
    if length == 0:
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for idx in range(length):
        a = float(vec_a[idx])
        b = float(vec_b[idx])
        dot += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class GoalCompass:
    """Persistence and drift checks for project north-star goals."""

    def __init__(
        self,
        *,
        metadata_store: SQLiteMetadataStore,
        embed_fn,
        drift_threshold: float = 0.55,
        signal_weight: float = 0.65,
        reminder_max_chars: int = 240,
    ):
        self.metadata = metadata_store
        self._embed_fn = embed_fn
        self.drift_threshold = drift_threshold
        self.signal_weight = signal_weight
        self.reminder_max_chars = reminder_max_chars

    @staticmethod
    def _compose_goal_text(goal_statement: str, constraints: List[str]) -> str:
        statement = (goal_statement or "").strip()
        lines = [statement] if statement else []
        for item in constraints or []:
            value = str(item).strip()
            if value:
                lines.append(f"- {value}")
        return "\n".join(lines).strip()

    async def _embed(self, text: str) -> List[float]:
        if not text.strip():
            return []
        result = self._embed_fn(text)
        if hasattr(result, "__await__"):
            return await result
        return result

    async def set_goal(
        self,
        *,
        user_id: str,
        namespace: str,
        project: str,
        goal_statement: str,
        constraints: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        cleaned_constraints = [str(c).strip() for c in (constraints or []) if str(c).strip()]
        goal_text = self._compose_goal_text(goal_statement, cleaned_constraints)
        goal_embedding = await self._embed(goal_text)
        self.metadata.set_project_goal(
            user_id=user_id,
            namespace=namespace,
            project=project,
            goal_statement=goal_statement.strip(),
            constraints=cleaned_constraints,
            goal_embedding=goal_embedding if goal_embedding else None,
        )
        goal = self.metadata.get_project_goal(
            user_id=user_id,
            namespace=namespace,
            project=project,
        )
        if goal is None:
            raise RuntimeError("Failed to persist project goal")
        return goal

    async def get_goal(
        self,
        *,
        user_id: str,
        namespace: str,
        project: str,
    ) -> Optional[Dict[str, Any]]:
        goal = self.metadata.get_project_goal(
            user_id=user_id,
            namespace=namespace,
            project=project,
        )
        if goal is None:
            return None

        if not goal.get("goal_embedding"):
            goal_text = self._compose_goal_text(
                goal.get("goal_statement", ""),
                goal.get("constraints", []),
            )
            embedding = await self._embed(goal_text)
            if embedding:
                self.metadata.set_project_goal(
                    user_id=user_id,
                    namespace=namespace,
                    project=project,
                    goal_statement=goal.get("goal_statement", ""),
                    constraints=goal.get("constraints", []),
                    goal_embedding=embedding,
                )
                goal["goal_embedding"] = embedding

        return goal

    async def evaluate_drift(
        self,
        *,
        text: str,
        user_id: str,
        namespace: str,
        project: str,
    ) -> Optional[Dict[str, Any]]:
        """Compare text intent to project goal and return drift diagnostics."""
        goal = await self.get_goal(user_id=user_id, namespace=namespace, project=project)
        if goal is None:
            return None

        goal_embedding = goal.get("goal_embedding") or []
        if not goal_embedding:
            return None

        query_embedding = await self._embed(text)
        if not query_embedding:
            return None

        cosine = _cosine_similarity(query_embedding, goal_embedding)
        similarity = max(0.0, min(1.0, (cosine + 1.0) / 2.0))
        is_drift = similarity < self.drift_threshold

        reminder = ""
        if is_drift:
            reminder = (
                f"Current project goal: {goal['goal_statement']}. "
                "This request may be drifting from that objective."
            )
            if goal.get("constraints"):
                reminder += f" Constraints: {', '.join(goal['constraints'])}."
            reminder = reminder[: self.reminder_max_chars].strip()

        return {
            "project": project,
            "namespace": namespace,
            "similarity": similarity,
            "threshold": self.drift_threshold,
            "is_drift": is_drift,
            "reminder": reminder,
            "goal_statement": goal.get("goal_statement", ""),
            "constraints": goal.get("constraints", []),
        }

