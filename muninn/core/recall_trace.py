"""
Muninn Explainable Recall Traces
----------------------------------
Per-signal attribution models that explain WHY each memory was retrieved.

This is a UNIQUE differentiator — no competitor (Mem0, Graphiti, Memento,
MemoryGraph) provides per-signal retrieval explanations.

RecallTrace objects describe the contribution of each retrieval signal
(vector, bm25, graph, temporal) to a memory's final ranking, enabling:
- Debugging: Why was an irrelevant memory returned?
- Trust: Users can verify retrieval logic is sound.
- Tuning: Adjust signal weights based on attribution data.
- Transparency: Agents can explain their reasoning.

Design: Pydantic models (not dataclass) for JSON serialization in API
responses and MCP tool output.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SignalContribution(BaseModel):
    """
    Contribution of a single retrieval signal to a memory's final score.

    Each signal (vector, bm25, graph, temporal) that contributed to finding
    a memory gets one SignalContribution entry.
    """
    signal: str = Field(
        description="Signal type: 'vector' | 'bm25' | 'graph' | 'temporal' | 'goal'"
    )
    raw_score: float = Field(
        description="Original score from that signal (cosine sim, BM25 score, etc.)"
    )
    rank: int = Field(
        description="Rank position in that signal's result list (0-indexed)"
    )
    rrf_contribution: float = Field(
        description="Actual RRF score contribution: weight / (K + rank + 1)"
    )
    weight: float = Field(
        description="Signal weight applied in fusion"
    )
    explanation: str = Field(
        description="Human-readable explanation of this signal's contribution"
    )


class RecallTrace(BaseModel):
    """
    Complete retrieval explanation for a single memory.

    Aggregates all signal contributions, importance boosting, reranking,
    and generates a human-readable explanation.
    """
    memory_id: str
    final_score: float = 0.0
    signals: List[SignalContribution] = Field(default_factory=list)
    rerank_score: Optional[float] = None
    importance_boost: float = 0.0
    dominant_signal: str = "unknown"
    explanation: str = ""

    def compute_final_score(self) -> None:
        """Compute final_score from signal contributions."""
        self.final_score = sum(s.rrf_contribution for s in self.signals)

    def determine_dominant_signal(self) -> None:
        """Identify the signal that contributed most."""
        if self.signals:
            best = max(self.signals, key=lambda s: s.rrf_contribution)
            self.dominant_signal = best.signal

    def generate_explanation(self) -> None:
        """
        Build a human-readable explanation of why this memory was recalled.

        Format: "Recalled primarily due to <dominant> (<detail>),
                 with <secondary signals>."
        """
        if not self.signals:
            self.explanation = "No retrieval signals (should not happen)."
            return

        self.determine_dominant_signal()

        # Sort by contribution (highest first)
        sorted_signals = sorted(
            self.signals, key=lambda s: s.rrf_contribution, reverse=True
        )

        # Build explanation from dominant + supporting signals
        parts = []
        for i, sig in enumerate(sorted_signals):
            if i == 0:
                parts.append(
                    f"Recalled primarily due to {sig.signal} "
                    f"({sig.explanation})"
                )
            else:
                parts.append(sig.explanation)

        if len(parts) == 1:
            self.explanation = parts[0]
        else:
            self.explanation = f"{parts[0]}, with {', '.join(parts[1:])}"

        # Append importance boost info if significant
        if self.importance_boost > 0.01:
            self.explanation += (
                f". Importance boosted score by "
                f"{self.importance_boost:.1%}."
            )

        # Append rerank info if available
        if self.rerank_score is not None:
            self.explanation += (
                f" Cross-encoder rerank score: {self.rerank_score:.3f}."
            )

    def finalize(self) -> "RecallTrace":
        """Compute all derived fields. Call after all signals are added."""
        self.compute_final_score()
        self.determine_dominant_signal()
        self.generate_explanation()
        return self

    def to_dict(self) -> dict:
        """Serialize for API/MCP responses."""
        return self.model_dump()


# ---------------------------------------------------------------------------
# Explanation helpers (used by HybridRetriever to describe each signal)
# ---------------------------------------------------------------------------

def explain_vector_signal(raw_score: float, rank: int) -> str:
    """Generate explanation for a vector similarity signal."""
    if raw_score > 0.85:
        strength = "High"
    elif raw_score > 0.70:
        strength = "Moderate"
    elif raw_score > 0.50:
        strength = "Weak"
    else:
        strength = "Marginal"
    return f"{strength} semantic similarity ({raw_score:.2f}) at rank #{rank}"


def explain_bm25_signal(raw_score: float, rank: int) -> str:
    """Generate explanation for a BM25 keyword match signal."""
    if raw_score > 5.0:
        strength = "Strong keyword match"
    elif raw_score > 2.0:
        strength = "Moderate keyword match"
    else:
        strength = "Weak keyword match"
    return f"{strength} (BM25 score {raw_score:.1f}, rank #{rank})"


def explain_graph_signal(raw_score: float, rank: int) -> str:
    """Generate explanation for a graph traversal signal."""
    return f"Connected via entity graph traversal (rank #{rank})"


def explain_temporal_signal(raw_score: float, rank: int) -> str:
    """Generate explanation for a temporal/recency signal."""
    return f"Temporal relevance (importance {raw_score:.2f}, rank #{rank})"


def explain_goal_signal(raw_score: float, rank: int) -> str:
    """Generate explanation for goal-alignment signal."""
    return f"Aligned with active project goal (score {raw_score:.2f}, rank #{rank})"


# Signal name → explanation function mapping
SIGNAL_EXPLAINERS = {
    "vector": explain_vector_signal,
    "bm25": explain_bm25_signal,
    "graph": explain_graph_signal,
    "temporal": explain_temporal_signal,
    "goal": explain_goal_signal,
}


def create_signal_contribution(
    signal: str,
    raw_score: float,
    rank: int,
    rrf_contribution: float,
    weight: float,
) -> SignalContribution:
    """
    Factory function to create a SignalContribution with auto-generated explanation.

    Args:
        signal: Signal type name.
        raw_score: Original score from the signal.
        rank: Position in that signal's ranked results.
        rrf_contribution: Computed RRF contribution.
        weight: Signal weight used.

    Returns:
        Fully populated SignalContribution.
    """
    explainer = SIGNAL_EXPLAINERS.get(signal, lambda s, r: f"{signal} signal (rank #{r})")
    return SignalContribution(
        signal=signal,
        raw_score=raw_score,
        rank=rank,
        rrf_contribution=rrf_contribution,
        weight=weight,
        explanation=explainer(raw_score, rank),
    )
