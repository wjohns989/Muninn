"""
Muninn Adaptive Retrieval Weights (v3.2.0)
-------------------------------------------
Entropy-based dynamic weighting that adapts per-query based on:
1. Query characteristics (length, temporal keywords, relational keywords)
2. Signal confidence via normalized entropy of result distributions
3. Historical feedback (optional, future extension)

Academic basis:
- "Entropy-Based Dynamic Hybrid Retrieval" (ICML 2025 Workshop)
- "Multi-Field Adaptive Retrieval" (ICLR 2025 Spotlight)
- Information-theoretic approach to multi-signal fusion

The WeightAdapter replaces fixed SIGNAL_WEIGHTS in hybrid.py when the
`adaptive_weights` feature flag is enabled.

Dependencies: None — pure Python math (stdlib only)
"""

import math
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger("Muninn.WeightAdapter")

# Default fixed weights — identical to hybrid.py SIGNAL_WEIGHTS
DEFAULT_WEIGHTS: Dict[str, float] = {
    "vector": 1.0,
    "graph": 1.0,
    "bm25": 0.8,
    "temporal": 0.5,
}

# Query classification keywords
TEMPORAL_KEYWORDS = frozenset({
    "recent", "recently", "latest", "today", "yesterday", "new", "newest",
    "last", "ago", "just", "current", "now", "updated", "fresh",
})

RELATIONAL_KEYWORDS = frozenset({
    "related", "connected", "about", "who", "between", "linked",
    "associated", "involving", "regarding", "concerning",
})

# Adaptation multipliers — constrained to prevent runaway weights
QUERY_SHORT_BM25_BOOST = 1.3       # Short queries (≤3 tokens) boost BM25
QUERY_TEMPORAL_BOOST = 2.0          # Temporal keywords boost temporal signal
QUERY_RELATIONAL_GRAPH_BOOST = 1.5  # Relational keywords boost graph signal

# Entropy-based confidence scaling bounds
ENTROPY_CONFIDENCE_FLOOR = 0.5      # Minimum multiplier (low confidence)
ENTROPY_CONFIDENCE_CEILING = 1.5    # Maximum multiplier (high confidence)


class WeightAdapter:
    """
    Computes dynamic per-query signal weights for RRF fusion.

    When adaptive_weights feature flag is ON, this replaces the fixed
    SIGNAL_WEIGHTS dict in HybridRetriever.

    Weight computation pipeline:
      1. Start with DEFAULT_WEIGHTS
      2. Apply query-characteristic multipliers
      3. Apply entropy-based signal confidence scaling
      4. Normalize to prevent weight explosion
    """

    def __init__(
        self,
        base_weights: Optional[Dict[str, float]] = None,
        temporal_keywords: Optional[frozenset] = None,
        relational_keywords: Optional[frozenset] = None,
    ):
        """
        Initialize with optional custom base weights and keyword sets.

        Args:
            base_weights: Starting weights per signal. Defaults to DEFAULT_WEIGHTS.
            temporal_keywords: Keywords that trigger temporal signal boost.
            relational_keywords: Keywords that trigger graph signal boost.
        """
        self.base_weights = dict(base_weights or DEFAULT_WEIGHTS)
        self.temporal_keywords = temporal_keywords or TEMPORAL_KEYWORDS
        self.relational_keywords = relational_keywords or RELATIONAL_KEYWORDS

    def compute_weights(
        self,
        query: str,
        signal_results: Optional[Dict[str, List[Tuple[str, int]]]] = None,
    ) -> Dict[str, float]:
        """
        Compute dynamic weights for a specific query and its signal results.

        Args:
            query: The search query text.
            signal_results: Dict mapping signal name → list of (doc_id, rank)
                            or (doc_id, score) tuples. Used for entropy-based
                            confidence computation. If None, only query-based
                            adaptation is applied.

        Returns:
            Dict[str, float] mapping signal name → adapted weight.
        """
        weights = dict(self.base_weights)

        # Phase 1: Query-based adaptation
        weights = self._adapt_for_query(query, weights)

        # Phase 2: Entropy-based signal confidence scaling
        if signal_results:
            weights = self._adapt_for_entropy(signal_results, weights)

        # Phase 3: Normalize to prevent weight explosion
        weights = self._normalize_weights(weights)

        logger.debug(
            "Adaptive weights for query '%s': %s",
            query[:50],
            {k: round(v, 3) for k, v in weights.items()},
        )

        return weights

    def _adapt_for_query(
        self,
        query: str,
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Adjust weights based on query characteristics.

        Heuristics:
          - Short queries (≤3 tokens): Boost BM25 (keyword precision matters more)
          - Temporal keywords present: Boost temporal signal
          - Relational keywords present: Boost graph signal
        """
        tokens = query.lower().split()
        token_set = frozenset(tokens)

        # Short queries → keyword match becomes more valuable
        if len(tokens) <= 3:
            weights["bm25"] *= QUERY_SHORT_BM25_BOOST

        # Temporal indicators → boost recency/temporal signal
        if token_set & self.temporal_keywords:
            weights["temporal"] *= QUERY_TEMPORAL_BOOST

        # Relational indicators → boost graph traversal
        if token_set & self.relational_keywords:
            weights["graph"] *= QUERY_RELATIONAL_GRAPH_BOOST

        return weights

    def _adapt_for_entropy(
        self,
        signal_results: Dict[str, List[Tuple[str, int]]],
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Scale weights by signal confidence estimated via normalized entropy.

        High entropy = scores are uniformly distributed = LOW confidence
        → reduce weight (signal can't distinguish relevant from irrelevant)

        Low entropy = scores are concentrated = HIGH confidence
        → increase weight (signal has strong discriminative power)

        The confidence multiplier is bounded to [FLOOR, CEILING] to prevent
        any single signal from dominating or being zeroed out.
        """
        for signal_name, results in signal_results.items():
            if signal_name not in weights:
                continue

            if not results or len(results) < 2:
                # Insufficient data — keep base weight unchanged
                continue

            # Extract rank-based scores for entropy computation
            # Transform ranks to pseudo-scores: score = 1/(rank+1)
            # This gives us a distribution to compute entropy over
            pseudo_scores = [1.0 / (rank + 1) for _, rank in results]

            entropy = self._normalized_entropy(pseudo_scores)
            # confidence ∈ [0, 1] where 1 = perfectly confident
            confidence = 1.0 - entropy

            # Map confidence to multiplier ∈ [FLOOR, CEILING]
            multiplier = (
                ENTROPY_CONFIDENCE_FLOOR
                + confidence * (ENTROPY_CONFIDENCE_CEILING - ENTROPY_CONFIDENCE_FLOOR)
            )

            weights[signal_name] *= multiplier

        return weights

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to maintain consistent scale with base weights.

        We normalize so that the sum of adapted weights equals the sum of
        base weights. This prevents weight inflation/deflation from changing
        the relative magnitude of RRF scores across different queries.
        """
        base_sum = sum(self.base_weights.values())
        adapted_sum = sum(weights.values())

        if adapted_sum <= 0:
            return dict(self.base_weights)

        scale = base_sum / adapted_sum
        return {k: v * scale for k, v in weights.items()}

    @staticmethod
    def _normalized_entropy(scores: List[float]) -> float:
        """
        Compute normalized Shannon entropy of a score distribution.

        H_norm = H(p) / H_max = -sum(p_i * log2(p_i)) / log2(n)

        Returns:
            Float in [0.0, 1.0] where:
              0.0 = all probability mass on one item (perfectly peaked)
              1.0 = uniform distribution (maximum uncertainty)
        """
        if not scores:
            return 1.0

        # Filter out non-positive scores
        positive_scores = [s for s in scores if s > 0]
        if not positive_scores:
            return 1.0

        n = len(positive_scores)
        if n == 1:
            return 0.0  # Single element = zero entropy (perfectly certain)

        # Normalize to probability distribution
        total = sum(positive_scores)
        probs = [s / total for s in positive_scores]

        # Shannon entropy
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        # Maximum possible entropy for n items
        max_entropy = math.log2(n)

        if max_entropy <= 0:
            return 0.0

        return min(1.0, entropy / max_entropy)

    def explain(
        self,
        query: str,
        signal_results: Optional[Dict[str, List[Tuple[str, int]]]] = None,
    ) -> Dict[str, str]:
        """
        Generate human-readable explanation of weight adaptation decisions.

        Returns:
            Dict mapping signal name → explanation string.
        """
        explanations: Dict[str, str] = {}
        weights = self.compute_weights(query, signal_results)
        base = self.base_weights

        for signal, weight in weights.items():
            base_w = base.get(signal, 0.0)
            if base_w == 0:
                explanations[signal] = f"{signal}: no base weight"
                continue

            ratio = weight / base_w
            if abs(ratio - 1.0) < 0.01:
                explanations[signal] = f"{signal}: unchanged ({weight:.3f})"
            elif ratio > 1.0:
                explanations[signal] = (
                    f"{signal}: boosted {ratio:.1f}x → {weight:.3f} "
                    f"(from {base_w:.3f})"
                )
            else:
                explanations[signal] = (
                    f"{signal}: reduced {ratio:.1f}x → {weight:.3f} "
                    f"(from {base_w:.3f})"
                )

        return explanations
