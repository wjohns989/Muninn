"""
Entropy Scoring for Retrieval Ambiguity
---------------------------------------
Calculates Shannon Entropy of retrieval scores to detect when the system is 'confused'
(i.e., many results have similar low/medium scores).
"""

import math
from typing import List, Dict, Any

def calculate_shannon_entropy(scores: List[float]) -> float:
    """
    Calculate Shannon entropy of a score distribution.
    H(X) = -sum(p(x) * log2(p(x)))
    
    Args:
        scores: List of non-negative scores (e.g., cosine similarity).
    
    Returns:
        float: Entropy value (bits). Higher = more uncertainty.
    """
    if not scores:
        return 0.0
    
    # 1. Normalize to probability distribution
    total = sum(scores)
    if total <= 0:
        return 0.0
        
    probs = [s / total for s in scores]
    
    # 2. Compute Entropy
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)
            
    return entropy

def normalize_entropy(entropy: float, n: int) -> float:
    """
    Normalize entropy to [0, 1] range.
    Max entropy for n items is log2(n).
    """
    if n <= 1:
        return 0.0
    max_h = math.log2(n)
    return min(1.0, entropy / max_h)
