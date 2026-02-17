"""
ColBERT Multi-Vector Interaction Engine
---------------------------------------
Implements late interaction (MaxSim) scoring for fine-grained retrieval precision.

Unlike dense retrievers that compress a document into a single vector,
ColBERT keeps token-level vectors and computes similarity via:
Sum(Max(QueryToken_i • DocToken_j))

This allows Muninn to match precise details lost in single-vector compression.
"""

import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger("Muninn.ColBERT")

class ColBERTScorer:
    """
    Lightweight ColBERT MaxSim scorer.
    Assumes vectors are already computed and normalized.
    """
    
    @staticmethod
    def maxsim_score(query_vectors: np.ndarray, doc_vectors: np.ndarray) -> float:
        """
        Compute MaxSim score between query and document.
        
        Args:
            query_vectors: shape (num_query_tokens, dim)
            doc_vectors: shape (num_doc_tokens, dim)
            
        Returns:
            sum of max similarities
        """
        # Similarity matrix: (num_query_tokens, num_doc_tokens)
        sim_matrix = np.dot(query_vectors, doc_vectors.T)
        
        # Max over doc tokens for each query token
        max_sims = np.max(sim_matrix, axis=1)
        
        # Sum of max sims
        return float(np.sum(max_sims))

    @staticmethod
    def batch_maxsim(query_vectors: np.ndarray, batch_doc_vectors: List[np.ndarray]) -> List[float]:
        """Compute scores for a batch of documents."""
        return [
            ColBERTScorer.maxsim_score(query_vectors, doc_vecs)
            for doc_vecs in batch_doc_vectors
        ]
