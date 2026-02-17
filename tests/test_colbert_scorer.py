"""
Tests for ColBERT multi-vector scoring logic.
"""

import numpy as np
from muninn.advanced.colbert import ColBERTScorer

def test_maxsim_score_exact_match():
    # Query: [A, B]
    # Doc: [A, B, C]
    # Expect max sim 1.0 for A, 1.0 for B -> total 2.0
    query = np.array([[1.0, 0.0], [0.0, 1.0]])
    doc = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    
    score = ColBERTScorer.maxsim_score(query, doc)
    assert np.isclose(score, 2.0)

def test_maxsim_score_partial_match():
    # Query: [A]
    # Doc: [B]
    # A=[1,0], B=[0,1] -> dot=0 -> max=0 -> score=0
    query = np.array([[1.0, 0.0]])
    doc = np.array([[0.0, 1.0]])
    
    score = ColBERTScorer.maxsim_score(query, doc)
    assert np.isclose(score, 0.0)

def test_batch_maxsim():
    query = np.array([[1.0, 0.0]])
    docs = [
        np.array([[1.0, 0.0]]),
        np.array([[0.0, 1.0]])
    ]
    scores = ColBERTScorer.batch_maxsim(query, docs)
    assert np.allclose(scores, [1.0, 0.0])
