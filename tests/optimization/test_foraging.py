"""
Tests for Muninn Epistemic Foraging (Phase 26)
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from muninn.optimization.foraging import ForagingEngine
from muninn.scoring.entropy import calculate_shannon_entropy, normalize_entropy

def test_entropy_calculation():
    # Certain distribution
    scores_certain = [1.0, 0.1, 0.1]
    h_certain = calculate_shannon_entropy(scores_certain)
    n_h_certain = normalize_entropy(h_certain, 3)
    
    # Ambiguous distribution
    scores_flat = [0.5, 0.5, 0.5]
    h_flat = calculate_shannon_entropy(scores_flat)
    n_h_flat = normalize_entropy(h_flat, 3)
    
    assert n_h_flat > n_h_certain
    assert n_h_flat == pytest.approx(1.0) # Flat distribution is max entropy

@pytest.fixture
def mock_memory():
    m = MagicMock()
    return m

@pytest.mark.asyncio
async def test_foraging_trigger_high_entropy(mock_memory):
    engine = ForagingEngine(mock_memory)
    
    # Setup ambiguous results
    results = [
        {"id": "m1", "score": 0.5},
        {"id": "m2", "score": 0.5},
        {"id": "m3", "score": 0.5}
    ]
    
    result = await engine.forage("test query", results, ambiguity_threshold=0.5)
    
    assert result["triggered"] is True
    assert result["entropy"] > 0.5

@pytest.mark.asyncio
async def test_foraging_not_triggered_low_entropy(mock_memory):
    engine = ForagingEngine(mock_memory)
    
    # Setup certain results
    results = [
        {"id": "m1", "score": 0.9},
        {"id": "m2", "score": 0.1},
        {"id": "m3", "score": 0.1}
    ]
    
    result = await engine.forage("test query", results, ambiguity_threshold=0.8)
    
    assert result["triggered"] is False
    assert result["reason"] == "low_entropy"
