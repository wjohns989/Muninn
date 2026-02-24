"""
Tests for Phase 24: CoALA Omission Filtering
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from muninn.reasoning.models import GapAnalysis
from muninn.reasoning.omission import OmissionDetector

# Mock memory record for retrieval
MOCK_MEMORY = {
    "id": "mem-1",
    "memory": "The production server IP is 10.0.0.5",
    "score": 0.9,
    "memory_type": "fact"
}

@pytest.fixture
def mock_memory():
    m = MagicMock()
    m.hunt = AsyncMock(return_value=[MOCK_MEMORY])
    # Mock extraction pipeline
    pipeline = MagicMock()
    client = AsyncMock()
    completions = MagicMock()
    completions.create = AsyncMock(return_value=GapAnalysis(
        verdict="sufficient",
        missing_info=[],
        reasoning="Production IP is present.",
        grounding_memory_ids=["mem-1"]
    ))
    client.chat.completions = completions
    pipeline.client = client
    pipeline.instructor_model = "test-model"
    m._extraction = pipeline
    return m

@pytest.mark.asyncio
async def test_detect_gaps_success(mock_memory):
    """Test standard happy path for omission detection."""
    detector = OmissionDetector(mock_memory)
    result = await detector.detect_gaps("Deploy to prod")
    
    assert isinstance(result, GapAnalysis)
    assert result.verdict == "sufficient"
    assert "mem-1" in result.grounding_memory_ids
    
    # Verify hunt called
    mock_memory.hunt.assert_called_once()
    args = mock_memory.hunt.call_args
    assert "Deploy to prod" in args.kwargs["query"]

@pytest.mark.asyncio
async def test_detect_gaps_no_pipeline(mock_memory):
    """Test behavior when extraction pipeline is missing."""
    mock_memory._extraction = None
    detector = OmissionDetector(mock_memory)
    
    result = await detector.detect_gaps("Deploy")
    assert result.verdict == "insufficient"
    assert "Reasoning engine unavailable" in result.missing_info

@pytest.mark.asyncio
async def test_detect_gaps_llm_failure(mock_memory):
    """Test exception handling during LLM call."""
    mock_memory._extraction.client.chat.completions.create.side_effect = Exception("LLM connection failed")
    detector = OmissionDetector(mock_memory)
    
    result = await detector.detect_gaps("Deploy")
    assert result.verdict == "insufficient"
    assert "Error during analysis" in result.missing_info
