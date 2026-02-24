import pytest
import time
import uuid
import asyncio

from muninn.core.memory import MuninnMemory
from muninn.store.sqlite_metadata import SQLiteMetadataStore
from muninn.core.types import MemoryRecord, MemoryType, Provenance
from muninn.scoring.importance import calculate_recency
from muninn.scoring.elo import elo_to_half_life_multiplier, INITIAL_ELO

@pytest.fixture
def temp_metadata_store(tmp_path):
    db_path = str(tmp_path / "test_elo.db")
    store = SQLiteMetadataStore(db_path)
    return store

@pytest.fixture
def mock_memory(temp_metadata_store):
    memory = MuninnMemory()
    memory._initialized = True
    memory._metadata = temp_metadata_store
    return memory

def create_mock_record(content="Hello"):
    return MemoryRecord(
        id=str(uuid.uuid4()),
        content=content,
        memory_type=MemoryType.EPISODIC,
        provenance=Provenance.USER_EXPLICIT,
        created_at=time.time(),
        metadata={}
    )

@pytest.mark.asyncio
async def test_elo_baseline_initialization(temp_metadata_store):
    """Test that new memories get the baseline Elo rating of 1200."""
    record = create_mock_record("The capital of France is Paris.")
    memory_id = temp_metadata_store.add(record)
    
    # Retrieve the record and check its metadata
    retrieved_record = temp_metadata_store.get(memory_id)
    assert retrieved_record is not None
    assert retrieved_record.metadata.get("elo_rating") == INITIAL_ELO

@pytest.mark.asyncio
async def test_record_retrieval_feedback_updates_elo(mock_memory):
    """Test that SNIPS feedback modifies the Elo rating correctly."""
    record = create_mock_record("Water boils at 100 degrees.")
    memory_id = mock_memory._metadata.add(record)
    
    # Send a positive outcome (e.g. user clicked or used it)
    await mock_memory.record_retrieval_feedback(
        query="At what temperature does water boil?",
        memory_id=memory_id,
        outcome=1.0,
        user_id="test_user"
    )
    
    # Check updated Elo rating
    retrieved_record = mock_memory._metadata.get(memory_id)
    new_elo = retrieved_record.metadata.get("elo_rating")
    assert new_elo > INITIAL_ELO, f"Expected Elo to increase from {INITIAL_ELO}, got {new_elo}"
    
    # Send a negative outcome
    await mock_memory.record_retrieval_feedback(
        query="Unrelated query",
        memory_id=memory_id,
        outcome=0.0,
        user_id="test_user"
    )
    
    # Elo should decrease from its new peak
    retrieved_record2 = mock_memory._metadata.get(memory_id)
    newer_elo = retrieved_record2.metadata.get("elo_rating")
    assert newer_elo < new_elo, f"Expected Elo to decrease from {new_elo}, got {newer_elo}"

def test_elo_half_life_multiplier():
    """Test that the mapping to half-life works based on Elo."""
    # Baseline
    assert elo_to_half_life_multiplier(1200) == 1.0
    
    # Above average should have multiplier > 1.0
    assert elo_to_half_life_multiplier(1600) > 1.0
    
    # Below average should have multiplier < 1.0
    assert elo_to_half_life_multiplier(800) < 1.0

def test_calculate_recency_with_elo():
    """Test that calculate_recency scales with Elo rating."""
    created_at = time.time() - 86400 * 7  # 7 days old
    
    # Standard 7-day half life -> ~0.5 score
    standard_recency = calculate_recency(created_at, half_life_days=7.0, elo_rating=1200)
    
    # High Elo -> > 7.0 half life -> higher recency score
    high_elo_recency = calculate_recency(created_at, half_life_days=7.0, elo_rating=1600)
    assert high_elo_recency > standard_recency
    
    # Low Elo -> < 7.0 half life -> lower recency score
    low_elo_recency = calculate_recency(created_at, half_life_days=7.0, elo_rating=800)
    assert low_elo_recency < standard_recency
