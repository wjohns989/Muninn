"""
Tests for Phase 22: Temporal Contradiction Detection
Verifies that the ConsolidationDaemon's MERGE phase correctly identifies
chronological contradictions via LLM synthesis, preventing merges and shadowing
the outdated episodic memory.
"""

import time
import pytest
from unittest.mock import MagicMock, patch, ANY

from muninn.core.types import MemoryRecord, MemoryType
from muninn.consolidation.daemon import ConsolidationDaemon
from muninn.extraction.models import TemporalContradictionResolution


@pytest.fixture
def mock_daemon():
    # Setup mocks
    metadata = MagicMock()
    vectors = MagicMock()
    graph = MagicMock()
    bm25 = MagicMock()
    config = MagicMock()
    
    daemon = ConsolidationDaemon(
        config=config,
        metadata=metadata,
        vectors=vectors,
        graph=graph,
        bm25=bm25,
        embed_fn=lambda x: [0.1],
        extractor=MagicMock()
    )
    
    # Enable extractor with a mock instructor route
    extractor = daemon.extractor
    extractor._instructor_routes_by_profile = {"balanced": [("test_route", MagicMock(is_available=True))]}
    extractor.model_profile = "balanced"
    
    return daemon


@pytest.mark.asyncio
async def test_v3_22_1_contradiction_detection_prevents_merge(mock_daemon):
    now = time.time()
    
    # Create two contradictory memories
    older_memory = MemoryRecord(
        id="mem_old",
        content="The system architecture uses SQLite.",
        memory_type=MemoryType.EPISODIC,
        created_at=now - 1000,
        user_id="user_1",
        namespace="global",
        importance=0.8,
        vector_id="v_old",
        access_count=1,
    )
    newer_memory = MemoryRecord(
        id="mem_new",
        content="The system architecture uses Postgres.",
        memory_type=MemoryType.EPISODIC,
        created_at=now,
        user_id="user_1",
        namespace="global",
        importance=0.8,
        vector_id="v_new",
        access_count=1,
    )
    
    # Mock metadata store finding these for consolidation
    mock_daemon.metadata.get_for_consolidation.return_value = [older_memory, newer_memory]
    
    # Mock find_merge_candidates to return them as highly similar candidates
    with patch("muninn.consolidation.daemon.find_merge_candidates") as mock_find_candidates:
        # Return tuples of (primary_id, secondary_id, score)
        mock_find_candidates.return_value = [("mem_new", "mem_old", 0.95)]
        
        # Mock metadata.get to return the actual records
        def mock_get(mem_id):
            if mem_id == "mem_old": return older_memory
            if mem_id == "mem_new": return newer_memory
            return None
        mock_daemon.metadata.get.side_effect = mock_get
        
        # Mock the LLM synthesis to confirm the contradiction
        with patch("muninn.extraction.temporal_synthesis.synthesize_temporal_contradiction") as mock_synth:
            mock_synth.return_value = TemporalContradictionResolution(
                contradiction_confirmed=True,
                superseding_fact="The system architecture uses Postgres.",
                outdated_fact="The system architecture uses SQLite.",
                explanation="Postgres replaces SQLite over time."
            )
            
            result = await mock_daemon._phase_merge()
            
            # The MERGE phase should have aborted the standard merge
            assert result["merged"] == 0
            
            # Verify that the LLM was called
            mock_synth.assert_called_once()
            
            # Verify that the older memory (mem_old) was shadowed / archived
            # It should have updated metadata with superseded_by and importance drop
            # Use ANY for the metadata check to debug what actually got updated
            mock_daemon.metadata.update_metadata.assert_called_with("mem_old", ANY)
            
            # Print to see what's happening
            print(f"older_memory.metadata in test: {older_memory.metadata}")
            print(f"Update call args: {mock_daemon.metadata.update_metadata.call_args}")
            
            # If the dict is copied, we can check the call args directly
            updated_dict = mock_daemon.metadata.update_metadata.call_args[0][1]
            assert updated_dict["temporal_shadowed_by"] == "mem_new"
            assert "superseded_at" in updated_dict
            
            assert older_memory.importance == 0.08000000000000002 # 0.8 * 0.1
            
            # Verify vectors/BM25 drops 
            mock_daemon.vectors.delete.assert_called_with(["mem_old"])
            mock_daemon.bm25.remove.assert_called_with("mem_old")
            
            # Verify graph shadow call 
            mock_daemon.graph.shadow_memory_edges.assert_called_with(
                memory_id="mem_old",
                superseded_at=newer_memory.created_at
            )
