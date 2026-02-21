"""
Tests for Phase 22: Temporal Knowledge Graph & Shadowing (Schema)
Verifies that VALID_DURING edges are properly created, queried, and shadowed.
"""

import os
import time
import shutil
import pytest

from muninn.store.graph_store import GraphStore
from muninn.advanced.temporal_kg import TemporalKnowledgeGraph


@pytest.fixture
def temp_graph_store(tmp_path):
    # Set up a temporary Kuzu DB
    db_path = tmp_path / "test_kuzu"
    store = GraphStore(db_path)
    yield store
    # Cleanup
    try:
        if store._db:
            store._db = None
        shutil.rmtree(db_path, ignore_errors=True)
    except Exception:
        pass


def test_v3_22_0_temporal_schema_init_and_add(temp_graph_store):
    tkg = TemporalKnowledgeGraph(temp_graph_store)
    tkg.initialize_schema()
    
    # Add a temporal fact
    now = time.time()
    tkg.add_temporal_fact(
        subject="AgentLoki",
        predicate="uses_database",
        obj="SQLite",
        valid_start=now - 86400, # 1 day ago
        source_memory="mem_123"
    )
    
    # Query it as currently valid (now)
    facts_now = tkg.query_valid_at(now)
    assert len(facts_now) == 1
    assert facts_now[0]["subject"] == "AgentLoki"
    assert facts_now[0]["predicate"] == "uses_database"
    assert facts_now[0]["object"] == "SQLite"
    assert facts_now[0]["valid_end"] is None


def test_v3_22_0_shadow_memory_edges(temp_graph_store):
    tkg = TemporalKnowledgeGraph(temp_graph_store)
    tkg.initialize_schema()
    
    now = time.time()
    # Fact 1 from older memory
    tkg.add_temporal_fact(
        subject="System",
        predicate="version",
        obj="v1",
        valid_start=now - 200000,
        source_memory="mem_old"
    )
    
    # Shadow it as of 'now'
    res = tkg.shadow_memory_edges(memory_id="mem_old", superseded_at=now)
    assert res is True
    
    # Query at 'now' should not return it, since its end_time is 'now'
    # Wait, query_valid_at checks if start_time <= ts AND end_time >= ts
    # So technically at EXACTLY now it might still be valid, but typically we query slightly after 
    # Let's query slightly after 'now'
    facts_after = tkg.query_valid_at(now + 1)
    assert len(facts_after) == 0
    
    # Query in the past should return it
    facts_past = tkg.query_valid_at(now - 100000)
    assert len(facts_past) == 1
    assert facts_past[0]["object"] == "v1"

