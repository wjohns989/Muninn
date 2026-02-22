"""
Tests for Temporal Knowledge Graph capabilities.
"""

import time
import pytest
from muninn.store.graph_store import GraphStore
from muninn.advanced.temporal_kg import TemporalKnowledgeGraph

@pytest.fixture
def graph_store(tmp_path):
    return GraphStore(tmp_path / "kuzu_test")

@pytest.fixture
def temporal_kg(graph_store):
    tkg = TemporalKnowledgeGraph(graph_store)
    tkg.initialize_schema()
    return tkg

def test_temporal_fact_lifecycle(temporal_kg):
    now = time.time()
    yesterday = now - 86400
    tomorrow = now + 86400
    
    # "Server A" was "Active" from yesterday until now
    assert temporal_kg.add_temporal_fact(
        "Server A", "status", "Active",
        valid_start=yesterday,
        source_memory="test_source",
        valid_end=now
    )
    
    # "Server A" is "Maintenance" from now until tomorrow
    assert temporal_kg.add_temporal_fact(
        "Server A", "status", "Maintenance",
        valid_start=now,
        source_memory="test_source",
        valid_end=tomorrow
    )
    
    # Query past
    past_facts = temporal_kg.query_valid_at(yesterday + 100)
    assert len(past_facts) == 1
    assert past_facts[0]["object"] == "Active"
    
    # Query future
    future_facts = temporal_kg.query_valid_at(now + 100)
    assert len(future_facts) == 1
    assert future_facts[0]["object"] == "Maintenance"

def test_snapshot_diff(temporal_kg):
    t0 = 1000
    t1 = 2000
    t2 = 3000
    
    temporal_kg.add_temporal_fact("User", "uses", "V1", t0, "source1", t1)
    temporal_kg.add_temporal_fact("User", "uses", "V2", t1, "source2", t2)
    
    diff = temporal_kg.snapshot_diff(t0 + 1, t1 + 1)
    
    assert len(diff["removed"]) == 1
    assert diff["removed"][0]["object"] == "V1"
    
    assert len(diff["added"]) == 1
    assert diff["added"][0]["object"] == "V2"
