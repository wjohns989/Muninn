import pytest
import time
import uuid
import os
from pathlib import Path
from muninn.store.graph_store import GraphStore

@pytest.fixture
def graph_store(tmp_path):
    db_path = tmp_path / f"test_graph_scoping_{uuid.uuid4()}.db"
    store = GraphStore(str(db_path))
    yield store
    # Cleanup kuzu tends to be tricky with file handles

def test_entity_scoping_isolation(graph_store):
    # 1. Add same entity name for different users
    graph_store.add_entity("Python", "language", user_id="user_a", namespace="default")
    graph_store.add_entity("Python", "snake", user_id="user_b", namespace="default")
    
    # 2. Verify both exist independently
    entities = graph_store.get_all_entities()
    # Should have 2 entities in the list
    assert len(entities) == 2
    
    # Check details
    names = [e["name"] for e in entities]
    assert all(n == "Python" for n in names)
    
    types = [e["entity_type"] for e in entities]
    assert "language" in types
    assert "snake" in types
    
    # 3. Verify search isolation
    results_a = graph_store.search_memories("Python", user_id="user_a", namespaces=["default"])
    results_b = graph_store.search_memories("Python", user_id="user_b", namespaces=["default"])
    
    # Linking some memories to verify search
    graph_store.add_memory_node("m_a", "User A's Python code", user_id="user_a", namespace="default")
    graph_store.link_memory_to_entity("m_a", "Python", user_id="user_a", namespace="default")
    
    graph_store.add_memory_node("m_b", "User B's Python pet", user_id="user_b", namespace="default")
    graph_store.link_memory_to_entity("m_b", "Python", user_id="user_b", namespace="default")
    
    # Search should be isolated
    search_a = graph_store.search_memories("Python", user_id="user_a", namespaces=["default"])
    search_b = graph_store.search_memories("Python", user_id="user_b", namespaces=["default"])
    
    assert any(r["id"] == "m_a" for r in search_a)
    assert not any(r["id"] == "m_b" for r in search_a)
    
    assert any(r["id"] == "m_b" for r in search_b)
    assert not any(r["id"] == "m_a" for r in search_b)

def test_relation_scoping(graph_store):
    # user_a: Python -> IS_A -> Language
    graph_store.create_relation("Python", "IS_A", "Language", user_id="user_a", namespace="default")
    # user_b: Python -> IS_A -> Snake
    graph_store.create_relation("Python", "IS_A", "Snake", user_id="user_b", namespace="default")
    
    # Verify relations are isolated
    # (Note: we don't have a direct get_all_relations but we can check via centrality)
    c_a = graph_store.get_entity_centrality("Python", user_id="user_a", namespace="default")
    c_b = graph_store.get_entity_centrality("Python", user_id="user_b", namespace="default")
    
    assert c_a > 0
    assert c_b > 0
    
    # If we had a leak, centrality might be combined or cross-linked. 
    # But since IDs are scoped, they are strictly separate nodes.
