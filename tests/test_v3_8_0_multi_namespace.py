import pytest
import asyncio
import kuzu
from pathlib import Path
from unittest.mock import MagicMock
from muninn.core.types import MemoryRecord, MemoryType
from muninn.store.graph_store import GraphStore
from muninn.retrieval.bm25 import BM25Index
from muninn.consolidation.daemon import ConsolidationDaemon
from muninn.core.config import ConsolidationConfig

@pytest.fixture
def mock_stores(tmp_path):
    import uuid
    db_path = tmp_path / f"test_graph_{uuid.uuid4()}.db"
    graph = GraphStore(db_path)
    bm25 = BM25Index()
    return graph, bm25

def test_graph_store_isolation_logic(mock_stores):
    graph, _ = mock_stores

    # 1. Add memories for different users
    graph.add_memory_node("m1", "User A secret info about Python", user_id="user_a", namespace="default")
    graph.add_memory_node("m2", "User B secret info about Java", user_id="user_b", namespace="default")

    # 2. Test Strategy 2: Fallback to summary keyword match (Regex)
    # This proves the isolation_clause works on the Memory table directly
    results_a = graph.search_memories("secret", user_id="user_a", namespaces=["default"])
    assert len(results_a) >= 1
    assert results_a[0]["id"] == "m1"

    # 3. Test Strategy 1: Entity-based search (MENTIONS) logic
    # v3.9.0: Entities are now scoped by user_id/namespace
    graph.add_entity("Python", "language", user_id="user_a", namespace="default")
    graph.link_memory_to_entity("m1", "Python", user_id="user_a", namespace="default")

    # Search for Python as User A
    results_python = graph.search_memories("Python", user_id="user_a", namespaces=["default"])
    assert len(results_python) >= 1
    assert any(r["id"] == "m1" for r in results_python)

    # 4. LEAK TEST: User B searching for Python should see NOTHING
    # Entity "Python" belongs to user_a/default scope, not user_b/default
    results_leak = graph.search_memories("Python", user_id="user_b", namespaces=["default"])
    assert not any(r["id"] == "m1" for r in results_leak)

def test_bm25_index_isolation(mock_stores):
    _, bm25 = mock_stores
    
    bm25.add("d1", "Python programming", user_id="u1", namespace="code")
    bm25.add("d2", "Python snake", user_id="u2", namespace="nature")
    
    # Search as u1
    res1 = bm25.search("Python", user_id="u1", namespaces=["code"])
    assert len(res1) == 1
    assert res1[0][0] == "d1"
    
    # Leak test: u1 searching u2's data
    res_leak = bm25.search("Python", user_id="u1", namespaces=["nature"])
    assert len(res_leak) == 0

@pytest.mark.asyncio
async def test_consolidation_merge_isolation():
    # Setup mock infrastructure
    metadata = MagicMock()
    vectors = MagicMock()
    graph = MagicMock()
    bm25 = MagicMock()
    config = ConsolidationConfig(enabled=True)
    
    daemon = ConsolidationDaemon(config, metadata, vectors, graph, bm25)
    
    # Create two records for DIFFERENT users that are "similar"
    r1 = MemoryRecord(id="r1", content="user a info", user_id="user1", namespace="ns1", memory_type=MemoryType.EPISODIC, vector_id="v1")
    r2 = MemoryRecord(id="r2", content="user b info", user_id="user2", namespace="ns1", memory_type=MemoryType.EPISODIC, vector_id="v2")
    
    metadata.get_for_consolidation.return_value = [r1, r2]
    metadata.get.side_effect = lambda x: r1 if x == "r1" else r2
    
    # Mock search to return EMPTY for isolated user-based search
    vectors.search.return_value = [] 
    vectors.get_vectors.return_value = {"v1": [0.1]*1536}
    
    results = await daemon._phase_merge()
    assert results["merged"] == 0
