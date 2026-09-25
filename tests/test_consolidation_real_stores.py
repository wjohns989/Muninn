"""Contract test: a full consolidation cycle against real SQLite, Qdrant, Kuzu and BM25.

Mocked stores accept any call signature, which hid mismatches such as passing a
list to VectorStore.delete or unknown keywords to VectorStore.upsert. This test
runs every phase end to end on the real backends.
"""

import time

import pytest

from muninn.consolidation.daemon import ConsolidationDaemon
from muninn.core.config import ConsolidationConfig
from muninn.core.types import MemoryRecord, MemoryType, Provenance
from muninn.retrieval.bm25 import BM25Index
from muninn.store.graph_store import GraphStore
from muninn.store.sqlite_metadata import SQLiteMetadataStore
from muninn.store.vector_store import VectorStore

DAY = 86400.0
VECTOR = [0.1, 0.2, 0.3, 0.4]


@pytest.fixture
def stores(tmp_path):
    metadata = SQLiteMetadataStore(tmp_path / "meta.db")
    vectors = VectorStore(tmp_path / "qdrant", embedding_dims=4)
    graph = GraphStore(tmp_path / "kuzu")
    bm25 = BM25Index()
    yield metadata, vectors, graph, bm25
    vectors.close()


def _add(stores, record, vector):
    metadata, vectors, graph, bm25 = stores
    metadata.add(record)
    user_id = record.metadata["user_id"]
    vectors.upsert(record.id, vector, {"user_id": user_id, "namespace": record.namespace})
    graph.add_memory_node(record.id, record.content[:200], user_id=user_id, namespace=record.namespace)
    bm25.add(record.id, record.content, user_id=user_id, namespace=record.namespace)


@pytest.mark.asyncio
async def test_full_cycle_on_real_stores(stores, tmp_path):
    metadata, vectors, graph, bm25 = stores
    now = time.time()
    scope = {"user_id": "u1"}
    # A near-duplicate pair (identical vectors) that the merge phase will find itself.
    _add(stores, MemoryRecord(id="dup-a", content="deploys run on Fridays", vector_id="dup-a",
                              importance=0.9, metadata=dict(scope)), VECTOR)
    _add(stores, MemoryRecord(id="dup-b", content="rollbacks use blue-green", vector_id="dup-b",
                              importance=0.2, metadata=dict(scope)), VECTOR)
    # A stale, redundant memory that decay should archive.
    _add(stores, MemoryRecord(id="stale", content="restated trivia", vector_id="stale",
                              provenance=Provenance.INGESTED, novelty_score=0.02,
                              created_at=now - 3650 * DAY, metadata=dict(scope)), [0.9, 0.1, 0.0, 0.0])
    # An expired working memory that decay should delete.
    _add(stores, MemoryRecord(id="scratch", content="scratch pad", vector_id="scratch",
                              memory_type=MemoryType.WORKING, created_at=now - 2 * DAY,
                              metadata=dict(scope)), [0.0, 0.0, 1.0, 0.0])

    daemon = ConsolidationDaemon(
        config=ConsolidationConfig(),
        metadata=metadata,
        vectors=vectors,
        graph=graph,
        bm25=bm25,
        embed_fn=lambda text: VECTOR,
        images_dir=tmp_path / "images",
    )

    result = await daemon.run_cycle()

    assert "error" not in result, result.get("error")
    phases = result["phases"]
    assert phases["merge"]["merged"] == 1
    assert phases["decay"]["expired"] == 1
    assert metadata.get("scratch") is None
    assert metadata.get("stale").archived is True
    # Decay re-scores importance before merge, so either record may survive.
    pair = {mid: metadata.get(mid) for mid in ("dup-a", "dup-b")}
    archived = [r for r in pair.values() if r.archived]
    assert len(archived) == 1
    survivor = next(r for r in pair.values() if not r.archived)
    assert archived[0].parent_id == survivor.id
    assert "rollbacks use blue-green" in survivor.content and "Fridays" in survivor.content
    # Archived and deleted memories are gone from recall; the survivor is findable by keyword.
    live_ids = {memory_id for memory_id, _ in vectors.search(VECTOR, 10, filters={"archived": False})}
    assert live_ids.isdisjoint({archived[0].id, "stale", "scratch"})
    assert survivor.id in live_ids
    assert [doc for doc, _ in bm25.search("Fridays", user_id="u1")] == [survivor.id]
    assert [doc for doc, _ in bm25.search("rollbacks", user_id="u1")] == [survivor.id]
