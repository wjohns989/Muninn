"""
test_v3_10_0_multivector.py
----------------------------
Phase 13 (v3.10.0) — Native ColBERT Multi-Vector Storage tests.

19 tests covering:
- Collection lifecycle (create, delete, recreate)
- Upsert / update / delete of multi-vector points
- Native MaxSim search (or centroid fallback when unavailable)
- Namespace & user-ID payload filters
- Score ordering (more relevant docs rank higher)
- Feature-flag gating
- Config defaults
"""

from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from muninn.store.multi_vector_store import MultiVectorStore, _MULTIVEC_AVAILABLE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def qdrant_client(tmp_path: Path):
    """Provide an in-process Qdrant client backed by a temp directory."""
    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(tmp_path / "qdrant_test"))
    yield client
    client.close()


@pytest.fixture()
def store(qdrant_client) -> MultiVectorStore:
    """A fresh MultiVectorStore for each test."""
    collection = f"test_multivec_{uuid.uuid4().hex[:8]}"
    s = MultiVectorStore(qdrant_client, collection_name=collection, dim=16)
    yield s


def _rand_vecs(n: int, dim: int = 16, seed: int = 42) -> np.ndarray:
    """Return L2-normalised random vectors, shape (n, dim)."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms > 0, norms, 1.0)


# ---------------------------------------------------------------------------
# 1. Collection creation
# ---------------------------------------------------------------------------


def test_multi_vector_collection_creation(store: MultiVectorStore):
    """ensure_collection() creates the collection on first call."""
    store.ensure_collection()
    assert store._initialized
    # Second call is idempotent
    store.ensure_collection()
    assert store._initialized


# ---------------------------------------------------------------------------
# 2. Upsert single document
# ---------------------------------------------------------------------------


def test_multi_vector_upsert_single_document(store: MultiVectorStore):
    """Upsert a single document and verify point count == 1."""
    vecs = _rand_vecs(5)
    point_id = store.upsert("mem_001", vecs)
    assert isinstance(point_id, str)
    assert store.count() == 1


# ---------------------------------------------------------------------------
# 3. Upsert multiple documents
# ---------------------------------------------------------------------------


def test_multi_vector_upsert_multiple_documents(store: MultiVectorStore):
    """Upserting N documents results in N points."""
    for i in range(6):
        store.upsert(f"mem_{i:03d}", _rand_vecs(4, seed=i))
    assert store.count() == 6


# ---------------------------------------------------------------------------
# 4. Update (re-upsert) existing document
# ---------------------------------------------------------------------------


def test_multi_vector_update_existing(store: MultiVectorStore):
    """Re-upserting the same memory_id overwrites the existing point."""
    vecs_v1 = _rand_vecs(3, seed=1)
    vecs_v2 = _rand_vecs(7, seed=2)
    store.upsert("mem_update", vecs_v1)
    assert store.count() == 1
    store.upsert("mem_update", vecs_v2)
    # Still exactly one point
    assert store.count() == 1


# ---------------------------------------------------------------------------
# 5. Delete
# ---------------------------------------------------------------------------


def test_multi_vector_delete(store: MultiVectorStore):
    """Deleting a memory removes its point."""
    store.upsert("mem_del", _rand_vecs(4))
    assert store.count() == 1
    ok = store.delete("mem_del")
    assert ok
    assert store.count() == 0


# ---------------------------------------------------------------------------
# 6. Search returns results with scores
# ---------------------------------------------------------------------------


def test_multi_vector_search_maxsim(store: MultiVectorStore):
    """search() returns (memory_id, score) tuples ordered by score."""
    # Index 3 documents
    store.upsert("doc_a", _rand_vecs(4, seed=10))
    store.upsert("doc_b", _rand_vecs(4, seed=20))
    store.upsert("doc_c", _rand_vecs(4, seed=30))

    query = _rand_vecs(2, seed=10)  # query similar to doc_a
    results = store.search(query, limit=3)

    assert len(results) > 0
    # All results are (str, float) tuples
    for mem_id, score in results:
        assert isinstance(mem_id, str)
        assert isinstance(score, float)
    # Results ordered descending
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# 7. Search on empty collection
# ---------------------------------------------------------------------------


def test_multi_vector_search_empty_collection(store: MultiVectorStore):
    """search() on an empty collection returns []."""
    store.ensure_collection()
    query = _rand_vecs(2)
    results = store.search(query, limit=5)
    assert results == []


# ---------------------------------------------------------------------------
# 8. Search with payload filters
# ---------------------------------------------------------------------------


def test_multi_vector_search_with_filters(store: MultiVectorStore):
    """Only documents matching the filter are returned."""
    store.upsert("ns_a_mem1", _rand_vecs(3, seed=1), metadata={"namespace": "alpha"})
    store.upsert("ns_b_mem1", _rand_vecs(3, seed=2), metadata={"namespace": "beta"})
    store.upsert("ns_a_mem2", _rand_vecs(3, seed=3), metadata={"namespace": "alpha"})

    query = _rand_vecs(2, seed=1)
    results = store.search(query, limit=5, filters={"namespace": "alpha"})

    returned_ids = {mid for mid, _ in results}
    assert "ns_b_mem1" not in returned_ids
    assert all(mid.startswith("ns_a_") for mid in returned_ids)


# ---------------------------------------------------------------------------
# 9. Fallback when multivec unavailable (centroid path)
# ---------------------------------------------------------------------------


def test_multi_vector_fallback_when_disabled(tmp_path: Path):
    """
    When _MULTIVEC_AVAILABLE is False (monkey-patched), the store uses the
    centroid fallback and still returns valid results.
    """
    import muninn.store.multi_vector_store as mvs_mod
    from qdrant_client import QdrantClient

    original = mvs_mod._MULTIVEC_AVAILABLE
    try:
        mvs_mod._MULTIVEC_AVAILABLE = False
        client = QdrantClient(path=str(tmp_path / "qdrant_fallback"))
        store = mvs_mod.MultiVectorStore(client, collection_name="fallback_test", dim=16)
        store._native_multivec = False  # force centroid path

        store.upsert("mem_fb", _rand_vecs(4, seed=7))
        assert store.count() == 1

        results = store.search(_rand_vecs(2, seed=7), limit=1)
        assert len(results) == 1
        assert results[0][0] == "mem_fb"
        client.close()
    finally:
        mvs_mod._MULTIVEC_AVAILABLE = original


# ---------------------------------------------------------------------------
# 10. Indexer initialisation
# ---------------------------------------------------------------------------


def test_multi_vector_indexer_init(store: MultiVectorStore):
    """MultiVectorStore initialises correctly with given dim."""
    assert store.dim == 16
    assert store.collection_name.startswith("test_multivec_")
    assert not store._initialized  # lazy: not yet created
    store.ensure_collection()
    assert store._initialized


# ---------------------------------------------------------------------------
# 11. Empty token vectors raise ValueError
# ---------------------------------------------------------------------------


def test_multi_vector_indexer_with_empty_vectors(store: MultiVectorStore):
    """Upserting a zero-row ndarray raises ValueError."""
    with pytest.raises(ValueError, match="at least one row"):
        store.upsert("mem_empty", np.empty((0, 16), dtype=np.float32))


# ---------------------------------------------------------------------------
# 12. Single token document
# ---------------------------------------------------------------------------


def test_multi_vector_indexer_with_single_token(store: MultiVectorStore):
    """A document with a single token vector indexes and retrieves correctly."""
    vecs = _rand_vecs(1, seed=99)
    store.upsert("mem_single", vecs)
    assert store.count() == 1
    results = store.search(vecs, limit=1)
    assert results[0][0] == "mem_single"


# ---------------------------------------------------------------------------
# 13. Large document (512 tokens)
# ---------------------------------------------------------------------------


def test_multi_vector_indexer_with_many_tokens(store: MultiVectorStore):
    """A 512-token document stores and can be retrieved."""
    vecs = _rand_vecs(512, seed=512)
    store.upsert("mem_long", vecs)
    assert store.count() == 1


# ---------------------------------------------------------------------------
# 14. Score ordering — relevant > irrelevant
# ---------------------------------------------------------------------------


def test_multi_vector_score_ordering(store: MultiVectorStore):
    """
    A document constructed from the same vectors as the query should score
    higher than an orthogonal document.
    """
    rng = np.random.default_rng(0)
    # Relevant: same vectors as query
    query_vecs = _rand_vecs(4, seed=0)
    # Irrelevant: orthogonalised via QR decomposition
    raw = rng.standard_normal((4, 16)).astype(np.float32)
    ortho, _ = np.linalg.qr(raw.T)
    irrel_vecs = ortho.T  # orthonormal rows
    irrel_vecs = irrel_vecs.astype(np.float32)

    store.upsert("relevant", query_vecs)
    store.upsert("irrelevant", irrel_vecs)

    results = store.search(query_vecs, limit=2)
    ids = [mid for mid, _ in results]
    # "relevant" should appear first (or at least score higher)
    assert ids[0] == "relevant"


# ---------------------------------------------------------------------------
# 15. Exact-match document has highest score
# ---------------------------------------------------------------------------


def test_multi_vector_score_exact_match(store: MultiVectorStore):
    """
    A document whose vectors exactly match the query vectors should return the
    maximum score from the collection.
    """
    q = _rand_vecs(4, seed=1)
    store.upsert("exact", q)
    store.upsert("other", _rand_vecs(4, seed=100))

    results = store.search(q, limit=2)
    assert len(results) >= 1
    # The exact-match document should be the top result
    assert results[0][0] == "exact"
    # Score should be positive
    assert results[0][1] > 0.0


# ---------------------------------------------------------------------------
# 16. get_vectors round-trip
# ---------------------------------------------------------------------------


def test_multi_vector_store_get_vectors(store: MultiVectorStore):
    """
    get_vectors() returns stored vectors of correct shape.
    For native multi-vec: shape == (n_tokens, dim).
    For centroid fallback: shape == (1, dim).
    """
    vecs = _rand_vecs(5, seed=42)
    store.upsert("mem_rv", vecs)
    retrieved = store.get_vectors("mem_rv")
    assert retrieved is not None
    assert retrieved.ndim == 2
    assert retrieved.shape[1] == 16


# ---------------------------------------------------------------------------
# 17. Namespace isolation
# ---------------------------------------------------------------------------


def test_multi_vector_namespace_isolation(store: MultiVectorStore):
    """Results scoped to namespace='ns1' should not include ns2 documents."""
    store.upsert("ns1_a", _rand_vecs(3, seed=1), metadata={"namespace": "ns1"})
    store.upsert("ns2_a", _rand_vecs(3, seed=2), metadata={"namespace": "ns2"})

    q = _rand_vecs(2, seed=1)
    r_ns1 = store.search(q, limit=5, filters={"namespace": "ns1"})
    r_ns2 = store.search(q, limit=5, filters={"namespace": "ns2"})

    ids_ns1 = {mid for mid, _ in r_ns1}
    ids_ns2 = {mid for mid, _ in r_ns2}
    assert "ns2_a" not in ids_ns1
    assert "ns1_a" not in ids_ns2


# ---------------------------------------------------------------------------
# 18. User isolation
# ---------------------------------------------------------------------------


def test_multi_vector_user_isolation(store: MultiVectorStore):
    """Results scoped to user_id='alice' should not include bob's documents."""
    store.upsert("alice_m", _rand_vecs(3, seed=3), metadata={"user_id": "alice"})
    store.upsert("bob_m", _rand_vecs(3, seed=4), metadata={"user_id": "bob"})

    q = _rand_vecs(2, seed=3)
    alice_results = store.search(q, limit=5, filters={"user_id": "alice"})
    bob_results = store.search(q, limit=5, filters={"user_id": "bob"})

    alice_ids = {mid for mid, _ in alice_results}
    bob_ids = {mid for mid, _ in bob_results}
    assert "bob_m" not in alice_ids
    assert "alice_m" not in bob_ids


# ---------------------------------------------------------------------------
# 19. Config defaults are sane
# ---------------------------------------------------------------------------


def test_multi_vector_config_defaults():
    """AdvancedConfig multi-vec defaults are correct (disabled by default)."""
    from muninn.core.config import AdvancedConfig

    cfg = AdvancedConfig()
    assert cfg.enable_colbert_multivec is False
    assert cfg.colbert_multivec_collection == "muninn_colbert_multivec"

    from muninn.core.feature_flags import FeatureFlags

    flags = FeatureFlags()
    assert flags.colbert_multivec is False
    assert flags.temporal_query_expansion is False
