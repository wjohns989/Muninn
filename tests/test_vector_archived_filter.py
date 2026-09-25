"""Regression: the default archived=False search filter must not hide un-archived vectors."""

import pytest

from muninn.store.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    vs = VectorStore(tmp_path / "vectors", embedding_dims=4)
    yield vs
    vs._get_client().close()


def test_archived_false_matches_points_without_archived_payload(store):
    store.upsert("plain", [0.1, 0.2, 0.3, 0.4], {"user_id": "u1", "namespace": "global"})

    hits = store.search([0.1, 0.2, 0.3, 0.4], 5, filters={"user_id": "u1", "archived": False})

    assert [memory_id for memory_id, _ in hits] == ["plain"]


def test_archived_false_still_excludes_archived_points(store):
    store.upsert("live", [0.1, 0.2, 0.3, 0.4], {"user_id": "u1", "archived": False})
    store.upsert("old", [0.1, 0.2, 0.3, 0.41], {"user_id": "u1", "archived": True})

    hits = store.search([0.1, 0.2, 0.3, 0.4], 5, filters={"user_id": "u1", "archived": False})

    assert [memory_id for memory_id, _ in hits] == ["live"]
