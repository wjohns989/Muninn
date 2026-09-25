"""Tests for bounded, scope-safe vector clustering."""

from unittest.mock import MagicMock

import pytest

from muninn.core.types import MemoryRecord, MemoryType
from muninn.optimization.clustering import VectorClusterEngine


def create_mock_record(
    memory_id: str,
    content: str,
    *,
    namespace: str = "global",
    project: str = "project-a",
    user_id: str = "user-a",
) -> MemoryRecord:
    return MemoryRecord(
        id=memory_id,
        content=content,
        memory_type=MemoryType.EPISODIC,
        created_at=1000.0,
        ingested_at=1000.0,
        metadata={"user_id": user_id},
        namespace=namespace,
        project=project,
    )


@pytest.fixture
def mock_memory():
    memory = MagicMock()
    memory._metadata.get_all = MagicMock()
    memory._metadata.get_by_ids = MagicMock()
    memory._vectors.get_vectors = MagicMock()
    memory._vectors.search = MagicMock()
    return memory


@pytest.mark.asyncio
async def test_clustering_batches_vectors_and_scopes_neighbor_search(mock_memory):
    engine = VectorClusterEngine(mock_memory)
    leader = create_mock_record("lead-1", "Leader")
    members = [leader] + [create_mock_record(f"follow-{i}", "Follower") for i in range(4)]
    mock_memory._metadata.get_all.return_value = [leader]
    mock_memory._vectors.get_vectors.return_value = {leader.id: [0.1] * 8}
    mock_memory._vectors.search.return_value = [(record.id, 0.9) for record in members]
    mock_memory._metadata.get_by_ids.return_value = members

    clusters = await engine.find_episodic_clusters(min_cluster_size=5)

    assert len(clusters) == 1
    assert clusters[0]["memory_ids"] == [record.id for record in members]
    mock_memory._vectors.get_vectors.assert_called_once_with([leader.id])
    filters = mock_memory._vectors.search.call_args.kwargs["filters"]
    assert filters == {
        "memory_type": "episodic",
        "namespace": "global",
        "project": "project-a",
        "user_id": "user-a",
    }


@pytest.mark.asyncio
async def test_clustering_excludes_stale_cross_scope_vector_hits(mock_memory):
    engine = VectorClusterEngine(mock_memory)
    leader = create_mock_record("lead-1", "Leader")
    same_scope = [leader] + [create_mock_record(f"same-{i}", "Same") for i in range(3)]
    cross_scope = create_mock_record("other-project", "Other", project="project-b")
    mock_memory._metadata.get_all.return_value = [leader]
    mock_memory._vectors.get_vectors.return_value = {leader.id: [0.1] * 8}
    mock_memory._vectors.search.return_value = [
        *((record.id, 0.9) for record in same_scope),
        (cross_scope.id, 0.99),
    ]
    mock_memory._metadata.get_by_ids.return_value = same_scope + [cross_scope]

    clusters = await engine.find_episodic_clusters(min_cluster_size=5)

    assert clusters == []
