import asyncio
from unittest.mock import MagicMock

from muninn.core.memory import MuninnMemory
from muninn.core.types import ExtractionResult, MemoryRecord, MemoryType, Provenance
from muninn.dedup.semantic_dedup import DedupResult


def _record(memory_id: str, namespace: str, user_id: str):
    return MemoryRecord(
        id=memory_id,
        content=f"memory {memory_id}",
        memory_type=MemoryType.EPISODIC,
        provenance=Provenance.AUTO_EXTRACTED,
        namespace=namespace,
        metadata={"user_id": user_id},
    )


def test_add_passes_namespace_and_user_filters_to_dedup_search():
    memory = MuninnMemory()
    memory._initialized = True
    memory._user_scope_migration_complete = True

    async def _extract(_content):
        return ExtractionResult()

    memory._extract = _extract
    memory._embed = lambda _text: [0.1, 0.2, 0.3]
    memory._vectors = MagicMock()
    memory._vectors.count.return_value = 1

    memory._dedup = MagicMock()
    memory._dedup.check_duplicate.return_value = DedupResult(
        is_duplicate=True,
        existing_memory_id="mem-1",
        similarity=0.99,
    )
    memory._conflict_detector = None

    result = asyncio.run(memory.add("same memory", namespace="project-a", user_id="user-1"))

    assert result["event"] == "DEDUP_SKIP"
    assert memory._dedup.check_duplicate.call_args.kwargs["filters"] == {
        "namespace": "project-a",
        "user_id": "user-1",
    }


def test_conflict_prefilter_candidates_are_scoped_to_namespace_and_user():
    memory = MuninnMemory()
    memory._initialized = True

    async def _extract(_content):
        return ExtractionResult()

    memory._extract = _extract
    memory._embed = lambda _text: [0.1, 0.2, 0.3]

    memory._vectors = MagicMock()
    memory._vectors.count.return_value = 1
    memory._vectors.search.side_effect = [
        [("mem-same", 0.95), ("mem-other", 0.94)],
        [("mem-same", 0.95)],
    ]
    memory._vectors.upsert = MagicMock()

    memory._metadata = MagicMock()
    memory._metadata.get_by_ids.return_value = [
        _record("mem-same", namespace="project-a", user_id="user-1"),
        _record("mem-other", namespace="project-b", user_id="user-2"),
    ]
    memory._metadata.add = MagicMock()

    memory._conflict_detector = MagicMock()
    memory._conflict_detector.detect_conflicts.return_value = []
    memory._conflict_resolver = MagicMock()

    memory._graph = MagicMock()
    memory._bm25 = MagicMock()
    memory._dedup = None

    result = asyncio.run(memory.add("new memory", namespace="project-a", user_id="user-1"))

    assert result["event"] == "ADD"
    detect_args = memory._conflict_detector.detect_conflicts.call_args.args
    assert len(detect_args[1]) == 1
    assert detect_args[1][0].id == "mem-same"


def test_conflict_prefilter_excludes_candidates_without_user_scope():
    memory = MuninnMemory()
    memory._initialized = True
    memory._user_scope_migration_complete = True

    async def _extract(_content):
        return ExtractionResult()

    memory._extract = _extract
    memory._embed = lambda _text: [0.1, 0.2, 0.3]

    memory._vectors = MagicMock()
    memory._vectors.count.return_value = 1
    memory._vectors.search.side_effect = [[("legacy", 0.95)], []]
    memory._vectors.upsert = MagicMock()

    legacy = _record("legacy", namespace="project-a", user_id="user-1")
    legacy.metadata = {}

    memory._metadata = MagicMock()
    memory._metadata.get_by_ids.return_value = [legacy]
    memory._metadata.add = MagicMock()

    memory._conflict_detector = MagicMock()
    memory._conflict_detector.detect_conflicts.return_value = []
    memory._conflict_resolver = MagicMock()

    memory._graph = MagicMock()
    memory._bm25 = MagicMock()
    memory._dedup = None

    result = asyncio.run(memory.add("new memory", namespace="project-a", user_id="user-1"))

    assert result["event"] == "ADD"
    assert memory._conflict_detector.detect_conflicts.call_count == 0


def test_run_user_scope_migration_updates_metadata_and_vector_payload():
    memory = MuninnMemory()
    memory._metadata = MagicMock()
    memory._vectors = MagicMock()

    legacy = _record("legacy-1", namespace="project-a", user_id="user-1")
    legacy.metadata = {}
    scoped = _record("scoped-1", namespace="project-a", user_id="user-1")

    memory._metadata.get_user_scope_backfill_failures.return_value = []
    memory._metadata.count_user_scope_backfill_failures.return_value = 0
    memory._metadata.get_missing_user_id_records.side_effect = [[legacy], []]
    memory._metadata.count_missing_user_id.return_value = 0

    stats = memory._run_user_scope_migration(default_user_id="global_user", batch_size=500, max_batches=5)

    memory._metadata.update.assert_called_once_with("legacy-1", metadata={"user_id": "global_user"})
    memory._vectors.set_payload.assert_called_once_with("legacy-1", {"user_id": "global_user"})
    memory._metadata.set_meta.assert_called_once_with("user_scope_migration_complete", "1")
    assert stats["complete"] == 1


def test_conflict_prefilter_stays_strict_until_migration_complete():
    memory = MuninnMemory()
    memory._initialized = True
    memory._user_scope_migration_complete = False

    async def _extract(_content):
        return ExtractionResult()

    memory._extract = _extract
    memory._embed = lambda _text: [0.1, 0.2, 0.3]

    memory._vectors = MagicMock()
    memory._vectors.count.return_value = 1
    memory._vectors.search.side_effect = [[("legacy", 0.95)], []]
    memory._vectors.upsert = MagicMock()

    legacy = _record("legacy", namespace="project-a", user_id="user-1")
    legacy.metadata = {}

    memory._metadata = MagicMock()
    memory._metadata.get_by_ids.return_value = [legacy]
    memory._metadata.add = MagicMock()

    memory._conflict_detector = MagicMock()
    memory._conflict_detector.detect_conflicts.return_value = []
    memory._conflict_resolver = MagicMock()

    memory._graph = MagicMock()
    memory._bm25 = MagicMock()
    memory._dedup = None

    result = asyncio.run(memory.add("new memory", namespace="project-a", user_id="user-1"))

    assert result["event"] == "ADD"
    assert memory._conflict_detector.detect_conflicts.call_count == 0

