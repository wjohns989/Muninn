from contextlib import nullcontext
from types import SimpleNamespace

import asyncio
import pytest

from muninn.core.ingestion_manager import IngestionManager
from muninn.core.types import ExtractionResult, MemoryType, Provenance


class _VectorsStub:
    def count(self) -> int:
        return 0


class _OtelStub:
    def add_event(self, *args, **kwargs):
        return None

    def maybe_content(self, content: str) -> str:
        return content

    def span(self, *args, **kwargs):
        return nullcontext()


class _MemoryStub:
    def __init__(self):
        self._otel = _OtelStub()
        self._vectors = _VectorsStub()
        self._dedup = None
        self._conflict_detector = None
        self._conflict_resolver = None
        self._metadata = SimpleNamespace()
        self.config = SimpleNamespace(
            extraction=SimpleNamespace(
                runtime_model_profile="low_latency",
                model_profile="balanced",
            )
        )
        self.extract_called = 0
        self.embed_called = 0

    async def _extract_with_profile(self, content: str, model_profile: str):
        self.extract_called += 1
        return ExtractionResult(
            summary=content[:10],
            entities=[],
            relations=[],
        )

    def _extract_entity_names(self, extraction: ExtractionResult):
        return []

    async def _embed(self, content: str):
        self.embed_called += 1
        return [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_ingestion_manager_can_skip_extraction_via_metadata_flag():
    memory = _MemoryStub()
    manager = IngestionManager(memory)

    result = await manager.process_add(
        content="bulk imported text",
        user_id="global_user",
        agent_id=None,
        metadata={"muninn_skip_extraction": True},
        namespace="global",
        memory_type=MemoryType.EPISODIC,
        provenance=Provenance.INGESTED,
        scope="project",
    )

    assert result["event"] == "PROCESS_COMPLETE"
    assert memory.extract_called == 0
    assert memory.embed_called == 1


@pytest.mark.asyncio
async def test_ingestion_manager_runs_extraction_when_skip_flag_absent():
    memory = _MemoryStub()
    manager = IngestionManager(memory)

    result = await manager.process_add(
        content="normal imported text",
        user_id="global_user",
        agent_id=None,
        metadata={},
        namespace="global",
        memory_type=MemoryType.EPISODIC,
        provenance=Provenance.INGESTED,
        scope="project",
    )

    assert result["event"] == "PROCESS_COMPLETE"
    assert memory.extract_called == 1
    assert memory.embed_called == 1


@pytest.mark.asyncio
async def test_ingestion_manager_extraction_timeout_falls_back():
    class _SlowMemory(_MemoryStub):
        async def _extract_with_profile(self, content: str, model_profile: str):
            self.extract_called += 1
            await asyncio.sleep(0.05)
            return ExtractionResult(summary="slow")

    memory = _SlowMemory()
    manager = IngestionManager(memory)

    result = await manager.process_add(
        content="normal imported text",
        user_id="global_user",
        agent_id=None,
        metadata={"muninn_extraction_timeout_seconds": 0.01},
        namespace="global",
        memory_type=MemoryType.EPISODIC,
        provenance=Provenance.INGESTED,
        scope="project",
    )

    assert result["event"] == "PROCESS_COMPLETE"
    assert memory.extract_called == 1
    assert memory.embed_called == 1
    assert result["record"].metadata.get("muninn_extraction_timed_out") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("kind, checked", [("conversation_turn", False), ("thread_summary", False), (None, True)])
async def test_imported_conversation_records_skip_dedup_and_conflict_resolution(kind, checked):
    """An old imported turn is a record of what was said: it never merges into or retires a memory."""
    from muninn.dedup.semantic_dedup import DedupStrategy

    class _Vectors:
        def count(self):
            return 5

        def search(self, *args, **kwargs):
            return []

    calls = []
    memory = _MemoryStub()
    memory._vectors = _Vectors()
    memory._dedup = SimpleNamespace(check_duplicate=lambda **kw: calls.append("dedup") or SimpleNamespace(
        is_duplicate=True, strategy=DedupStrategy.SKIP, existing_memory_id="m1", model_dump=lambda: {}))
    memory._conflict_detector = SimpleNamespace(detect_conflicts=lambda *a: calls.append("conflict") or [])
    metadata = {"import_source": "agent_history", "kind": kind} if kind else {}

    result = await IngestionManager(memory).process_add(
        content="User: continue", user_id="global_user", agent_id=None, metadata=metadata, namespace="global",
        memory_type=MemoryType.EPISODIC, provenance=Provenance.INGESTED, scope="project",
    )
    assert ("dedup" in calls) is checked
    assert result["event"] == ("DEDUP_SKIP" if checked else "PROCESS_COMPLETE")
