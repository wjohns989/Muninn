import asyncio
from unittest.mock import MagicMock

from muninn.core.memory import MuninnMemory
from muninn.core.types import ExtractionResult, MemoryRecord, MemoryType, Provenance


def test_update_persists_content_with_metadata_update_signature():
    memory = MuninnMemory()
    memory._initialized = True

    record = MemoryRecord(
        id="mem-1",
        content="old",
        memory_type=MemoryType.EPISODIC,
        provenance=Provenance.AUTO_EXTRACTED,
        namespace="project-a",
        metadata={"user_id": "user-1"},
    )

    async def _extract(_content):
        return ExtractionResult()

    memory._extract = _extract
    memory._embed = lambda _text: [0.1, 0.2, 0.3]

    memory._metadata = MagicMock()
    memory._metadata.get.return_value = record
    memory._vectors = MagicMock()
    memory._graph = MagicMock()
    memory._bm25 = MagicMock()

    result = asyncio.run(memory.update("mem-1", "new content"))

    assert result["event"] == "UPDATE"
    memory._metadata.update.assert_called_once_with(
        "mem-1",
        content="new content",
        metadata={"user_id": "user-1"},
    )


def test_update_uses_runtime_model_profile_for_extraction():
    memory = MuninnMemory()
    memory._initialized = True
    memory.config.extraction.runtime_model_profile = "low_latency"

    record = MemoryRecord(
        id="mem-2",
        content="old",
        memory_type=MemoryType.EPISODIC,
        provenance=Provenance.AUTO_EXTRACTED,
        namespace="project-a",
        metadata={"user_id": "user-1"},
    )

    captured = {"profile": None}

    async def _extract(_content, model_profile=None):
        captured["profile"] = model_profile
        return ExtractionResult()

    memory._extract = _extract
    memory._embed = lambda _text: [0.1, 0.2, 0.3]

    memory._metadata = MagicMock()
    memory._metadata.get.return_value = record
    memory._vectors = MagicMock()
    memory._graph = MagicMock()
    memory._bm25 = MagicMock()

    result = asyncio.run(memory.update("mem-2", "new content"))

    assert result["event"] == "UPDATE"
    assert captured["profile"] == "low_latency"
