import asyncio
from unittest.mock import MagicMock, AsyncMock

from muninn.core.memory import MuninnMemory
from muninn.core.types import ExtractionResult
from muninn.core.ingestion_manager import IngestionManager


def test_add_propagates_project_and_branch_from_metadata():
    memory = MuninnMemory()
    memory._initialized = True

    async def _extract(_content):
        return ExtractionResult()

    memory._extract = _extract
    async def _embed(_text):
        return [0.2, 0.1, 0.4]
    memory._embed = _embed

    memory._metadata = MagicMock()
    memory._vectors = MagicMock()
    memory._vectors.count.return_value = 0
    memory._graph = MagicMock()
    memory._bm25 = MagicMock()
    memory._goal_compass = None
    memory._ingestion_manager = IngestionManager(memory)

    asyncio.run(
        memory.add(
            content="Ship roadmap fix",
            user_id="global_user",
            namespace="global",
            metadata={"project": "muninn_mcp", "branch": "feat/roi"},
        )
    )

    added_record = memory._metadata.add.call_args.args[0]
    assert added_record.project == "muninn_mcp"
    assert added_record.branch == "feat/roi"
    assert added_record.metadata["project"] == "muninn_mcp"

    upsert_payload = memory._vectors.upsert.call_args.kwargs["metadata"]
    assert upsert_payload["project"] == "muninn_mcp"
    assert upsert_payload["branch"] == "feat/roi"


def test_import_handoff_duplicate_event_is_idempotent():
    memory = MuninnMemory()
    memory._initialized = True
    memory._metadata = MagicMock()
    memory._metadata.has_handoff_event.return_value = True

    payload = {
        "schema_version": 1,
        "project": "muninn_mcp",
        "namespace": "global",
        "user_id": "global_user",
        "goal": None,
        "decisions": [],
        "open_questions": [],
        "memories": [],
        "watermark_created_at": 0.0,
    }
    checksum = MuninnMemory._handoff_checksum(payload)
    bundle = {
        **payload,
        "checksum": f"sha256:{checksum}",
        "event_id": "handoff:muninn_mcp:global:deadbeef",
    }

    result = asyncio.run(
        memory.import_handoff(
            bundle=bundle,
            user_id="global_user",
            namespace="global",
            project="muninn_mcp",
            source="unit-test",
        )
    )

    assert result["event"] == "HANDOFF_DUPLICATE"
    assert result["checksum_verified"] is True
    assert result["imported"] == 0