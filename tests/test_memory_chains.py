import asyncio
import time
from unittest.mock import MagicMock

from muninn.chains import MemoryChainDetector
from muninn.core.memory import MuninnMemory
from muninn.core.types import Entity, ExtractionResult, MemoryRecord, MemoryType, Provenance


def test_chain_detector_prefers_causal_link_when_markers_present():
    detector = MemoryChainDetector(threshold=0.2, max_hours_apart=24.0, max_links_per_memory=3)
    now = time.time()

    predecessor = MemoryRecord(
        id="mem-prev",
        content="Investigated Redis queue backlog",
        memory_type=MemoryType.EPISODIC,
        provenance=Provenance.AUTO_EXTRACTED,
        project="proj-a",
        namespace="global",
        created_at=now - 3600,
        metadata={"user_id": "user-1", "entity_names": ["Redis", "Queue"]},
    )
    successor = MemoryRecord(
        id="mem-next",
        content="Queue recovered",
        memory_type=MemoryType.EPISODIC,
        provenance=Provenance.AUTO_EXTRACTED,
        project="proj-a",
        namespace="global",
        created_at=now,
        metadata={"user_id": "user-1"},
    )

    links = detector.detect_links(
        successor_record=successor,
        successor_content="Queue recovered because redis cache warmed.",
        successor_entity_names=["Queue", "Redis"],
        candidate_records=[predecessor],
    )

    assert len(links) == 1
    assert links[0].predecessor_id == "mem-prev"
    assert links[0].successor_id == "mem-next"
    assert links[0].relation_type == "CAUSES"
    assert links[0].confidence >= 0.2


def test_add_persists_chain_links_when_detector_enabled():
    memory = MuninnMemory()
    memory._initialized = True
    memory._chain_detector = MemoryChainDetector(
        threshold=0.2,
        max_hours_apart=24.0,
        max_links_per_memory=3,
    )

    now = time.time()
    candidate = MemoryRecord(
        id="mem-prev",
        content="Investigated Redis queue backlog",
        memory_type=MemoryType.EPISODIC,
        provenance=Provenance.AUTO_EXTRACTED,
        project="proj-a",
        namespace="global",
        created_at=now - 1200,
        metadata={"user_id": "user-1", "entity_names": ["Redis", "Queue"]},
    )

    async def _extract(_content: str) -> ExtractionResult:
        return ExtractionResult(
            entities=[
                Entity(name="Redis", entity_type="tech"),
                Entity(name="Queue", entity_type="concept"),
            ],
            relations=[],
        )

    memory._extract = _extract
    memory._embed = lambda _text: [0.1, 0.2, 0.3]

    memory._metadata = MagicMock()
    memory._metadata.get_all.return_value = [candidate]
    memory._vectors = MagicMock()
    memory._vectors.count.return_value = 0
    memory._graph = MagicMock()
    memory._graph.add_chain_link.return_value = True
    memory._bm25 = MagicMock()
    memory._goal_compass = None

    result = asyncio.run(
        memory.add(
            "Queue recovered because redis cache warmed.",
            user_id="user-1",
            metadata={"project": "proj-a"},
            namespace="global",
        )
    )

    assert result["event"] == "ADD"
    assert result["chain_links_created"] >= 1
    assert memory._graph.add_chain_link.call_count >= 1

    stored_record = memory._metadata.add.call_args.args[0]
    assert stored_record.metadata["entity_names"] == ["Redis", "Queue"]
