"""Tests for muninn.core.types — Pydantic models and enums."""

import uuid
from muninn.core.types import (
    MemoryType,
    Provenance,
    MemoryRecord,
    SearchResult,
    Entity,
    Relation,
    ExtractionResult,
    AddMemoryRequest,
    SearchMemoryRequest,
    UpdateMemoryRequest,
    HealthResponse,
)


class TestMemoryType:
    def test_enum_values(self):
        assert MemoryType.WORKING == "working"
        assert MemoryType.EPISODIC == "episodic"
        assert MemoryType.SEMANTIC == "semantic"
        assert MemoryType.PROCEDURAL == "procedural"

    def test_all_types_exist(self):
        types = [e.value for e in MemoryType]
        assert len(types) == 4


class TestProvenance:
    def test_enum_values(self):
        assert Provenance.USER_EXPLICIT == "user_explicit"
        assert Provenance.ASSISTANT_CONFIRMED == "assistant_confirmed"
        assert Provenance.AUTO_EXTRACTED == "auto_extracted"
        assert Provenance.INGESTED == "ingested"

    def test_all_provenances_exist(self):
        provs = [e.value for e in Provenance]
        assert len(provs) == 4


class TestMemoryRecord:
    def test_defaults(self):
        rec = MemoryRecord(content="hello world")
        assert rec.content == "hello world"
        assert rec.memory_type == MemoryType.EPISODIC
        assert rec.provenance == Provenance.AUTO_EXTRACTED
        assert rec.namespace == "global"
        assert rec.importance == 0.5
        assert rec.access_count == 0
        assert rec.consolidation_gen == 0
        assert rec.consolidated is False
        assert rec.id is not None
        uuid.UUID(rec.id)  # Verify UUID format

    def test_custom_values(self):
        rec = MemoryRecord(
            content="test",
            memory_type=MemoryType.SEMANTIC,
            provenance=Provenance.USER_EXPLICIT,
            project="project_x",
            namespace="ns1",
            importance=0.9,
            metadata={"key": "val"},
        )
        assert rec.memory_type == MemoryType.SEMANTIC
        assert rec.provenance == Provenance.USER_EXPLICIT
        assert rec.project == "project_x"
        assert rec.namespace == "ns1"
        assert rec.importance == 0.9
        assert rec.metadata == {"key": "val"}


class TestSearchResult:
    def test_basic(self):
        rec = MemoryRecord(content="found it")
        sr = SearchResult(memory=rec, score=0.95)
        assert sr.memory.content == "found it"
        assert sr.score == 0.95
        assert sr.source == "vector"


class TestEntity:
    def test_basic(self):
        e = Entity(name="Python", entity_type="tech")
        assert e.name == "Python"
        assert e.entity_type == "tech"
        assert e.source_memory_id is None


class TestRelation:
    def test_basic(self):
        r = Relation(subject="Python", predicate="uses", object="FastAPI")
        assert r.subject == "Python"
        assert r.predicate == "uses"
        assert r.object == "FastAPI"
        assert r.confidence == 1.0


class TestExtractionResult:
    def test_defaults(self):
        er = ExtractionResult()
        assert er.entities == []
        assert er.relations == []
        assert er.summary is None
        assert er.temporal_context is None

    def test_populated(self):
        er = ExtractionResult(
            entities=[Entity(name="A", entity_type="tech")],
            relations=[Relation(subject="A", predicate="uses", object="C")],
            summary="A summary",
            temporal_context="2025-03-15",
        )
        assert len(er.entities) == 1
        assert len(er.relations) == 1
        assert er.summary == "A summary"
        assert er.temporal_context == "2025-03-15"


class TestAPIModels:
    def test_add_memory_request(self):
        req = AddMemoryRequest(content="something to remember")
        assert req.content == "something to remember"
        assert req.user_id == "global_user"

    def test_search_memory_request(self):
        req = SearchMemoryRequest(query="what do I know?")
        assert req.query == "what do I know?"
        assert req.limit == 10
        assert req.rerank is True

    def test_update_memory_request(self):
        req = UpdateMemoryRequest(memory_id="abc", data="new content")
        assert req.memory_id == "abc"
        assert req.data == "new content"

    def test_health_response(self):
        hr = HealthResponse(
            status="healthy",
            backend="muninn-native",
            memory_count=42,
        )
        assert hr.status == "healthy"
        assert hr.backend == "muninn-native"
        assert hr.memory_count == 42
        assert hr.graph_nodes == 0
        assert hr.reranker == "inactive"
