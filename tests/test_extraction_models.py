"""Tests for muninn.extraction.models — Pydantic extraction schemas."""

import pytest
from pydantic import ValidationError

from muninn.extraction.models import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractedMemoryFacts,
    EXTRACTION_SYSTEM_PROMPT,
    VALID_ENTITY_TYPES,
    VALID_PREDICATES,
)


class TestExtractedEntity:
    """Test entity extraction model."""

    def test_basic_entity(self):
        e = ExtractedEntity(name="Python", entity_type="tech")
        assert e.name == "Python"
        assert e.entity_type == "tech"
        assert e.confidence == 0.8  # default

    def test_custom_confidence(self):
        e = ExtractedEntity(name="John", entity_type="person", confidence=0.95)
        assert e.confidence == 0.95

    def test_confidence_bounds(self):
        """Confidence must be 0.0-1.0."""
        with pytest.raises(ValidationError):
            ExtractedEntity(name="X", entity_type="tech", confidence=1.5)
        with pytest.raises(ValidationError):
            ExtractedEntity(name="X", entity_type="tech", confidence=-0.1)

    def test_serialization(self):
        e = ExtractedEntity(name="GitHub", entity_type="org")
        d = e.model_dump()
        assert d["name"] == "GitHub"
        assert d["entity_type"] == "org"


class TestExtractedRelation:
    """Test relation extraction model."""

    def test_basic_relation(self):
        r = ExtractedRelation(
            subject="John",
            predicate="uses",
            object="Python",
        )
        assert r.subject == "John"
        assert r.predicate == "uses"
        assert r.object == "Python"
        assert r.confidence == 0.8  # default
        assert r.temporal_context is None

    def test_with_temporal_context(self):
        r = ExtractedRelation(
            subject="Company",
            predicate="migrated_from",
            object="Java",
            temporal_context="since 2024",
        )
        assert r.temporal_context == "since 2024"

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            ExtractedRelation(
                subject="A", predicate="uses", object="B", confidence=2.0
            )


class TestExtractedMemoryFacts:
    """Test complete extraction result model."""

    def test_empty_defaults(self):
        facts = ExtractedMemoryFacts()
        assert facts.entities == []
        assert facts.relations == []
        assert facts.key_facts == []
        assert facts.summary is None
        assert facts.temporal_context is None

    def test_full_extraction(self):
        facts = ExtractedMemoryFacts(
            entities=[
                ExtractedEntity(name="Python", entity_type="tech"),
                ExtractedEntity(name="FastAPI", entity_type="tech"),
            ],
            relations=[
                ExtractedRelation(
                    subject="FastAPI",
                    predicate="depends_on",
                    object="Python",
                ),
            ],
            key_facts=[
                "FastAPI is a Python web framework.",
                "FastAPI supports async operations.",
            ],
            summary="FastAPI is an async Python web framework.",
            temporal_context="present",
        )
        assert len(facts.entities) == 2
        assert len(facts.relations) == 1
        assert len(facts.key_facts) == 2
        assert facts.summary is not None

    def test_json_serialization(self):
        facts = ExtractedMemoryFacts(
            entities=[ExtractedEntity(name="Test", entity_type="concept")],
        )
        json_str = facts.model_dump_json()
        assert "Test" in json_str

    def test_schema_generation(self):
        """Verify Pydantic can generate JSON schema (needed by Instructor)."""
        schema = ExtractedMemoryFacts.model_json_schema()
        assert "properties" in schema
        assert "entities" in schema["properties"]
        assert "relations" in schema["properties"]
        assert "key_facts" in schema["properties"]


class TestConstants:
    """Test module-level constants."""

    def test_valid_entity_types(self):
        assert "person" in VALID_ENTITY_TYPES
        assert "tech" in VALID_ENTITY_TYPES
        assert "org" in VALID_ENTITY_TYPES
        assert len(VALID_ENTITY_TYPES) >= 8

    def test_valid_predicates(self):
        assert "uses" in VALID_PREDICATES
        assert "prefers" in VALID_PREDICATES
        assert "depends_on" in VALID_PREDICATES
        assert len(VALID_PREDICATES) >= 10

    def test_extraction_prompt_nonempty(self):
        assert len(EXTRACTION_SYSTEM_PROMPT) > 100
        assert "extract" in EXTRACTION_SYSTEM_PROMPT.lower()
