"""Tests for muninn.extraction.instructor_extractor — Instructor-based extraction."""

import pytest
from unittest.mock import patch, MagicMock

from muninn.core.types import ExtractionResult
from muninn.extraction.instructor_extractor import InstructorExtractor


class TestInstructorExtractorInit:
    """Test extractor initialization."""

    def test_init_without_instructor_installed(self):
        """Should gracefully handle missing instructor dependency."""
        with patch.dict("sys.modules", {"instructor": None, "openai": None}):
            # Even if import fails, extractor should init without raising
            extractor = InstructorExtractor(
                base_url="http://localhost:11434/v1",
                model="llama3.2:3b",
            )
            # Won't be available, but shouldn't crash
            assert isinstance(extractor, InstructorExtractor)

    def test_init_stores_config(self):
        """Constructor should store configuration."""
        extractor = InstructorExtractor(
            base_url="http://localhost:11434/v1",
            model="test-model",
            api_key="test-key",
            max_retries=3,
        )
        assert extractor.base_url == "http://localhost:11434/v1"
        assert extractor.model == "test-model"
        assert extractor.max_retries == 3


class TestInstructorExtractorExtract:
    """Test extraction behavior."""

    def test_extract_when_unavailable(self):
        """Should return empty result when not available."""
        extractor = InstructorExtractor(
            base_url="http://nonexistent:9999/v1",
            model="fake",
        )
        extractor._available = False
        result = extractor.extract("Some text to extract from")
        assert isinstance(result, ExtractionResult)
        assert len(result.entities) == 0
        assert len(result.relations) == 0

    def test_extract_returns_extraction_result(self):
        """Result should always be an ExtractionResult, even on failure."""
        extractor = InstructorExtractor(
            base_url="http://localhost:11434/v1",
            model="fake",
        )
        extractor._available = False
        result = extractor.extract("test")
        assert isinstance(result, ExtractionResult)


class TestInstructorExtractorConversion:
    """Test ExtractedMemoryFacts → ExtractionResult conversion."""

    def test_conversion_entities(self):
        from muninn.extraction.models import ExtractedMemoryFacts, ExtractedEntity

        extractor = InstructorExtractor(
            base_url="http://localhost:11434/v1",
            model="test",
        )

        facts = ExtractedMemoryFacts(
            entities=[
                ExtractedEntity(name="Python", entity_type="tech", confidence=0.95),
                ExtractedEntity(name="FastAPI", entity_type="tech", confidence=0.90),
            ],
            summary="Testing conversion.",
        )

        result = extractor._convert_to_extraction_result(facts)
        assert len(result.entities) == 2
        assert result.entities[0].name == "Python"
        assert result.entities[0].entity_type == "tech"
        assert result.summary == "Testing conversion."

    def test_conversion_relations(self):
        from muninn.extraction.models import (
            ExtractedMemoryFacts,
            ExtractedEntity,
            ExtractedRelation,
        )

        extractor = InstructorExtractor(
            base_url="http://localhost:11434/v1",
            model="test",
        )

        facts = ExtractedMemoryFacts(
            entities=[
                ExtractedEntity(name="FastAPI", entity_type="tech"),
                ExtractedEntity(name="Python", entity_type="tech"),
            ],
            relations=[
                ExtractedRelation(
                    subject="FastAPI",
                    predicate="depends_on",
                    object="Python",
                    confidence=0.9,
                ),
            ],
        )

        result = extractor._convert_to_extraction_result(facts)
        assert len(result.relations) == 1
        assert result.relations[0].subject == "FastAPI"
        assert result.relations[0].predicate == "depends_on"
        assert result.relations[0].confidence == 0.9

    def test_conversion_empty_facts(self):
        from muninn.extraction.models import ExtractedMemoryFacts

        extractor = InstructorExtractor(
            base_url="http://localhost:11434/v1",
            model="test",
        )

        facts = ExtractedMemoryFacts()
        result = extractor._convert_to_extraction_result(facts)
        assert len(result.entities) == 0
        assert len(result.relations) == 0
        assert result.summary is None

    def test_conversion_temporal_context(self):
        from muninn.extraction.models import ExtractedMemoryFacts

        extractor = InstructorExtractor(
            base_url="http://localhost:11434/v1",
            model="test",
        )

        facts = ExtractedMemoryFacts(
            temporal_context="since 2024",
        )
        result = extractor._convert_to_extraction_result(facts)
        assert result.temporal_context == "since 2024"


class TestInstructorExtractorProbe:
    """Test endpoint probing."""

    def test_probe_when_unavailable(self):
        extractor = InstructorExtractor(
            base_url="http://nonexistent:9999/v1",
            model="fake",
        )
        extractor._client = None
        result = extractor.probe_endpoint()
        assert result is False
