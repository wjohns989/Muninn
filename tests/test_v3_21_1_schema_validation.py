"""
Phase 21: Zero-Trust Parser Isolation & Ingestion Safety (SOTA+)

Testing the schema validation and enforcement layer (InstructorExtractor).
Verifies that:
1. Valid text is correctly validated against the `ExtractedMemoryFacts` Pydantic model.
2. The extractor returns empty/safe results when the backend LLM is unavailable.
3. The bridge between Instructor output and native `ExtractionResult` works without data loss.
"""

from typing import List
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from muninn.core.types import ExtractionResult
from muninn.extraction.instructor_extractor import InstructorExtractor
from muninn.extraction.models import ExtractedMemoryFacts, ExtractedEntity, ExtractedRelation


@pytest.fixture
def mock_instructor_client():
    """Returns a mock Instructor client that yields predetermined ExtractedMemoryFacts."""
    mock_instructor = MagicMock()
    mock_openai = MagicMock()
    
    with patch.dict("sys.modules", {"instructor": mock_instructor, "openai": mock_openai}):
        # Set up the mock client
        mock_client = MagicMock()
        mock_instructor.from_openai.return_value = mock_client
        mock_instructor.Mode.JSON = "JSON"
        
        # Create the extractor instance
        extractor = InstructorExtractor(base_url="http://mock", model="mock-model")
        
        # It should mark itself available if the imports succeeded in the mock
        assert extractor.is_available is True
        
        yield extractor, mock_client


def test_schema_bridging_valid_output(mock_instructor_client):
    """
    Test that a valid ExtractedMemoryFacts response from the LLM
    is correctly brigded into Muninn's native ExtractionResult.
    """
    extractor, mock_client = mock_instructor_client
    
    # Mock the LLM returning a perfect Pydantic object
    mock_facts = ExtractedMemoryFacts(
        entities=[
            ExtractedEntity(name="Muninn", entity_type="project", confidence=0.99),
            ExtractedEntity(name="Loki", entity_type="project", confidence=0.95),
        ],
        relations=[
            ExtractedRelation(subject="Muninn", predicate="depends_on", object="Loki", confidence=0.90)
        ],
        key_facts=["Muninn is a memory system.", "Loki is a vision model."],
        summary="Muninn and Loki are related systems.",
        temporal_context="present"
    )
    
    mock_client.chat.completions.create.return_value = mock_facts
    
    # Run the extraction
    result = extractor.extract("Some text mentioning Muninn and Loki.")
    
    # Verify the bridge to ExtractionResult
    assert isinstance(result, ExtractionResult)
    assert len(result.entities) == 2
    assert result.entities[0].name == "Muninn"
    assert result.entities[0].entity_type == "project"
    
    assert len(result.relations) == 1
    assert result.relations[0].subject == "Muninn"
    assert result.relations[0].predicate == "depends_on"
    assert result.relations[0].object == "Loki"
    
    assert result.summary == "Muninn and Loki are related systems."
    assert result.temporal_context == "present"


def test_schema_graceful_failure_handling(mock_instructor_client):
    """
    Test that if Instructor completely fails (e.g. max retries exceeded for bad JSON),
    the extractor catches the exception and returns a safe, empty result
    instead of crashing the ingestion pipeline.
    """
    extractor, mock_client = mock_instructor_client
    
    # Mock the LLM throwing a validation error (or any other exception)
    mock_client.chat.completions.create.side_effect = ValidationError.from_exception_data(
        title="mock", line_errors=[]
    )
    
    # Run the extraction
    result = extractor.extract("Some text that causes the LLM to outputs garbage.")
    
    # Verify it fails open gracefully
    assert isinstance(result, ExtractionResult)
    assert len(result.entities) == 0
    assert len(result.relations) == 0
    assert result.summary is None


def test_schema_unavailable_bypass():
    """
    Test that if Instructor is not available (e.g. missing deps or bad init),
    extract() returns an empty ExtractionResult without attempting a call.
    """
    # Force the import in instructor_extractor to fail
    with patch.dict("sys.modules", {"instructor": None}):
        # Need to reload or re-import it so the try/except block fires
        from muninn.extraction.instructor_extractor import InstructorExtractor
        extractor = InstructorExtractor(base_url="http://mock", model="mock-model")
        
        assert extractor.is_available is False
        
        result = extractor.extract("This text shouldn't be processed.")
        
        assert isinstance(result, ExtractionResult)
        assert len(result.entities) == 0
        assert len(result.relations) == 0
        assert result.summary is None
