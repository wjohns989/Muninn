"""Tests for muninn.extraction.rules — Rule-based entity/keyword extraction."""

import pytest
from muninn.extraction.rules import rule_based_extract
from muninn.core.types import ExtractionResult


class TestRuleBasedExtract:
    def test_returns_extraction_result(self):
        result = rule_based_extract("Hello world")
        assert isinstance(result, ExtractionResult)

    def test_extracts_tech_entities(self):
        result = rule_based_extract(
            "Python is a programming language used for machine learning with TensorFlow"
        )
        entity_names = [e.name for e in result.entities]
        assert "Python" in entity_names
        assert "TensorFlow" in entity_names

    def test_empty_input(self):
        result = rule_based_extract("")
        assert isinstance(result, ExtractionResult)
        assert result.entities == []
        assert result.relations == []
        assert result.summary == ""

    def test_extracts_entities_from_tech_names(self):
        result = rule_based_extract(
            "Django and Flask are Python web frameworks. React uses JavaScript."
        )
        entity_names = [e.name for e in result.entities]
        assert "Django" in entity_names
        assert "Flask" in entity_names
        assert "React" in entity_names

    def test_url_extraction(self):
        result = rule_based_extract(
            "Check out https://github.com/muninn for more information"
        )
        assert isinstance(result, ExtractionResult)
        entity_names = [e.name for e in result.entities]
        assert any("github.com" in n for n in entity_names)

    def test_date_extraction(self):
        result = rule_based_extract(
            "The meeting is scheduled for 2025-03-15 at 3pm"
        )
        assert isinstance(result, ExtractionResult)
        assert result.temporal_context is not None
        assert "2025-03-15" in result.temporal_context

    def test_relation_extraction(self):
        result = rule_based_extract(
            "FastAPI uses Pydantic for data validation."
        )
        assert isinstance(result, ExtractionResult)
        # Relations depend on pattern matching against known entities

    def test_long_content(self):
        text = "This is a test sentence about various topics. " * 100
        result = rule_based_extract(text)
        assert isinstance(result, ExtractionResult)
        # Should not crash on long input

    def test_summary_generated(self):
        result = rule_based_extract("This is the first sentence. This is the second.")
        assert result.summary is not None
        assert len(result.summary) > 0

    def test_file_path_extraction(self):
        result = rule_based_extract("Edit the file src/main.py to fix the bug")
        entity_names = [e.name for e in result.entities]
        assert any("main.py" in n for n in entity_names)
