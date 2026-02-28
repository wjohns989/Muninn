"""Tests for muninn.extraction.pipeline profile-based routing."""

from muninn.core.types import Entity, ExtractionResult
from muninn.extraction.pipeline import ExtractionPipeline


class _StubExtractor:
    def __init__(self, result: ExtractionResult, available: bool = True):
        self._result = result
        self.is_available = available

    def extract(self, _text: str) -> ExtractionResult:
        return self._result


def test_normalize_openai_base_url_adds_v1_suffix():
    assert ExtractionPipeline._normalize_openai_base_url("http://localhost:11434") == "http://localhost:11434/v1"
    assert ExtractionPipeline._normalize_openai_base_url("http://localhost:11434/v1") == "http://localhost:11434/v1"


def test_build_instructor_route_specs_balanced_prefers_xlam_first():
    routes = ExtractionPipeline._build_instructor_route_specs(
        profile="balanced",
        xlam_url="http://localhost:8001/v1",
        xlam_model="xLAM",
        ollama_url="http://localhost:11434",
        ollama_model="llama3.2:3b",
        ollama_balanced_model="qwen3:8b",
        ollama_high_reasoning_model="qwen3:14b",
    )
    assert routes[0][0].startswith("xlam")
    assert routes[0][1] == "http://localhost:8001/v1"


def test_build_instructor_route_specs_low_latency_prefers_ollama_first():
    routes = ExtractionPipeline._build_instructor_route_specs(
        profile="low_latency",
        xlam_url="http://localhost:8001/v1",
        xlam_model="xLAM",
        ollama_url="http://localhost:11434",
        ollama_model="llama3.2:3b",
        ollama_balanced_model="qwen3:8b",
        ollama_high_reasoning_model="qwen3:14b",
    )
    assert routes[0][0].startswith("ollama-low")
    assert routes[0][1] == "http://localhost:11434/v1"


def test_extract_uses_requested_profile_routes(monkeypatch):
    monkeypatch.setattr(
        "muninn.extraction.pipeline.rule_based_extract",
        lambda _text: ExtractionResult(),
    )

    pipeline = ExtractionPipeline.__new__(ExtractionPipeline)
    pipeline.model_profile = "balanced"
    pipeline._instructor_routes_by_profile = {
        "balanced": [("balanced", _StubExtractor(ExtractionResult()))],
        "high_reasoning": [
            (
                "high",
                _StubExtractor(
                    ExtractionResult(
                        entities=[Entity(name="Alpha", entity_type="concept")]
                    )
                ),
            )
        ],
    }
    pipeline._xlam_available = False
    pipeline._ollama_available = False

    # ``extract`` is an async coroutine; run it synchronously for unit tests
    import asyncio
    result = asyncio.run(pipeline.extract("test", model_profile="high_reasoning"))
    assert len(result.entities) == 1
    assert result.entities[0].name == "Alpha"


def test_extract_invalid_profile_falls_back_to_default_profile(monkeypatch):
    monkeypatch.setattr(
        "muninn.extraction.pipeline.rule_based_extract",
        lambda _text: ExtractionResult(),
    )

    pipeline = ExtractionPipeline.__new__(ExtractionPipeline)
    pipeline.model_profile = "balanced"
    pipeline._instructor_routes_by_profile = {
        "balanced": [
            (
                "balanced",
                _StubExtractor(
                    ExtractionResult(
                        entities=[Entity(name="Beta", entity_type="concept")]
                    )
                ),
            )
        ]
    }
    pipeline._xlam_available = False
    pipeline._ollama_available = False

    import asyncio
    result = asyncio.run(pipeline.extract("test", model_profile="not-a-profile"))
    assert len(result.entities) == 1
    assert result.entities[0].name == "Beta"
