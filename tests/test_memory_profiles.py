import asyncio

import pytest

from muninn.core.memory import MuninnMemory


def test_get_model_profiles_returns_active_policy():
    memory = MuninnMemory()
    memory._initialized = True

    result = asyncio.run(memory.get_model_profiles())

    assert "supported_profiles" in result
    assert result["active"]["runtime_model_profile"] == "low_latency"
    assert result["active"]["ingestion_model_profile"] == "balanced"
    assert result["active"]["legacy_ingestion_model_profile"] == "balanced"


def test_set_model_profiles_updates_policy_and_pipeline_default():
    memory = MuninnMemory()
    memory._initialized = True

    class _Extraction:
        def __init__(self):
            self.model_profile = "balanced"

    memory._extraction = _Extraction()

    result = asyncio.run(
        memory.set_model_profiles(
            model_profile="high_reasoning",
            runtime_model_profile="low_latency",
            ingestion_model_profile="balanced",
            legacy_ingestion_model_profile="high_reasoning",
        )
    )

    assert result["event"] == "MODEL_PROFILE_POLICY_UPDATED"
    assert result["updates"]["model_profile"]["to"] == "high_reasoning"
    assert result["policy"]["active"]["model_profile"] == "high_reasoning"
    assert memory._extraction.model_profile == "high_reasoning"


def test_set_model_profiles_rejects_invalid_profile():
    memory = MuninnMemory()
    memory._initialized = True

    with pytest.raises(ValueError, match="Unsupported runtime_model_profile"):
        asyncio.run(memory.set_model_profiles(runtime_model_profile="invalid"))
