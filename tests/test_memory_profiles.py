import asyncio

import pytest

from muninn.core.memory import MuninnMemory


class _Metadata:
    def __init__(self):
        self.events = []

    def record_profile_policy_event(self, *, source, updates, policy):
        event_id = len(self.events) + 1
        self.events.append(
            {
                "id": event_id,
                "source": source,
                "updates": updates,
                "policy": policy,
            }
        )
        return event_id

    def get_profile_policy_events(self, *, limit=25):
        return list(reversed(self.events))[:limit]


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
    memory._metadata = _Metadata()

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
            source="test_suite",
        )
    )

    assert result["event"] == "MODEL_PROFILE_POLICY_UPDATED"
    assert result["updates"]["model_profile"]["to"] == "high_reasoning"
    assert result["policy"]["active"]["model_profile"] == "high_reasoning"
    assert memory._extraction.model_profile == "high_reasoning"
    assert result["audit_event_id"] == 1
    events = asyncio.run(memory.get_model_profile_events(limit=10))
    assert events["event"] == "MODEL_PROFILE_EVENTS"
    assert events["count"] == 1
    assert events["events"][0]["source"] == "test_suite"


def test_set_model_profiles_rejects_invalid_profile():
    memory = MuninnMemory()
    memory._initialized = True
    memory._metadata = _Metadata()

    with pytest.raises(ValueError, match="Unsupported runtime_model_profile"):
        asyncio.run(memory.set_model_profiles(runtime_model_profile="invalid"))
