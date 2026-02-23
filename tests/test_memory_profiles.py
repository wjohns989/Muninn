import asyncio
import time

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
                "created_at": time.time(),
            }
        )
        return event_id

    def get_profile_policy_events(self, *, limit=25):
        return list(reversed(self.events))[:limit]

    def get_profile_policy_event_stats_since(self, *, since_epoch: float):
        in_window = [e for e in self.events if float(e.get("created_at", 0.0)) >= since_epoch]
        per_source = {}
        for event in in_window:
            source = str(event.get("source") or "unknown")
            per_source[source] = per_source.get(source, 0) + 1
        if per_source:
            top_source = sorted(per_source.items(), key=lambda item: (-item[1], item[0]))[0]
            top_source_name = top_source[0]
            top_source_count = top_source[1]
        else:
            top_source_name = None
            top_source_count = 0
        return {
            "events_in_window": len(in_window),
            "distinct_sources": len(per_source),
            "top_source": top_source_name,
            "top_source_count": top_source_count,
        }


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
    assert result["alert_evaluation"]["event"] == "MODEL_PROFILE_ALERT_EVALUATION"
    assert result["alert_hook"]["configured"] is False


def test_set_model_profiles_rejects_invalid_profile():
    memory = MuninnMemory()
    memory._initialized = True
    memory._metadata = _Metadata()

    with pytest.raises(ValueError, match="Unsupported runtime_model_profile"):
        asyncio.run(memory.set_model_profiles(runtime_model_profile="invalid"))


def test_get_model_profile_alerts_detects_churn(monkeypatch):
    memory = MuninnMemory()
    memory._initialized = True
    memory._metadata = _Metadata()

    for _ in range(3):
        memory._metadata.record_profile_policy_event(
            source="runtime_api",
            updates={"runtime_model_profile": {"from": "balanced", "to": "low_latency"}},
            policy={"active": {"runtime_model_profile": "low_latency"}},
        )

    result = asyncio.run(
        memory.get_model_profile_alerts(
            window_seconds=600,
            churn_threshold=2,
            source_churn_threshold=2,
            distinct_sources_threshold=2,
        )
    )

    assert result["event"] == "MODEL_PROFILE_ALERT_EVALUATION"
    assert result["stats"]["events_in_window"] == 3
    codes = {a["code"] for a in result["alerts"]}
    assert "PROFILE_POLICY_CHURN" in codes
    assert "PROFILE_POLICY_SOURCE_CHURN" in codes
    assert "PROFILE_POLICY_MULTI_SOURCE_CHURN" not in codes
