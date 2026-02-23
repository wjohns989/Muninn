from muninn.store.sqlite_metadata import SQLiteMetadataStore


def test_profile_policy_events_roundtrip(tmp_path):
    store = SQLiteMetadataStore(tmp_path / "meta.db")

    event_id = store.record_profile_policy_event(
        source="unit_test",
        updates={"runtime_model_profile": {"from": "balanced", "to": "low_latency"}},
        policy={"active": {"runtime_model_profile": "low_latency"}},
    )

    assert event_id >= 1
    events = store.get_profile_policy_events(limit=5)
    assert len(events) == 1
    assert events[0]["source"] == "unit_test"
    assert events[0]["updates"]["runtime_model_profile"]["to"] == "low_latency"


def test_profile_policy_event_stats_since(tmp_path):
    store = SQLiteMetadataStore(tmp_path / "meta.db")
    now = __import__("time").time()

    event_a = store.record_profile_policy_event(
        source="runtime_api",
        updates={"runtime_model_profile": {"from": "balanced", "to": "low_latency"}},
        policy={"active": {"runtime_model_profile": "low_latency"}},
    )
    event_b = store.record_profile_policy_event(
        source="runtime_api",
        updates={"ingestion_model_profile": {"from": "balanced", "to": "high_reasoning"}},
        policy={"active": {"ingestion_model_profile": "high_reasoning"}},
    )
    event_c = store.record_profile_policy_event(
        source="sdk",
        updates={"model_profile": {"from": "balanced", "to": "high_reasoning"}},
        policy={"active": {"model_profile": "high_reasoning"}},
    )
    assert event_a and event_b and event_c

    stats = store.get_profile_policy_event_stats_since(since_epoch=now - 5.0)

    assert stats["events_in_window"] == 3
    assert stats["distinct_sources"] == 2
    assert stats["top_source"] == "runtime_api"
    assert stats["top_source_count"] == 2
