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
