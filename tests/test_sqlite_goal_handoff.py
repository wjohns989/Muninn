from muninn.store.sqlite_metadata import SQLiteMetadataStore


def test_project_goal_roundtrip(tmp_path):
    store = SQLiteMetadataStore(tmp_path / "goal.db")

    store.set_project_goal(
        user_id="u1",
        namespace="global",
        project="muninn_mcp",
        goal_statement="Ship ROI-first roadmap tranche",
        constraints=["local-first", "backward-compatible"],
        goal_embedding=[0.1, 0.2, 0.3],
    )

    goal = store.get_project_goal(
        user_id="u1",
        namespace="global",
        project="muninn_mcp",
    )

    assert goal is not None
    assert goal["goal_statement"] == "Ship ROI-first roadmap tranche"
    assert goal["constraints"] == ["local-first", "backward-compatible"]
    assert goal["goal_embedding"] == [0.1, 0.2, 0.3]


def test_handoff_event_ledger_is_idempotent(tmp_path):
    store = SQLiteMetadataStore(tmp_path / "handoff.db")

    inserted_first = store.record_handoff_event("evt-123", source="unit-test")
    inserted_second = store.record_handoff_event("evt-123", source="unit-test")

    assert inserted_first is True
    assert inserted_second is False
    assert store.has_handoff_event("evt-123") is True


def test_user_profile_roundtrip(tmp_path):
    store = SQLiteMetadataStore(tmp_path / "profile.db")
    payload = {
        "preferences": {"coding_style": "roi-first"},
        "skills": ["python", "mcp"],
        "paths": {"workspace": "C:/Users/user/muninn_mcp"},
    }
    store.set_user_profile(
        user_id="global_user",
        profile=payload,
        source="unit-test",
    )

    profile = store.get_user_profile(user_id="global_user")
    assert profile is not None
    assert profile["user_id"] == "global_user"
    assert profile["profile"] == payload
    assert profile["source"] == "unit-test"
    assert isinstance(profile["updated_at"], float)
