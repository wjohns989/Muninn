import asyncio

from muninn.goal.compass import GoalCompass
from muninn.store.sqlite_metadata import SQLiteMetadataStore


def _embed(text: str):
    lowered = text.lower()
    if "muninn" in lowered or "ship" in lowered:
        return [1.0, 0.0]
    if "vacation" in lowered or "travel" in lowered:
        return [0.0, 1.0]
    return [0.5, 0.5]


def test_goal_compass_detects_drift(tmp_path):
    store = SQLiteMetadataStore(tmp_path / "goal_compass.db")
    compass = GoalCompass(
        metadata_store=store,
        embed_fn=_embed,
        drift_threshold=0.7,
        signal_weight=0.65,
        reminder_max_chars=220,
    )

    asyncio.run(
        compass.set_goal(
            user_id="global_user",
            namespace="global",
            project="muninn_mcp",
            goal_statement="Ship Muninn roadmap reliability tranche",
            constraints=["Keep local-first", "Maintain backward compatibility"],
        )
    )

    aligned = asyncio.run(
        compass.evaluate_drift(
            text="Investigate Muninn retrieval scoring regression",
            user_id="global_user",
            namespace="global",
            project="muninn_mcp",
        )
    )
    assert aligned is not None
    assert aligned["is_drift"] is False

    drifted = asyncio.run(
        compass.evaluate_drift(
            text="Plan family vacation itinerary for summer",
            user_id="global_user",
            namespace="global",
            project="muninn_mcp",
        )
    )
    assert drifted is not None
    assert drifted["is_drift"] is True
    assert "Ship Muninn roadmap reliability tranche" in drifted["reminder"]

