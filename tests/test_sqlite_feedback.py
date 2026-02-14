from muninn.store.sqlite_metadata import SQLiteMetadataStore


def test_add_retrieval_feedback_and_compute_multipliers(tmp_path):
    store = SQLiteMetadataStore(tmp_path / "feedback.db")

    # Strong positive vector outcomes
    store.add_retrieval_feedback(
        user_id="u1",
        namespace="global",
        project="muninn_mcp",
        query_text="query a",
        memory_id="m1",
        outcome=1.0,
        signals={"vector": 0.9, "bm25": 0.1},
        source="test",
    )
    store.add_retrieval_feedback(
        user_id="u1",
        namespace="global",
        project="muninn_mcp",
        query_text="query b",
        memory_id="m2",
        outcome=1.0,
        signals={"vector": 0.8, "bm25": 0.2},
        source="test",
    )
    # Negative bm25-dominant example
    store.add_retrieval_feedback(
        user_id="u1",
        namespace="global",
        project="muninn_mcp",
        query_text="query c",
        memory_id="m3",
        outcome=0.0,
        signals={"vector": 0.1, "bm25": 0.9},
        source="test",
    )

    multipliers = store.get_feedback_signal_multipliers(
        user_id="u1",
        namespace="global",
        project="muninn_mcp",
        lookback_days=30,
        min_total_signal_weight=0.1,
        floor=0.75,
        ceiling=1.25,
    )

    assert "vector" in multipliers
    assert "bm25" in multipliers
    assert multipliers["vector"] > multipliers["bm25"]
    assert 0.75 <= multipliers["vector"] <= 1.25
    assert 0.75 <= multipliers["bm25"] <= 1.25


def test_feedback_multipliers_respect_scope(tmp_path):
    store = SQLiteMetadataStore(tmp_path / "feedback_scope.db")

    store.add_retrieval_feedback(
        user_id="alice",
        namespace="global",
        project="proj-a",
        query_text="a",
        memory_id="m1",
        outcome=1.0,
        signals={"vector": 1.0},
        source="test",
    )
    store.add_retrieval_feedback(
        user_id="bob",
        namespace="global",
        project="proj-a",
        query_text="b",
        memory_id="m2",
        outcome=0.0,
        signals={"vector": 1.0},
        source="test",
    )

    alice = store.get_feedback_signal_multipliers(
        user_id="alice",
        namespace="global",
        project="proj-a",
        min_total_signal_weight=0.1,
    )
    bob = store.get_feedback_signal_multipliers(
        user_id="bob",
        namespace="global",
        project="proj-a",
        min_total_signal_weight=0.1,
    )

    assert alice["vector"] > bob["vector"]


def test_feedback_multipliers_snips_estimator_uses_propensity(tmp_path):
    store = SQLiteMetadataStore(tmp_path / "feedback_snips.db")

    store.add_retrieval_feedback(
        user_id="u1",
        namespace="global",
        project="muninn_mcp",
        query_text="q1",
        memory_id="m1",
        outcome=1.0,
        rank=1,
        sampling_prob=1.0,
        signals={"vector": 1.0},
        source="test",
    )
    store.add_retrieval_feedback(
        user_id="u1",
        namespace="global",
        project="muninn_mcp",
        query_text="q2",
        memory_id="m2",
        outcome=0.0,
        rank=10,
        sampling_prob=1.0,
        signals={"vector": 1.0},
        source="test",
    )

    mean_multipliers = store.get_feedback_signal_multipliers(
        user_id="u1",
        namespace="global",
        project="muninn_mcp",
        estimator="weighted_mean",
        min_total_signal_weight=0.1,
        floor=0.75,
        ceiling=1.25,
    )
    snips_multipliers = store.get_feedback_signal_multipliers(
        user_id="u1",
        namespace="global",
        project="muninn_mcp",
        estimator="snips",
        min_total_signal_weight=0.1,
        min_effective_samples=1.0,
        floor=0.75,
        ceiling=1.25,
    )

    assert "vector" in mean_multipliers
    assert "vector" in snips_multipliers
    assert snips_multipliers["vector"] < mean_multipliers["vector"]


def test_feedback_add_clamps_rank_and_sampling_prob(tmp_path):
    store = SQLiteMetadataStore(tmp_path / "feedback_sanitize.db")
    feedback_id = store.add_retrieval_feedback(
        user_id="u1",
        namespace="global",
        project="muninn_mcp",
        query_text="q1",
        memory_id="m1",
        outcome=1.0,
        rank=0,  # invalid rank should be persisted as NULL
        sampling_prob=9.0,  # out of bounds should clamp to 1.0
        signals={"vector": 1.0},
        source="test",
    )

    conn = store._get_conn()
    row = conn.execute(
        "SELECT rank, sampling_prob FROM retrieval_feedback WHERE id = ?",
        (feedback_id,),
    ).fetchone()
    assert row is not None
    assert row["rank"] is None
    assert row["sampling_prob"] == 1.0
