"""Tests for self-supervised adaptive importance (ACT-R activation + online learner)."""

import json
import random
import time
from unittest.mock import MagicMock

import pytest

from muninn.consolidation.daemon import ADAPTIVE_STATE_KEY, ConsolidationDaemon
from muninn.core.config import ConsolidationConfig
from muninn.core.types import MemoryRecord
from muninn.scoring.adaptive import (
    FEATURE_NAMES,
    AdaptiveImportanceModel,
    base_level_activation,
    build_features,
    outcome_label,
    rank_auc,
)
from muninn.store.sqlite_metadata import SQLiteMetadataStore

DAY = 86400.0


def test_activation_rewards_recency_and_frequency():
    assert base_level_activation([1.0], 0.5) > base_level_activation([30.0], 0.5)
    assert base_level_activation([1.0, 2.0, 3.0], 0.5) > base_level_activation([1.0], 0.5)
    # A faster decay rate penalises old presentations more.
    assert base_level_activation([30.0], 0.8) < base_level_activation([30.0], 0.2)


def test_features_only_use_accesses_before_prediction_time():
    now = 1_000_000.0
    record = MemoryRecord(id="m", content="x", created_at=now - 10 * DAY)
    before = build_features(record, [now - DAY], 1, now)
    with_future = build_features(record, [now - DAY, now + DAY], 1, now)
    assert before == with_future
    assert len(before) == len(FEATURE_NAMES)


def test_outcome_label_requires_new_session_within_horizon():
    t = 100.0
    horizon = 10.0
    assert outcome_label([(t + 5, "fresh")], t, horizon) == 1
    assert outcome_label([(t - 1, "s1"), (t + 5, "s1")], t, horizon) == 0  # same session echo
    assert outcome_label([(t + 50, "late")], t, horizon) == 0  # outside horizon
    assert outcome_label([(t + 5, None)], t, horizon) == 1  # sessionless call counts as new context
    assert outcome_label([], t, horizon) == 0


def test_rank_auc():
    assert rank_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert rank_auc([0.5, 0.5], [0, 1]) == pytest.approx(0.5)
    assert rank_auc([0.2, 0.3], [1, 1]) is None


def _synthetic_example(rng, now):
    """Memories retrieved recently and often are the ones retrieved again."""
    created = now - rng.uniform(10, 200) * DAY
    hot = rng.random() < 0.3
    accesses = sorted(now - rng.uniform(0.1, 3) * DAY for _ in range(rng.randint(3, 8))) if hot else (
        sorted(now - rng.uniform(40, 120) * DAY for _ in range(rng.randint(0, 2)))
    )
    record = MemoryRecord(id="m", content="x", created_at=min([created] + accesses), novelty_score=rng.random())
    features = build_features(record, accesses, len(accesses), now)
    label = int(rng.random() < (0.8 if hot else 0.05))
    legacy = rng.random()  # stand-in for a score that carries no signal about re-retrieval
    return features, label, legacy


def test_model_learns_and_promotes_itself_only_after_beating_legacy():
    rng = random.Random(7)
    now = 5_000_000.0
    model = AdaptiveImportanceModel(min_examples=200)
    for step in range(600):
        features, label, legacy = _synthetic_example(rng, now)
        model.observe(features, model.predict(features), legacy, label)
        if step < 150:
            assert model.active is False  # not enough evidence yet
    stats = model.evaluation()
    assert stats["model_auc"] > 0.85
    assert model.active is True

    hot_features, _, _ = _synthetic_example(random.Random(1), now)
    cold = build_features(MemoryRecord(id="c", content="x", created_at=now - 150 * DAY), [], 0, now)
    assert model.importance(model.predict(hot_features)) > model.importance(model.predict(cold))


def test_model_state_round_trips():
    rng = random.Random(3)
    model = AdaptiveImportanceModel(min_examples=10, min_class_examples=2)
    for _ in range(50):
        features, label, legacy = _synthetic_example(rng, 1_000_000.0)
        model.observe(features, model.predict(features), legacy, label)
    clone = AdaptiveImportanceModel(min_examples=10, min_class_examples=2)
    clone.load_json(model.to_json())
    assert clone.weights == model.weights
    assert clone.examples_seen == model.examples_seen
    assert clone.evaluation() == model.evaluation()
    assert clone.active == model.active

    stale = json.loads(model.to_json())
    stale["features"] = ["old_feature"]
    fresh = AdaptiveImportanceModel()
    fresh.load_json(json.dumps(stale))
    assert fresh.examples_seen == 0


class _Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


def _daemon(tmp_path, **config):
    graph = MagicMock()
    graph.get_memory_node_degrees_batch.side_effect = lambda ids: {i: 0.0 for i in ids}
    daemon = ConsolidationDaemon(
        config=ConsolidationConfig(**config),
        metadata=SQLiteMetadataStore(tmp_path / "meta.db"),
        vectors=MagicMock(),
        graph=graph,
        bm25=MagicMock(),
    )
    daemon._clock = _Clock(time.time())
    return daemon


def _log_access(store, memory_id, at, session_id):
    conn = store._get_conn()
    conn.execute(
        "INSERT INTO access_events (memory_id, accessed_at, session_id, rank) VALUES (?, ?, ?, 1)",
        (memory_id, at, session_id),
    )
    conn.commit()


@pytest.mark.asyncio
async def test_daemon_labels_its_own_predictions_from_later_retrievals(tmp_path):
    daemon = _daemon(tmp_path, adaptive_horizon_days=7)
    t0 = daemon._clock.now
    for memory_id in ("reused", "ignored", "archived"):
        daemon.metadata.add(MemoryRecord(id=memory_id, content=memory_id, metadata={"user_id": "u1"}))

    await daemon._phase_decay()
    assert daemon.metadata.count_importance_predictions() == 3

    _log_access(daemon.metadata, "reused", t0 + 2 * DAY, "new-session")
    daemon.metadata.update("archived", archived=True)
    daemon._clock.now = t0 + 8 * DAY

    result = await daemon._phase_learn()

    assert result["resolved"] == 2 and result["positives"] == 1 and result["censored"] == 1
    assert daemon.metadata.count_importance_predictions() == 0
    state = json.loads(daemon.metadata.get_meta(ADAPTIVE_STATE_KEY))
    assert state["examples_seen"] == 2
    assert daemon.status["adaptive_importance"]["examples_seen"] == 2


@pytest.mark.asyncio
async def test_learned_score_drives_retention_not_ranking(tmp_path):
    daemon = _daemon(tmp_path, adaptive_horizon_days=7)
    now = daemon._clock.now
    for memory_id, age_days in (("mature", 30), ("young", 1)):
        daemon.metadata.add(MemoryRecord(id=memory_id, content=memory_id, created_at=now - age_days * DAY,
                                         metadata={"user_id": "u1"}))
    daemon._adaptive.active = True
    daemon._adaptive.weights = [0.0] * (len(FEATURE_NAMES) - 1) + [-3.0]  # p ~ 0.05: "not needed"

    result = await daemon._phase_decay()

    assert result["learned_scores"] == 1
    assert daemon.metadata.get("mature").archived is True  # retention follows the learned score
    assert daemon.metadata.get("young").archived is False  # too young to judge
    # Ranking importance is still the hand-weighted score, so the learner cannot
    # influence which memories get retrieved and thereby its own labels.
    legacy = _daemon(tmp_path / "legacy", importance_model="legacy")
    legacy.metadata.add(MemoryRecord(id="mature", content="mature", created_at=now - 30 * DAY,
                                     metadata={"user_id": "u1"}))
    await legacy._phase_decay()
    assert daemon.metadata.get("mature").importance == pytest.approx(
        legacy.metadata.get("mature").importance, abs=1e-6)


@pytest.mark.asyncio
async def test_shadow_and_legacy_modes_never_apply_learned_scores(tmp_path):
    shadow = _daemon(tmp_path, importance_model="shadow")
    shadow.metadata.add(MemoryRecord(id="m", content="a", created_at=time.time() - 30 * DAY,
                                     metadata={"user_id": "u1"}))
    shadow._adaptive.active = True
    result = await shadow._phase_decay()
    assert result["learned_scores"] == 0
    assert shadow.metadata.count_importance_predictions() == 1

    legacy = _daemon(tmp_path / "legacy", importance_model="legacy")
    assert await legacy._phase_learn() == {"skipped": "legacy"}
    assert legacy.status["adaptive_importance"] == {"mode": "legacy"}


def test_record_access_batch_logs_session_and_rank(tmp_path):
    store = SQLiteMetadataStore(tmp_path / "meta.db")
    for memory_id in ("a", "b"):
        store.add(MemoryRecord(id=memory_id, content=memory_id))

    store.record_access_batch(["a", "b"], session_id="s1")
    store.record_access_batch(["a"])

    events = store.get_access_events(["a", "b"])
    assert [sid for _, sid in events["a"]] == ["s1", None]
    assert [sid for _, sid in events["b"]] == ["s1"]
    assert store.get("a").access_count == 2
    ranks = [r[0] for r in store._get_conn().execute(
        "SELECT rank FROM access_events WHERE session_id = 's1' ORDER BY rank").fetchall()]
    assert ranks == [1, 2]
    assert store.prune_access_events(before=time.time() + 1) == 3
