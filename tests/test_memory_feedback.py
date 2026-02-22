import asyncio
import os
from unittest.mock import MagicMock, AsyncMock

from muninn.core.memory import MuninnMemory
from muninn.core.feature_flags import reset_flags


def test_feedback_multiplier_cache_hits_and_ttl():
    memory = MuninnMemory()
    memory._initialized = True
    memory._metadata = MagicMock()
    memory._metadata.get_feedback_signal_multipliers.return_value = {"vector": 1.1}

    first = memory._get_feedback_signal_multipliers_cached(
        user_id="global_user",
        namespace="global",
        project="muninn_mcp",
    )
    second = memory._get_feedback_signal_multipliers_cached(
        user_id="global_user",
        namespace="global",
        project="muninn_mcp",
    )

    assert first == {"vector": 1.1}
    assert second == {"vector": 1.1}
    memory._metadata.get_feedback_signal_multipliers.assert_called_once()
    kwargs = memory._metadata.get_feedback_signal_multipliers.call_args.kwargs
    assert kwargs["estimator"] == memory.config.retrieval_feedback.estimator
    assert kwargs["propensity_floor"] == memory.config.retrieval_feedback.propensity_floor
    assert kwargs["min_effective_samples"] == memory.config.retrieval_feedback.min_effective_samples


def test_record_retrieval_feedback_invalidates_cache():
    memory = MuninnMemory()
    memory._initialized = True
    memory._metadata = MagicMock()
    memory._metadata.add_retrieval_feedback.return_value = 42
    mock_mem = MagicMock()
    mock_mem.metadata = {"elo_rating": 1200.0}
    memory._metadata.get.return_value = mock_mem

    memory._feedback_multiplier_cache[("global_user", "global", "muninn_mcp")] = (9999999999.0, {"vector": 1.2})

    result = asyncio.run(
        memory.record_retrieval_feedback(
            query="what is current goal",
            memory_id="mem-1",
            outcome=1.0,
            user_id="global_user",
            namespace="global",
            project="muninn_mcp",
            signals={"vector": 0.9},
            source="unit-test",
        )
    )

    assert result["feedback_id"] == 42
    assert ("global_user", "global", "muninn_mcp") not in memory._feedback_multiplier_cache
    memory._metadata.add_retrieval_feedback.assert_called_once()


def test_record_retrieval_feedback_forwards_rank_and_sampling_prob():
    memory = MuninnMemory()
    memory._initialized = True
    memory._metadata = MagicMock()
    memory._metadata.add_retrieval_feedback.return_value = 43
    mock_mem = MagicMock()
    mock_mem.metadata = {"elo_rating": 1200.0}
    memory._metadata.get.return_value = mock_mem

    result = asyncio.run(
        memory.record_retrieval_feedback(
            query="what changed",
            memory_id="mem-2",
            outcome=0.5,
            user_id="global_user",
            namespace="global",
            project="muninn_mcp",
            rank=3,
            sampling_prob=0.4,
            signals={"vector": 0.6},
            source="unit-test",
        )
    )

    kwargs = memory._metadata.add_retrieval_feedback.call_args.kwargs
    assert kwargs["rank"] == 3
    assert kwargs["sampling_prob"] == 0.4
    assert result["rank"] == 3
    assert result["sampling_prob"] == 0.4


def test_search_passes_feedback_multipliers_when_enabled():
    os.environ["MUNINN_RETRIEVAL_FEEDBACK"] = "1"
    reset_flags()
    try:
        memory = MuninnMemory()
        memory._initialized = True
        memory._metadata = MagicMock()
        memory._metadata.get_feedback_signal_multipliers.return_value = {"vector": 1.1}
        memory._retriever = MagicMock()
        memory._retriever.search = AsyncMock(return_value=[])
        memory._goal_compass = None
        memory.config.retrieval_feedback.enabled = True

        result = asyncio.run(
            memory.search(
                query="memory query",
                user_id="global_user",
                project="muninn_mcp",
            )
        )

        assert result == []
        kwargs = memory._retriever.search.call_args.kwargs
        assert kwargs["feedback_signal_multipliers"] == {"vector": 1.1}
    finally:
        os.environ.pop("MUNINN_RETRIEVAL_FEEDBACK", None)
        reset_flags()
