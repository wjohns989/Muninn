import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from muninn.consolidation.daemon import ConsolidationDaemon
from muninn.core.config import ConsolidationConfig


def _make_daemon(config: ConsolidationConfig | None = None) -> ConsolidationDaemon:
    return ConsolidationDaemon(
        config=config or ConsolidationConfig(),
        metadata=MagicMock(),
        vectors=MagicMock(),
        graph=MagicMock(),
        bm25=MagicMock(),
        embed_fn=MagicMock(),
    )


def test_consolidation_does_not_load_integrity_model_during_construction():
    with (
        patch("muninn.conflict.detector.ConflictDetector") as detector_class,
        patch("muninn.conflict.resolver.ConflictResolver") as resolver_class,
    ):
        daemon = _make_daemon()

    detector_class.assert_not_called()
    resolver_class.assert_not_called()
    assert daemon._conflict_detector is None
    assert daemon._conflict_resolver is None


@pytest.mark.asyncio
async def test_cycle_scoped_integrity_model_is_released_after_phase():
    detector = MagicMock(is_available=True)
    with (
        patch("muninn.conflict.detector.ConflictDetector", return_value=detector) as detector_class,
        patch("muninn.conflict.resolver.ConflictResolver") as resolver_class,
    ):
        daemon = _make_daemon(ConsolidationConfig(integrity_resource_mode="cycle"))
        daemon.metadata.get_for_consolidation.return_value = []

        result = await daemon._phase_integrity()

    assert result["audited"] == 0
    detector_class.assert_called_once()
    resolver_class.assert_called_once()
    assert daemon._conflict_detector is None
    assert daemon._conflict_resolver is None


def test_persistent_integrity_mode_preserves_legacy_eager_loading():
    with (
        patch("muninn.conflict.detector.ConflictDetector") as detector_class,
        patch("muninn.conflict.resolver.ConflictResolver") as resolver_class,
    ):
        daemon = _make_daemon(ConsolidationConfig(integrity_resource_mode="persistent"))

    detector_class.assert_called_once()
    resolver_class.assert_called_once()
    assert daemon._conflict_detector is detector_class.return_value
    assert daemon._conflict_resolver is resolver_class.return_value


@pytest.mark.asyncio
async def test_overlapping_consolidation_cycles_are_serialized():
    daemon = _make_daemon()
    first_cycle_entered = asyncio.Event()
    release_first_cycle = asyncio.Event()

    async def controlled_decay():
        first_cycle_entered.set()
        await release_first_cycle.wait()
        return {}

    daemon._phase_decay = AsyncMock(side_effect=controlled_decay)
    for phase_name in (
        "_phase_merge",
        "_phase_promote",
        "_phase_replay",
        "_phase_statistics",
        "_phase_maintenance",
        "_phase_optimization",
        "_phase_integrity",
    ):
        setattr(daemon, phase_name, AsyncMock(return_value={}))

    first = asyncio.create_task(daemon.run_cycle())
    await first_cycle_entered.wait()
    second = asyncio.create_task(daemon.run_cycle())
    await asyncio.sleep(0)

    assert daemon._phase_decay.await_count == 1

    release_first_cycle.set()
    await asyncio.gather(first, second)
    assert daemon._phase_decay.await_count == 2


@pytest.mark.asyncio
async def test_stop_waits_for_active_cycle_before_releasing_integrity_resources():
    daemon = _make_daemon()
    cycle_entered = asyncio.Event()
    release_cycle = asyncio.Event()

    async def controlled_decay():
        cycle_entered.set()
        await release_cycle.wait()
        return {}

    daemon._phase_decay = AsyncMock(side_effect=controlled_decay)
    for phase_name in (
        "_phase_merge",
        "_phase_promote",
        "_phase_replay",
        "_phase_statistics",
        "_phase_maintenance",
        "_phase_optimization",
        "_phase_integrity",
    ):
        setattr(daemon, phase_name, AsyncMock(return_value={}))
    daemon._release_integrity_components = MagicMock()

    cycle = asyncio.create_task(daemon.run_cycle())
    await cycle_entered.wait()
    stop = asyncio.create_task(daemon.stop())
    await asyncio.sleep(0)

    daemon._release_integrity_components.assert_not_called()

    release_cycle.set()
    await asyncio.gather(cycle, stop)
    daemon._release_integrity_components.assert_called_once_with()
