from unittest.mock import MagicMock, patch

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
