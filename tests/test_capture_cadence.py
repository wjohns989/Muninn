import pytest

from muninn.history.capture_cadence import SmallCaptureCadence


class FakeClock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_fresh_start_and_restart_both_wait_full_quiet_grace():
    clock = FakeClock()
    first = SmallCaptureCadence(clock=clock)
    assert not first.planning_ready()
    assert not first.analysis_ready()
    clock.advance(299.999)
    assert not first.planning_ready()

    clock.advance(0.001)
    assert first.planning_ready()
    assert first.analysis_ready()

    restarted = SmallCaptureCadence(clock=clock)
    assert not restarted.planning_ready()
    assert not restarted.analysis_ready()


def test_accepted_activity_resets_quiet_period():
    clock = FakeClock()
    cadence = SmallCaptureCadence(clock=clock)
    clock.advance(299)
    cadence.note_activity()
    clock.advance(299.999)
    assert not cadence.planning_ready()
    clock.advance(0.001)
    assert cadence.planning_ready()


def test_analysis_cooldown_does_not_block_planning_and_has_exact_boundary():
    clock = FakeClock()
    cadence = SmallCaptureCadence(clock=clock)
    clock.advance(299)
    cadence.note_attempt()
    clock.advance(1)
    assert cadence.planning_ready()
    assert not cadence.analysis_ready()
    clock.advance(28.999)
    assert not cadence.analysis_ready()
    clock.advance(0.001)
    assert cadence.analysis_ready()


def test_continuous_capture_cannot_starve_planning_or_analysis():
    clock = FakeClock()
    cadence = SmallCaptureCadence(quiet_seconds=10, interval_seconds=4,
                                  max_wait_seconds=30, clock=clock)
    for _ in range(5):
        clock.advance(5)
        cadence.note_activity()
        assert not cadence.planning_ready()
        assert not cadence.analysis_ready()
    clock.advance(5)
    cadence.note_activity()
    assert cadence.planning_ready()
    assert cadence.analysis_ready()
    cadence.note_attempt()
    clock.advance(4)
    cadence.note_activity()
    assert not cadence.analysis_ready()


def test_max_wait_boundary_resets_after_attempt_even_with_new_activity():
    clock = FakeClock()
    cadence = SmallCaptureCadence(quiet_seconds=10, interval_seconds=4,
                                  max_wait_seconds=30, clock=clock)
    clock.advance(29.999)
    cadence.note_activity()
    assert not cadence.planning_ready()
    clock.advance(0.001)
    assert cadence.analysis_ready()
    cadence.note_attempt()
    clock.advance(29.999)
    cadence.note_activity()
    assert not cadence.analysis_ready()
    clock.advance(0.001)
    assert cadence.analysis_ready()
    restarted = SmallCaptureCadence(quiet_seconds=10, interval_seconds=4,
                                    max_wait_seconds=30, clock=clock)
    restarted.note_activity()
    assert not restarted.analysis_ready()


@pytest.mark.parametrize("field", ["quiet_seconds", "interval_seconds", "max_wait_seconds"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 86400.001, float("inf"), float("nan"), "30"])
def test_invalid_durations_are_rejected(field, value):
    kwargs = {field: value}
    with pytest.raises(ValueError):
        SmallCaptureCadence(**kwargs)


def test_snapshot_reports_only_bounded_nonsecret_timing_state():
    clock = FakeClock()
    cadence = SmallCaptureCadence(quiet_seconds=10, interval_seconds=4,
                                  max_wait_seconds=30, clock=clock)
    cadence.note_attempt()
    clock.advance(3)
    snapshot = cadence.snapshot()
    assert snapshot == {
        "planning_ready": False,
        "analysis_ready": False,
        "quiet_remaining_seconds": 7.0,
        "max_wait_remaining_seconds": 27.0,
        "attempt_cooldown_remaining_seconds": 1.0,
    }
