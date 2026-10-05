"""Bounded remote catch-up with isolated stores and fake clocks/providers."""
import asyncio

import pytest

from muninn.history.capture_cadence import CaptureBacklogDrain, SmallCaptureCadence


def test_busy_admission_does_not_permanently_stop_catchup():
    drain = CaptureBacklogDrain(160.0, clock=lambda: 100.0)
    drain.halt("remote_admission_busy")
    assert drain.active(pending=100, remote_enabled=True)
    drain.halt("remote_cost_unresolved")
    assert not drain.active(pending=100, remote_enabled=True)
from tests.test_capture_automatic_service import enabled_service
from tests.test_capture_window_jobs import window_fixture


def test_drain_requires_consent_backlog_and_unexpired_deadline():
    now = [100.0]
    drain = CaptureBacklogDrain(160.0, clock=lambda: now[0])
    assert drain.active(pending=100, remote_enabled=True)
    assert not drain.active(pending=99, remote_enabled=True)
    assert not drain.active(pending=100, remote_enabled=False)
    now[0] = 160.0
    assert not drain.active(pending=100, remote_enabled=True)


def test_drain_halts_and_restart_keeps_original_expiry():
    now = [100.0]
    drain = CaptureBacklogDrain(160.0, clock=lambda: now[0])
    drain.halt("remote_cost_unresolved")
    assert not drain.active(pending=100, remote_enabled=True)
    assert drain.snapshot()["halted_reason"] == "remote_cost_unresolved"
    now[0] = 161.0
    assert not CaptureBacklogDrain(160.0, clock=lambda: now[0]).active(
        pending=100, remote_enabled=True)


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), -1, 10901])
def test_invalid_or_unbounded_drain_deadline_fails_closed(deadline):
    with pytest.raises(ValueError):
        CaptureBacklogDrain(deadline, clock=lambda: 100.0)


def test_activity_does_not_reset_drain_attempt_interval():
    now = [0.0]
    cadence = SmallCaptureCadence(clock=lambda: now[0])
    assert cadence.attempt_ready()
    cadence.note_attempt()
    for moment in (5.0, 10.0, 29.999):
        now[0] = moment
        cadence.note_activity()
        assert not cadence.attempt_ready()
    now[0] = 30.0
    cadence.note_activity()
    assert cadence.attempt_ready()
    assert not cadence.analysis_ready()  # Ordinary capture still waits for quiet.


def test_remote_only_claim_leaves_local_bound_jobs_untouched(tmp_path):
    journal, _archive, receipt = window_fixture(tmp_path)
    journal.queue_capture_windows(receipt, limit=1)
    journal.queue_capture_windows(receipt, limit=1, remote_policy_generation=2)
    job = journal.claim_analysis(include_capture=True, include_search=False,
                                 capture_remote_only=True)
    assert job.remote_policy_generation == 2 and job.target["ordinal"] == 1
    local = journal.claim_analysis(include_capture=True, include_search=False)
    assert local.remote_policy_generation == -1 and local.target["ordinal"] == 0
    assert journal.verify_all() == 0


@pytest.mark.asyncio
async def test_drain_planner_bypasses_quiet_but_retains_foreground_gate(monkeypatch, tmp_path):
    service, _archive, source, now = enabled_service(monkeypatch, tmp_path)
    source.write_text('{"type":"event_msg","payload":{"type":"user_message","message":"Isolated catch-up observation."}}\n')
    await service.capture(str(source), "codex")
    monkeypatch.setattr(service, "_capture_drain_active", lambda: True)
    journal = service._require_capture_journal()
    search = journal.enqueue_search("chat")
    assert not await service._process_capture_plan_once(automatic=True)
    assert journal.cancel_search(search)
    assert now[0] == 0 and not service._capture_cadence.planning_ready()
    assert await service._process_capture_plan_once(automatic=True)


@pytest.mark.asyncio
async def test_drain_consumer_claims_only_remote_windows_during_activity(monkeypatch, tmp_path):
    service, _archive, _source, _now = enabled_service(monkeypatch, tmp_path)
    monkeypatch.setattr(service, "_capture_drain_active", lambda: True)
    calls = []

    async def once(**kwargs):
        calls.append(kwargs)
        raise asyncio.CancelledError

    monkeypatch.setattr(service, "_process_secure_analysis_once", once)
    with pytest.raises(asyncio.CancelledError):
        await service._secure_analysis_loop()
    assert calls == [{"include_capture": True, "include_search": False,
                      "capture_remote_only": True}]


@pytest.mark.asyncio
async def test_generation_revocation_halts_drain_without_inference(monkeypatch, tmp_path):
    from muninn.history.remote_policy import write_policy

    service, _archive, source, _now = enabled_service(monkeypatch, tmp_path)
    monkeypatch.setenv("MUNINN_CAPTURE_AUTO_REMOTE", "1")
    write_policy(service.data_dir, enabled=True, daily_usd=5, monthly_usd=50,
                 override_ceiling=False, fallback=lambda: (False, 1, 30, False))
    source.write_text('{"type":"event_msg","payload":{"type":"user_message","message":"Isolated revocation observation."}}\n')
    await service.capture(str(source), "codex")
    await service._process_capture_plan_once()
    write_policy(service.data_dir, enabled=True, daily_usd=4, monthly_usd=50,
                 override_ceiling=False, fallback=lambda: (False, 1, 30, False))
    service._capture_drain = CaptureBacklogDrain(160.0, clock=lambda: 100.0)

    async def forbidden(*args, **kwargs):
        pytest.fail("Revoked generation must dispatch neither remote nor local inference")

    monkeypatch.setattr("muninn.history.secure_analysis.analyze_cited_window", forbidden)
    assert await service._process_secure_analysis_once(include_capture=True,
                                                      capture_remote_only=True)
    assert service._capture_drain.snapshot()["halted_reason"] == "remote_consent_revoked"
    assert not service._capture_drain.active(pending=100, remote_enabled=True)
    assert not service._capture_drain_active()
    with service._require_capture_journal()._connect() as db:
        row = db.execute("SELECT state,error_code,remote_dispatched FROM history_analysis_jobs").fetchone()
    assert tuple(row) == ("retry", "remote_consent_revoked", 0)
