"""Automatic capture cadence with isolated stores; no real provider or service."""
import asyncio

import pytest

from muninn.history.service import HistoryService
from tests.test_capture_enrichment_service import setup_service
from tests.test_capture_window_jobs import window_fixture
from tests.test_secure_analysis_journal import _result, _target


def test_planning_gate_yields_to_foreground_search(tmp_path):
    journal, _archive, _receipt = window_fixture(tmp_path)
    assert journal.capture_planning_ready()
    search_id = journal.enqueue_search("observation")
    assert not journal.capture_planning_ready()
    search = journal.claim_search()
    assert not journal.capture_planning_ready()
    assert journal.cancel_search(search_id)
    assert journal.capture_planning_ready()


def test_planning_gate_skips_full_automatic_queue(tmp_path):
    journal, _archive, receipt = window_fixture(tmp_path, text="Long ordinary message. " * 4500)
    assert journal.queue_capture_windows(receipt, limit=32)["queued"] == 24
    assert not journal.capture_planning_ready()


def test_capture_only_claim_never_dispatches_search_analysis(tmp_path):
    journal, archive, receipt = window_fixture(tmp_path)
    journal.queue_capture_windows(receipt, limit=1)
    search_id = journal.enqueue_search("needle")
    search = journal.claim_search()
    assert journal.finish_search(search_id, search.lease_token, _result(), analysis_target=_target(archive))
    job = journal.claim_analysis(include_capture=True, include_search=False)
    assert job.lane == 1
    assert journal.get_search_job(search_id)["analysis_job_id"] != job.job_id
    assert journal.claim_analysis(include_search=False) is None
    assert journal.claim_analysis().lane == 0


@pytest.mark.parametrize("value", [0, 1, None, "false"])
def test_search_admission_requires_boolean(tmp_path, value):
    journal, _archive, _receipt = window_fixture(tmp_path)
    with pytest.raises(ValueError, match="lane admission"):
        journal.claim_analysis(include_search=value)


def enabled_service(monkeypatch, tmp_path):
    from muninn.history.capture_cadence import SmallCaptureCadence as CaptureCadence
    service, archive, source = setup_service(monkeypatch, tmp_path)
    monkeypatch.setenv("MUNINN_CAPTURE_AUTO_ANALYSIS", "1")
    monkeypatch.setenv("MUNINN_SECURE_AUTO_ANALYSIS", "0")
    now = [0.0]
    service._capture_cadence = CaptureCadence(clock=lambda: now[0])
    return service, archive, source, now


@pytest.mark.asyncio
async def test_automatic_planner_requires_quiet_and_foreground_gate(monkeypatch, tmp_path):
    service, archive, source, now = enabled_service(monkeypatch, tmp_path)
    source.write_text('{"type":"event_msg","payload":{"type":"user_message","message":"Actual isolated chat."}}\n')
    await service.capture(str(source), "codex")
    journal = service._require_capture_journal()
    prepare = journal.queue_capture_windows
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return prepare(*args, **kwargs)

    monkeypatch.setattr(journal, "queue_capture_windows", counted)
    assert not await service._process_capture_plan_once(automatic=True)
    now[0] = 300.0
    search_id = journal.enqueue_search("chat")
    assert not await service._process_capture_plan_once(automatic=True)
    now[0] = 1800.0
    service._capture_cadence.note_activity()
    assert service._capture_cadence.planning_ready()
    assert not await service._process_capture_plan_once(automatic=True)
    assert calls == []
    assert journal.cancel_search(search_id)
    assert await service._process_capture_plan_once(automatic=True)
    assert calls == [1]
    assert journal.claim_analysis(include_capture=True, include_search=False).lane == 1


@pytest.mark.asyncio
async def test_search_arriving_during_plan_prevents_model_claim(monkeypatch, tmp_path):
    service, _archive, source, now = enabled_service(monkeypatch, tmp_path)
    source.write_text('{"type":"event_msg","payload":{"type":"user_message","message":"Isolated new chat."}}\n')
    await service.capture(str(source), "codex")
    now[0] = 300.0
    journal = service._require_capture_journal()
    prepare = journal.queue_capture_windows
    search_ids = []

    def interrupted_by_search(*args, **kwargs):
        search_ids.append(journal.enqueue_search("chat"))
        return prepare(*args, **kwargs)

    monkeypatch.setattr(journal, "queue_capture_windows", interrupted_by_search)
    assert await service._process_capture_plan_once(automatic=True)
    assert journal.claim_analysis(include_capture=True, include_search=False) is None
    assert journal.cancel_search(search_ids[0])
    assert journal.claim_analysis(include_capture=True, include_search=False).lane == 1


@pytest.mark.asyncio
async def test_disabled_auto_flag_does_not_prepare(monkeypatch, tmp_path):
    service, _archive, _source, now = enabled_service(monkeypatch, tmp_path)
    now[0] = 300.0
    monkeypatch.setenv("MUNINN_CAPTURE_AUTO_ANALYSIS", "0")
    monkeypatch.setattr(service._require_capture_journal(), "next_capture_plan",
                        lambda: pytest.fail("disabled planner accessed source"))
    assert not await service._process_capture_plan_once(automatic=True)


@pytest.mark.asyncio
async def test_one_analysis_consumer_and_one_cpu_planner_start_stop(monkeypatch, tmp_path):
    service, _archive, _source, _now = enabled_service(monkeypatch, tmp_path)
    blockers = []

    async def wait():
        event = asyncio.Event()
        blockers.append(event)
        await event.wait()

    for name in ("_secure_capture_loop", "_secure_scan_loop", "_secure_search_loop",
                 "_secure_capture_plan_loop", "_secure_analysis_loop"):
        monkeypatch.setattr(service, name, wait)
    monkeypatch.setenv("MUNINN_HISTORY_INDEX_AUTO", "0")
    await service.start()
    model_worker = service._secure_analysis_task
    planner = service._secure_capture_plan_task
    assert model_worker is not None and planner is not None
    await service.start()
    assert service._secure_analysis_task is model_worker
    assert service._secure_capture_plan_task is planner
    await asyncio.sleep(0)
    assert len(blockers) == 5
    await service.stop()
    assert model_worker.done() and planner.done()
    assert service._secure_capture_plan_task is None


@pytest.mark.asyncio
async def test_analysis_loop_preserves_disabled_search_lane(monkeypatch, tmp_path):
    service, _archive, _source, now = enabled_service(monkeypatch, tmp_path)
    now[0] = 300.0
    calls = []

    async def once(**kwargs):
        calls.append(kwargs)
        raise asyncio.CancelledError

    monkeypatch.setattr(service, "_process_secure_analysis_once", once)
    with pytest.raises(asyncio.CancelledError):
        await service._secure_analysis_loop()
    assert calls == [{"include_capture": True, "include_search": False}]


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", ["MUNINN_CAPTURE_QUIET_SECONDS", "MUNINN_CAPTURE_INTERVAL_SECONDS",
                                     "MUNINN_CAPTURE_MAX_WAIT_SECONDS"])
async def test_invalid_cadence_cannot_leave_background_tasks(monkeypatch, tmp_path, setting):
    service, _archive, _source, _now = enabled_service(monkeypatch, tmp_path)
    monkeypatch.setenv(setting, "nan")
    with pytest.raises(ValueError):
        await service.start()
    assert service._secure_capture_task is None
    assert service._secure_search_task is None
    assert service._secure_analysis_task is None
    assert service._secure_capture_plan_task is None


@pytest.mark.asyncio
async def test_archive_activity_resets_quiet_but_rejected_capture_does_not(monkeypatch, tmp_path):
    service, _archive, source, now = enabled_service(monkeypatch, tmp_path)
    now[0] = 300.0
    assert service._capture_cadence.planning_ready()
    with pytest.raises((ValueError, FileNotFoundError)):
        await service.capture(str(source), "codex")
    assert service._capture_cadence.planning_ready()
    source.write_text('{"type":"event_msg","payload":{"type":"user_message","message":"Accepted isolated capture."}}\n')
    await service.capture(str(source), "codex")
    assert not service._capture_cadence.planning_ready()
    now[0] = 600.0
    assert service._capture_cadence.planning_ready()


@pytest.mark.asyncio
async def test_revoked_capture_flag_removes_capture_admission(monkeypatch, tmp_path):
    service, _archive, _source, now = enabled_service(monkeypatch, tmp_path)
    now[0] = 300.0
    monkeypatch.setenv("MUNINN_CAPTURE_AUTO_ANALYSIS", "0")
    monkeypatch.setenv("MUNINN_SECURE_AUTO_ANALYSIS", "1")
    calls = []

    async def once(**kwargs):
        calls.append(kwargs)
        raise asyncio.CancelledError

    monkeypatch.setattr(service, "_process_secure_analysis_once", once)
    with pytest.raises(asyncio.CancelledError):
        await service._secure_analysis_loop()
    assert calls == [{"include_capture": False, "include_search": True}]


@pytest.mark.asyncio
@pytest.mark.parametrize("continuous_capture", [False, True])
async def test_timer_loops_publish_new_capture_without_manual_batch(monkeypatch, tmp_path,
                                                                     continuous_capture):
    from muninn.history import secure_analysis
    from muninn.history.capture_cadence import SmallCaptureCadence
    service, _archive, source_path, now = enabled_service(monkeypatch, tmp_path)
    source_path.write_text('{"type":"event_msg","payload":{"type":"user_message","message":"Remember this isolated project decision."}}\n')
    await service.capture(str(source_path), "codex")
    journal = service._require_capture_journal()
    receipt = journal.pending_enrichment()[0]
    if continuous_capture:
        service._capture_cadence = SmallCaptureCadence(
            quiet_seconds=10, interval_seconds=4, max_wait_seconds=30,
            clock=lambda: now[0])
        for moment in (5, 10, 15, 20, 25, 30):
            now[0] = float(moment)
            service._capture_cadence.note_activity()
        assert service._capture_cadence.analysis_ready()
    else:
        now[0] = 300.0
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("local capture must not read remote policy or credentials")

    monkeypatch.setattr("muninn.history.auto_routing.remote_policy_snapshot", forbidden)

    async def analyze(history, source, descriptor, **kwargs):
        assert kwargs["allow_remote"] is False
        assert kwargs["expected_remote_generation"] == -1
        assert await kwargs["before_remote"]() is False
        calls.append(source.reopen(descriptor)["text"])
        result = {"status": "ok", "provider": "ollama", "model": "isolated-model-stub",
                  "analysis": {"summary": "No supported claim extracted.", "decisions": [],
                               "open_items": [], "uncertainty": "Timer plumbing only, not quality proof."}}
        return {**result, "extraction": {"format": 1, "window": descriptor, "proposals": [],
                                         "model_identity": "a" * 64, "result": result}}

    monkeypatch.setattr(secure_analysis, "analyze_cited_window", analyze)
    consumer = asyncio.create_task(service._secure_analysis_loop())
    planner = asyncio.create_task(service._secure_capture_plan_loop())
    try:
        async def completed():
            while journal.capture_window_status(receipt)["state"] != "completed":
                await asyncio.sleep(0.02)
        await asyncio.wait_for(completed(), 5)
        state = journal.capture_window_status(receipt)
        assert state["acknowledged"] == state["windows"] == 1
        assert len(calls) == 1 and "isolated project decision" in calls[0]
        assert journal.verify_all() == 0
        assert not service._capture_cadence.analysis_ready()
    finally:
        consumer.cancel()
        planner.cancel()
        await asyncio.gather(consumer, planner, return_exceptions=True)
