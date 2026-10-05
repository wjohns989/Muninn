"""Real encrypted journal/outbox/publication with an isolated fake transport."""
import asyncio
import json
import sqlite3

import httpx
import pytest

from muninn.history.capture_journal import CaptureJournal
from muninn.history.cited_analysis_source import CitedAnalysisSource
from muninn.history.historical_batch import MODEL, BatchError
from muninn.history.historical_batch_worker import HistoricalBatchWorker, _decode, transport
from muninn.history.remote_accounting import status
from muninn.history.remote_policy import write_policy
from tests.test_historical_batch_jobs import fixture


def responses(archive, outbox, ident):
    items = outbox.read(ident)["items"]
    submitted = {"id": "batch_fixture", "model": MODEL, "endpoint": "/v1/chat/completions",
                 "completion_window": "24h", "status": "validating",
                 "request_counts": {"total": len(items)}}
    source = CitedAnalysisSource(archive)
    rows = []
    for item in items:
        text = source.reopen(item["window"])["text"]
        output = {"summary": "A cited observation.", "decisions": [], "open_items": [], "uncertainty": "",
                  "proposals": [{"type": "fact", "text": "An ordinary cited observation.",
                                 "quote": text[:64], "start": 0}]}
        rows.append({"custom_id": item["custom_id"], "error": None, "response": {
            "status_code": 200, "body": {"model": MODEL, "choices": [{"finish_reason": "stop",
                "message": {"content": json.dumps(output)}}]}}})
    terminal = {**submitted, "status": "completed", "results": rows,
                "request_counts": {"total": len(items), "completed": len(items), "failed": 0},
                "usage": {"cost": 0.001, "is_byok": False}}
    return submitted, terminal


def ready(**kwargs):
    return {"admission_ready": True, "usage_daily_usd": 0, "usage_monthly_usd": 0}


@pytest.mark.asyncio
async def test_submit_restart_poll_settle_publish_without_reconsent_or_resend(tmp_path):
    journal, archive, outbox, ident, bindings = fixture(tmp_path)
    journal.reserve_historical_batch(ident)
    submitted, terminal = responses(archive, outbox, ident)
    calls = []

    async def send(method, provider_id=None, body=None):
        calls.append(method)
        if method == "POST":
            assert journal.historical_batch_owner()["phase"] == "sent"
            assert outbox.read(ident)["state"] == "submission_unknown"
            assert status(journal.policy_root)["unresolved"] == 1
            assert list(body)[-1] == "requests"
            return submitted
        assert provider_id == "batch_fixture" and body is None
        return terminal

    worker = HistoricalBatchWorker(journal, authorize_submit=lambda generation: generation == 1,
                                   send=send, provider_status=ready, clock=lambda: 0)
    assert await worker.step()
    assert worker.status["state"] == "submitted"
    assert await worker.step()  # The polling interval prevents repeated HTTP.
    assert calls == ["POST"]
    write_policy(journal.policy_root, enabled=False, daily_usd=5, monthly_usd=50,
                 override_ceiling=False, fallback=lambda: (False, 1, 30, False))
    recovered = HistoricalBatchWorker(CaptureJournal(archive), send=send)
    assert await recovered.step()
    assert recovered.status["state"] == "passed"
    assert calls == ["POST", "GET"]
    assert status(journal.policy_root)["daily_cost_usd"] == 0.001
    assert journal.verify_publications(job_ids=[b[0] for b in bindings]) == 2
    assert not await recovered.step()
    assert outbox.read(ident)["state"] == "terminal_saved"  # No deletion.
    assert calls == ["POST", "GET"]


@pytest.mark.asyncio
async def test_no_retention_consent_never_calls_provider_or_reserves(tmp_path):
    journal, _archive, _outbox, ident, _bindings = fixture(tmp_path)
    journal.reserve_historical_batch(ident)
    async def forbidden(*args, **kwargs):
        pytest.fail("no batch egress without separately scoped consent")
    worker = HistoricalBatchWorker(journal, send=forbidden)
    assert await worker.step()
    assert worker.status["state"] == "consent_required"
    assert status(journal.policy_root)["unresolved"] == 0


@pytest.mark.asyncio
async def test_lost_post_response_never_retries_and_preserves_charge_uncertainty(tmp_path):
    journal, archive, outbox, ident, _bindings = fixture(tmp_path)
    journal.reserve_historical_batch(ident)
    calls = []
    async def lost(method, **kwargs):
        calls.append(method)
        raise TimeoutError()
    worker = HistoricalBatchWorker(journal, authorize_submit=lambda _: True,
                                   send=lost, provider_status=ready)
    with pytest.raises(TimeoutError):
        await worker.step()
    recovered = HistoricalBatchWorker(CaptureJournal(archive), authorize_submit=lambda _: True,
                                      send=lost, provider_status=ready)
    assert await recovered.step()
    assert recovered.status["state"] == "submission_unknown"
    assert outbox.read(ident)["state"] == "submission_unknown"
    assert status(journal.policy_root)["unresolved"] == 1
    assert calls == ["POST"]


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["model", "billing", "byok", "invalid_item"])
async def test_retained_invalid_terminal_blocks_next_checkpoint(tmp_path, defect):
    journal, archive, outbox, ident, bindings = fixture(tmp_path)
    journal.reserve_historical_batch(ident)
    submitted, terminal = responses(archive, outbox, ident)
    if defect == "model":
        terminal["model"] = "unrelated"
    elif defect == "billing":
        terminal["usage"] = {}
    elif defect == "byok":
        terminal["usage"]["is_byok"] = True
    else:
        terminal["results"][0]["response"]["body"]["choices"][0]["message"]["content"] = "not JSON"
    async def send(method, **kwargs):
        return submitted if method == "POST" else terminal
    worker = HistoricalBatchWorker(journal, authorize_submit=lambda _: True, send=send, provider_status=ready)
    await worker.step()
    worker.next_poll = 0
    worker.next_step = 0
    if defect == "invalid_item":
        assert await worker.step()
        assert worker.status["state"] == "checkpoint_unresolved"
        assert worker.status["invalid_items"] == 1
        assert journal.verify_publications(job_ids=[bindings[1][0]]) == 1
        assert status(journal.policy_root)["daily_cost_usd"] == 0.001
    else:
        with pytest.raises(BatchError):
            await worker.step()
        assert status(journal.policy_root)["unresolved"] == 1
    assert journal.historical_batch_owner()["phase"] == "sent"
    assert outbox.read(ident)["state"] == ("submitted" if defect == "model" else "terminal_saved")


@pytest.mark.asyncio
async def test_local_publication_failure_recovers_retained_result_without_http(tmp_path, monkeypatch):
    journal, archive, outbox, ident, bindings = fixture(tmp_path)
    journal.reserve_historical_batch(ident)
    submitted, terminal = responses(archive, outbox, ident)
    calls = []
    async def send(method, **kwargs):
        calls.append(method)
        return submitted if method == "POST" else terminal
    worker = HistoricalBatchWorker(journal, authorize_submit=lambda _: True, send=send, provider_status=ready)
    await worker.step()
    worker.next_poll = 0
    worker.next_step = 0
    real = CitedAnalysisSource.record_proposals
    def fail(*args, **kwargs):
        raise OSError("isolated failure")
    monkeypatch.setattr(CitedAnalysisSource, "record_proposals", fail)
    with pytest.raises(OSError):
        await worker.step()
    monkeypatch.setattr(CitedAnalysisSource, "record_proposals", real)
    with sqlite3.connect(journal.path) as db:
        db.execute("UPDATE history_analysis_jobs SET lease_until=0 WHERE lease_until IS NOT NULL")
    recovered = HistoricalBatchWorker(CaptureJournal(archive), send=send)
    assert await recovered.step()
    assert recovered.status["state"] == "passed"
    assert journal.verify_publications(job_ids=[b[0] for b in bindings]) == 2
    assert calls == ["POST", "GET"]


@pytest.mark.parametrize("raw", [b'{"id":1,"id":2}', b'{"a":NaN}', b'[]', b'not JSON'])
def test_strict_transport_decoder(raw):
    with pytest.raises(BatchError):
        _decode(raw)


def test_response_bound_before_decoding():
    with pytest.raises(BatchError, match="bound"):
        _decode(b" " * (4 * 1024 * 1024 + 1))


@pytest.mark.asyncio
async def test_service_uses_existing_consumer_and_drains_shutdown(tmp_path):
    from muninn.history.service import HistoryService
    journal, _archive, _outbox, ident, _bindings = fixture(tmp_path)
    journal.reserve_historical_batch(ident)
    service = HistoryService.__new__(HistoryService)
    service._require_capture_journal = lambda: journal
    started, finish = asyncio.Event(), asyncio.Event()
    class Worker:
        status = {"state": "working"}
        async def step(self):
            started.set()
            await finish.wait()
            self.status = {"state": "durable"}
    service._historical_batch_worker = Worker()
    task = asyncio.create_task(service._process_secure_batch_once())
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()  # Does not abandon an active transport/write.
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert service._historical_batch_worker.status["state"] == "durable"


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", [None, "redirect", "duplicate", "large"])
async def test_actual_transport_fixed_origin_single_request_bounded_json(monkeypatch, defect):
    from muninn.history import llm_settings
    monkeypatch.setattr(llm_settings, "api_key", lambda: "test-only-placeholder")
    real_client = httpx.AsyncClient
    calls = []
    def handler(request):
        calls.append(request)
        assert str(request.url) == "https://openrouter.ai/api/v1/batches/batch_fixture"
        assert request.method == "GET"
        if defect == "redirect":
            return httpx.Response(302, headers={"Location": "https://untrusted.invalid/"})
        if defect == "duplicate":
            return httpx.Response(200, content=b'{"id":1,"id":2}')
        if defect == "large":
            return httpx.Response(200, content=b" " * (4 * 1024 * 1024 + 1))
        return httpx.Response(200, json={"id": "batch_fixture"})
    def client(*args, **kwargs):
        assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
        return real_client(*args, **kwargs, transport=httpx.MockTransport(handler))
    monkeypatch.setattr(httpx, "AsyncClient", client)
    if defect:
        with pytest.raises(BatchError):
            await transport("GET", provider_id="batch_fixture")
    else:
        assert await transport("GET", provider_id="batch_fixture") == {"id": "batch_fixture"}
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_recovery_consumer_starts_with_new_work_automation_disabled(tmp_path, monkeypatch):
    from muninn.history.service import HistoryService
    journal, archive, _outbox, ident, _bindings = fixture(tmp_path)
    journal.reserve_historical_batch(ident)
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    for name in ("MUNINN_CAPTURE_ENRICHMENT", "MUNINN_CAPTURE_AUTO_ANALYSIS",
                 "MUNINN_SECURE_AUTO_ANALYSIS", "MUNINN_HISTORY_INDEX_AUTO"):
        monkeypatch.setenv(name, "0")
    service = HistoryService(None, tmp_path / "history_vault", home=tmp_path,
        secure_archive_root=archive.root, archive_passphrase="test-only portable passphrase")
    service._capture_journal = journal
    async def wait():
        await asyncio.Event().wait()
    for name in ("_secure_capture_loop", "_secure_scan_loop", "_secure_search_loop", "_secure_analysis_loop"):
        monkeypatch.setattr(service, name, wait)
    await service.start()
    try:
        consumer = service._secure_analysis_task
        assert consumer is not None
        assert service._secure_capture_plan_task is None
        await service.start()
        assert service._secure_analysis_task is consumer
    finally:
        await service.stop()
