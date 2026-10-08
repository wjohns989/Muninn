"""Actual encrypted local workflow, synthetic provider only; no live secrets."""
import asyncio
import json

import httpx
import pytest

from muninn.history.classification_worker import (
    process_classification, _dispatch_period_open as utc_period_open,
    _model_context_bound as public_model_bound,
)
from muninn.history.memory_classification import prepare_classification
from muninn.history.memory_ledger import MemoryLedger
from muninn.history.remote_accounting import reserve
from tests.test_classification_enrollment import cohort_ack
from tests.test_remote_accounting import policy, READY


@pytest.fixture(autouse=True)
def completed_paid_checkpoint(monkeypatch):
    from muninn.history.capture_journal import CaptureJournal
    # Isolate classification after the existing checkpoint gate; real paid
    # checkpoint ownership/publication proofs have their own integration tests.
    monkeypatch.setattr(CaptureJournal, "historical_batch_owner", lambda _: {"phase": "passed", "id": "f" * 32})
    from muninn.history import classification_worker as module
    monkeypatch.setattr(module, "_model_context_bound", lambda: 1050000)
    monkeypatch.setattr(module, "_dispatch_period_open", lambda: True)


def response(body, *, cost=0.003, malformed=False):
    payload = json.loads(body["messages"][1]["content"])
    rows = [{"id": row["id"], "bucket": "preference", "disposition": "accepted",
        "evidence_refs": [row["id"]], "reason": "source_supported", "confidence": 0.99}
        for row in payload["candidates"]]
    return 200, {"usage": {"cost": cost}, "model": "synthetic-luna", "choices": [
        {"message": {"content": "bad" if malformed else json.dumps({"items": rows})}}]}


@pytest.mark.asyncio
async def test_grouped_workflow_publishes_once_and_recovers_without_provider(tmp_path):
    journal, archive, refs = cohort_ack(tmp_path)
    policy(tmp_path)
    calls = []
    async def send(body, *, before_post):
        assert await before_post()
        calls.append(body)
        assert body["provider"] == {"zdr": True, "data_collection": "deny", "require_parameters": True,
                                    "max_price": {"prompt": 0.25, "completion": 1, "request": 0}}
        assert all(ref not in json.dumps(body) for ref in refs)
        return response(body)
    assert await process_classification(journal, enabled=lambda: True, send=send, key_status=lambda: READY)
    assert len(calls) == 1 and journal.classification_status() == {"published": 1}
    assert all(MemoryLedger(archive, read_only=True).get(ref)["placement"]["status"] == "accepted" for ref in refs)
    assert not await process_classification(journal, enabled=lambda: False, recovery_only=True, send=send)
    assert journal.verify_classifications() == 1


@pytest.mark.asyncio
async def test_distinct_cohorts_continue_after_lifetime_pilot(tmp_path, monkeypatch):
    from muninn.history import classification_jobs
    # Separate two valid singleton review jobs; cohort grouping itself has
    # independent tests, and one extraction intentionally permits at most 12.
    monkeypatch.setattr(classification_jobs, "related_cohorts", lambda rows: [[row[0]] for row in rows])
    journal, archive, refs = cohort_ack(tmp_path, count=2)
    policy(tmp_path)
    calls = []
    async def send(body, *, before_post):
        assert await before_post()
        calls.append(body)
        return response(body)
    for _ in range(2):
        assert await process_classification(journal, enabled=lambda: True, send=send, key_status=lambda: READY), (
            len(calls), journal.classification_status())
    assert len(calls) == 2 and journal.classification_status() == {"published": 2}
    assert journal.verify_classifications() == 2
    assert all(MemoryLedger(archive, read_only=True).get(ref)["placement"]["status"] == "accepted" for ref in refs)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["headroom", "catalog", "price", "rollover"])
async def test_presend_failure_stays_pending_without_charge(tmp_path, monkeypatch, failure):
    from muninn.history import classification_worker as module
    from muninn.history.remote_accounting import AdmissionError, status
    journal, archive, _refs = cohort_ack(tmp_path)
    policy(tmp_path)
    checks, posts = [], []
    def key_status():
        checks.append(True)
        return {**READY, "usage_daily_usd": 4.0 if failure == "headroom" and len(checks) > 1 else 0}
    if failure == "catalog":
        def unavailable():
            raise AdmissionError("classification_catalog_unavailable")
        monkeypatch.setattr(module, "_model_context_bound", unavailable)
    async def send(body, *, before_post):
        if failure == "price":
            body["provider"].pop("max_price")
        if failure == "rollover":
            monkeypatch.setattr(module, "_dispatch_period_open", lambda: False)
        if not await before_post():
            return None
        posts.append(True)
        return response(body)
    assert not await process_classification(journal, enabled=lambda: True, send=send, key_status=key_status)
    assert posts == [] and status(tmp_path)["unresolved"] == 0
    assert journal.classification_status() == {"pending": 1}


def test_price_contract_full_context_bound_and_mutations(tmp_path):
    from decimal import Decimal
    from muninn.history import classification_worker as module
    journal, archive, refs = cohort_ack(tmp_path)
    prepared = prepare_classification(MemoryLedger(archive, read_only=True), refs)
    body = module.request_body(prepared)
    assert module._cost_ceiling(body, 1050000) == Decimal("1.3125")
    import copy
    for field, value in [("models", ["other-model"]), ("model", "other-model"),
                         ("max_completion_tokens", 4097), ("tools", []),
                         ("plugins", [{"id": "web"}]), ("service_tier", "priority")]:
        altered = copy.deepcopy(body)
        altered[field] = value
        with pytest.raises(module.AdmissionError, match="price_contract"):
            module._cost_ceiling(altered, 1050000)
    for context in (None, True, 0, "1050000", 8000001):
        with pytest.raises(module.AdmissionError):
            module._cost_ceiling(body, context)


@pytest.mark.parametrize("offset,allowed", [(86100, True), (86280, False), (86399, False), (86400, True)])
def test_utc_boundary_guard(monkeypatch, offset, allowed):
    from muninn.history import classification_worker as module
    monkeypatch.setattr(module.time, "time", lambda: offset)
    assert utc_period_open() is allowed


@pytest.mark.parametrize("data,valid", [
    ([{"id": "openai/gpt-6-luna-pro", "context_length": 1050000}], True),
    ([], False), ([{"id": "other-model", "context_length": 1050000}], False),
    ([{"id": "openai/gpt-6-luna-pro", "context_length": True}], False),
    ([{"id": "openai/gpt-6-luna-pro", "context_length": "1050000"}], False),
    ([None], False),
    ([{"id": "openai/gpt-6-luna-pro", "context_length": 1050000}] * 2, False),
])
def test_public_catalog_identity_and_bounds(monkeypatch, data, valid):
    from muninn.history import classification_worker as module
    # Capture original before the fixture's synthetic metadata injection.
    client_class = httpx.Client
    def handler(request):
        assert request.method == "GET" and request.url.host == "openrouter.ai"
        assert request.url.params["q"] == "openai/gpt-6-luna-pro"
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"data": data})
    def client(**kwargs):
        assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
        return client_class(transport=httpx.MockTransport(handler), **kwargs)
    monkeypatch.setattr(module.httpx, "Client", client)
    if valid:
        assert public_model_bound() == 1050000
    else:
        with pytest.raises(module.AdmissionError, match="catalog_unavailable"):
            public_model_bound()


@pytest.mark.asyncio
@pytest.mark.parametrize("block", ["foreground", "owned", "revoked", "no_owner"])
async def test_new_dispatch_refused_under_current_gates(tmp_path, monkeypatch, block):
    journal, archive, refs = cohort_ack(tmp_path)
    policy(tmp_path)
    if block == "foreground":
        journal.enqueue_search("foreground priority")
    elif block == "owned":
        monkeypatch.setattr(journal, "historical_batch_owner", lambda: {"phase": "sent"})
    elif block == "no_owner":
        monkeypatch.setattr(journal, "historical_batch_owner", lambda: None)
    def forbidden(*args, **kwargs):
        raise AssertionError("No provider or credential access is allowed")
    assert not await process_classification(journal, enabled=lambda: block != "revoked", send=forbidden, key_status=forbidden)
    assert journal.classification_status() == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["schema", "bill", "timeout"])
async def test_charged_or_unknown_failure_never_dispatches_again(tmp_path, failure):
    journal, archive, refs = cohort_ack(tmp_path)
    policy(tmp_path)
    calls = []
    async def send(body, *, before_post):
        assert await before_post()
        calls.append(body)
        if failure == "timeout":
            raise httpx.ReadTimeout("synthetic timeout")
        return response(body, cost=None if failure == "bill" else 0.003, malformed=failure == "schema")
    await process_classification(journal, enabled=lambda: True, send=send, key_status=lambda: READY)
    assert journal.classification_status() == {"needs_user" if failure == "schema" else "outcome_unknown": 1}
    assert not await process_classification(journal, enabled=lambda: True, send=send, key_status=lambda: READY)
    assert len(calls) == 1 and journal.verify_classifications() == 1


def test_released_unsent_crash_reclaims_exact_prepared_input(tmp_path):
    import time
    journal, archive, refs = cohort_ack(tmp_path)
    generation = policy(tmp_path).generation
    journal.discover_classifications()
    job = journal.claim_classification()
    plan = prepare_classification(MemoryLedger(archive, read_only=True), refs)
    journal.prepare_classification_job(job["job_id"], job["lease"], plan)
    admission = reserve(tmp_path, generation, READY, classification_job=job["job_id"], classification_input=plan.input_sha256)
    admission.mark_unknown()
    journal.mark_classification_dispatch(job["job_id"], job["lease"], admission.identifier, generation)
    admission.release_unsent()  # Simulated crash before journal cleanup, no POST.
    reclaimed = journal.claim_classification(now=time.time() + 200)
    assert reclaimed["job_id"] == job["job_id"] and reclaimed["admission"] is None
    journal.prepare_classification_job(reclaimed["job_id"], reclaimed["lease"], plan)
    assert journal.verify_classifications() == 1


@pytest.mark.asyncio
async def test_shutdown_drains_one_sent_response_without_redispatch(tmp_path):
    journal, archive, refs = cohort_ack(tmp_path)
    policy(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def send(body, *, before_post):
        assert await before_post()
        calls.append(body)
        entered.set()
        await release.wait()
        return response(body)
    task = asyncio.create_task(process_classification(journal, enabled=lambda: True, send=send, key_status=lambda: READY))
    await asyncio.wait_for(entered.wait(), timeout=20)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(calls) == 1 and journal.classification_status() == {"published": 1}


@pytest.mark.asyncio
async def test_revoke_during_client_entry_refuses_post_and_releases_unsent(tmp_path, monkeypatch):
    from muninn.history import classification_worker as module
    from muninn.history.remote_accounting import status
    journal, archive, refs = cohort_ack(tmp_path)
    policy(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    enabled = [True]
    class Client:
        def __init__(self, **kwargs):
            assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
        async def __aenter__(self):
            entered.set()
            await release.wait()
            return self
        async def __aexit__(self, *args):
            pass
        def stream(self, *args, **kwargs):
            raise AssertionError("Revoked request must not reach transport")
    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    monkeypatch.setattr(module.llm_settings, "api_key", lambda: "synthetic-placeholder")
    task = asyncio.create_task(process_classification(journal, enabled=lambda: enabled[0], key_status=lambda: READY))
    await asyncio.wait_for(entered.wait(), timeout=20)
    enabled[0] = False
    release.set()
    assert not await task
    assert status(tmp_path)["unresolved"] == 0
    assert journal.classification_status() == {"pending": 1}


@pytest.mark.asyncio
async def test_human_rejection_during_key_lookup_prevents_charge(tmp_path):
    from muninn.history.remote_accounting import status
    journal, archive, refs = cohort_ack(tmp_path)
    policy(tmp_path)
    posted = []
    def key_status():
        MemoryLedger(archive).resolve_review(refs[0], state="rejected", expected_state="provisional", reason="user_rejected")
        return READY
    async def send(body, *, before_post):
        assert await before_post()
        posted.append(body)
        return response(body)
    assert await process_classification(journal, enabled=lambda: True, send=send, key_status=key_status)
    assert posted == [] and status(tmp_path)["unresolved"] == 0
    assert journal.classification_status() == {"needs_user": 1}
    assert MemoryLedger(archive, read_only=True).get(refs[0])["state"] == "rejected"


@pytest.mark.asyncio
async def test_cancel_staged_publication_drains_native_writer(tmp_path, monkeypatch):
    import threading
    journal, archive, refs = cohort_ack(tmp_path)
    policy(tmp_path)
    async def send(body, *, before_post):
        assert await before_post()
        return response(body)
    original = journal.publish_classification
    monkeypatch.setattr(journal, "publish_classification", lambda _job: None)
    await process_classification(journal, enabled=lambda: True, send=send, key_status=lambda: READY)
    assert journal.classification_status() == {"staged": 1}
    entered, release = threading.Event(), threading.Event()
    def blocked(job_id):
        entered.set()
        assert release.wait(10)
        return original(job_id)
    monkeypatch.setattr(journal, "publish_classification", blocked)
    task = asyncio.create_task(process_classification(journal, enabled=lambda: False, recovery_only=True))
    assert await asyncio.to_thread(entered.wait, 10)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert journal.classification_status() == {"published": 1}


@pytest.mark.asyncio
async def test_heartbeat_cancellation_drains_its_native_writer(tmp_path, monkeypatch):
    import threading
    from muninn.history import classification_worker as module
    journal, archive, refs = cohort_ack(tmp_path)
    policy(tmp_path)
    entered, release = threading.Event(), threading.Event()
    reply_release = asyncio.Event()
    original = journal.heartbeat_classification
    def heartbeat(job_id, lease):
        entered.set()
        assert release.wait(10)
        return original(job_id, lease)
    monkeypatch.setattr(journal, "heartbeat_classification", heartbeat)
    monkeypatch.setattr(module, "_HEARTBEAT_SECONDS", 0.01)
    async def send(body, *, before_post):
        assert await before_post()
        await reply_release.wait()
        return response(body)
    task = asyncio.create_task(process_classification(journal, enabled=lambda: True, send=send, key_status=lambda: READY))
    assert await asyncio.to_thread(entered.wait, 10)
    reply_release.set()
    await asyncio.sleep(0.1)
    assert not task.done()
    release.set()
    assert await task
    assert journal.classification_status() == {"published": 1}


@pytest.mark.asyncio
async def test_callback_timeout_drains_marker_before_proven_unsent_cleanup(tmp_path, monkeypatch):
    import threading
    from muninn.history.remote_accounting import status
    journal, archive, refs = cohort_ack(tmp_path)
    policy(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = journal.mark_classification_dispatch
    def mark(*args):
        entered.set()
        assert release.wait(10)
        return original(*args)
    monkeypatch.setattr(journal, "mark_classification_dispatch", mark)
    posted = []
    async def send(body, *, before_post):
        async with asyncio.timeout(None) as deadline:
            async def expire_after_marker():
                assert await asyncio.to_thread(entered.wait, 10)
                deadline.reschedule(asyncio.get_running_loop().time() + 0.01)
            timer = asyncio.create_task(expire_after_marker())
            try:
                assert await before_post()
                posted.append(body)
                return response(body)
            finally:
                timer.cancel()
                await asyncio.gather(timer, return_exceptions=True)
    task = asyncio.create_task(process_classification(journal, enabled=lambda: True, send=send, key_status=lambda: READY))
    assert await asyncio.to_thread(entered.wait, 10)
    await asyncio.sleep(0.1)
    assert not task.done() and posted == []
    release.set()
    assert not await task
    assert posted == [] and status(tmp_path)["unresolved"] == 0
    assert journal.classification_status() == {"pending": 1}


@pytest.mark.asyncio
async def test_service_start_recovers_staged_work_with_new_inference_disabled(tmp_path, monkeypatch):
    from muninn.history.service import HistoryService
    from muninn.history import llm_settings
    journal, archive, refs = cohort_ack(tmp_path)
    policy(tmp_path)
    async def send(body, *, before_post):
        assert await before_post()
        return response(body)
    monkeypatch.setattr(journal, "publish_classification", lambda _job: None)
    await process_classification(journal, enabled=lambda: True, send=send, key_status=lambda: READY)
    assert journal.classification_status() == {"staged": 1}
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(archive.root))
    for flag in ("MUNINN_CAPTURE_ENRICHMENT", "MUNINN_CAPTURE_AUTO_ANALYSIS", "MUNINN_CAPTURE_AUTO_REMOTE",
                 "MUNINN_SECURE_AUTO_ANALYSIS", "MUNINN_HISTORY_INDEX_AUTO"):
        monkeypatch.setenv(flag, "0")
    def forbidden():
        raise AssertionError("Paid recovery requires no provider key")
    monkeypatch.setattr(llm_settings, "api_key", forbidden)
    service = HistoryService(None, tmp_path / "history_vault", home=tmp_path,
        archive_passphrase="synthetic recovery passphrase")
    async def idle():
        await asyncio.Event().wait()
    for loop in ("_secure_capture_loop", "_secure_scan_loop", "_secure_search_loop"):
        monkeypatch.setattr(service, loop, idle)
    async def no_batch():
        return False
    monkeypatch.setattr(service, "_process_secure_batch_once", no_batch)
    await service.start()
    try:
        assert service._secure_analysis_task is not None
        async def published():
            while service._require_capture_journal().classification_status() != {"published": 1}:
                await asyncio.sleep(0.01)
        await asyncio.wait_for(published(), timeout=10)
        assert service.status()["memory_classification"] == {"published": 1}
    finally:
        await service.stop()
