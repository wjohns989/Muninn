"""Preserved analysis coverage, not a new inference or a duplicate citation."""
import json

import pytest

from muninn.history.capture_journal import CaptureJournal
from muninn.history.cited_windows import CitedWindowPlanStore
from muninn.history.cited_analysis_source import CitedAnalysisSource
from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.secure_analysis import _cited_model_identity
from tests.test_capture_window_jobs import window_fixture

DIGEST = "a" * 64
MODEL = "isolated-fixture-model"


def growth(tmp_path, *, empty=False, publish=True, rewrite=False):
    journal, archive, receipt = window_fixture(tmp_path)
    journal.queue_capture_windows(receipt, limit=1)
    old_job = journal.claim_analysis(include_capture=True)
    plans = CitedWindowPlanStore(archive)
    old_entry = plans.source.ledger._entries[(receipt["blob"], 0)]
    old_window = plans.window_at(old_entry, 0, old_job.target["plan_attempt"], 0)
    source = CitedAnalysisSource(archive)
    window = source.reopen(old_window)
    stage = {"format": 1, "window": old_window,
             "model_identity": _cited_model_identity(window, "ollama", MODEL, DIGEST),
             "proposals": [] if empty else [{"type": "observation", "text": "A local capture observation.",
                             "quote": "A local capture observation.", "start": 0}],
             "result": {"status": "ok", "provider": "ollama", "model": MODEL,
                        "analysis": {"summary": "Isolated plumbing fixture.", "decisions": [],
                                     "open_items": [], "uncertainty": "Not model-quality proof."}}}
    assert journal.bind_analysis_window(old_job.job_id, old_job.lease_token, old_window)
    assert journal.stage_analysis(old_job.job_id, old_job.lease_token, stage)
    refs = []
    if publish:
        assert journal.begin_publication(old_job.job_id, old_job.lease_token)
        refs = source.record_proposals(old_window, stage["proposals"], model_identity=stage["model_identity"])
        assert journal.acknowledge_publication(old_job.job_id, old_job.lease_token, refs)
    else:
        assert journal.fail_analysis(old_job.job_id, old_job.lease_token, "cancelled")
    path = tmp_path / "session.jsonl"
    addition = json.dumps({"type": "event_msg", "payload": {
        "type": "user_message", "message": "New context at the end."}}) + "\n"
    if rewrite:
        path.write_text(path.read_text(encoding="utf-8").replace("observation", "other words") + addition,
                        encoding="utf-8")
    else:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(addition)
    current = archive.archive_file(path, "codex", include_snapshot_receipt=True)["snapshot_receipt"]
    assert journal.enqueue_enrichment_receipt(current) == "queued"
    journal.queue_capture_windows(current, limit=1)
    job = journal.claim_analysis(include_capture=True)
    plans = CitedWindowPlanStore(archive)
    entry = plans.source.ledger._entries[(current["blob"], 1)]
    descriptor = plans.window_at(entry, 1, job.target["plan_attempt"], 0)
    assert journal.bind_analysis_window(job.job_id, job.lease_token, descriptor)
    return journal, archive, current, job, old_job, refs


@pytest.mark.parametrize("empty", [False, True])
def test_preserved_window_reuses_acked_analysis_without_new_ledger_events(tmp_path, empty):
    journal, archive, receipt, job, old_job, refs = growth(tmp_path, empty=empty)
    before = CitedAnalysisSource(archive).ledger.verify_all()
    assert journal.acknowledge_capture_reuse(job.job_id, job.lease_token, model=MODEL, weights_digest=DIGEST)
    visible = journal.get_analysis_job(job.job_id)
    assert visible["state"] == "reused" and visible["memory_refs"] == refs
    assert visible["result"] is None and visible["coverage_basis"] == "preserved_parent_analysis"
    status = journal.capture_window_status(receipt)
    assert status["acknowledged"] == 1 and status["jobs"] == {"reused": 1}
    assert CitedAnalysisSource(archive).ledger.verify_all() == before
    assert journal.claim_analysis(include_capture=True) is None
    assert not journal.acknowledge_capture_reuse(job.job_id, job.lease_token, model=MODEL, weights_digest=DIGEST)
    assert journal.verify_all() == 0
    restored = SecureHistoryArchive.restore_from_backup(archive.root, tmp_path / "restored",
                                                         "test-only portable passphrase")
    assert CaptureJournal(restored).get_analysis_job(job.job_id)["state"] == "reused"


@pytest.mark.parametrize("reason", ["different_weights", "different_model", "rewritten", "no_ack"])
def test_missing_occurrence_contract_or_ack_does_not_advance_coverage(tmp_path, reason):
    journal, archive, receipt, job, old_job, refs = growth(tmp_path,
        publish=reason != "no_ack", rewrite=reason == "rewritten")
    assert not journal.acknowledge_capture_reuse(job.job_id, job.lease_token,
        model=MODEL if reason != "different_model" else "another-fixture-model",
        weights_digest=DIGEST if reason != "different_weights" else "b" * 64)
    assert journal.get_analysis_job(job.job_id)["state"] == "running"
    assert journal.capture_window_status(receipt)["acknowledged"] == 0


@pytest.mark.parametrize("damage", ["reuse", "parent_ack", "parent_stage", "missing_reuse"])
def test_reuse_and_original_ack_tamper_block_status_and_portable_verification(tmp_path, damage):
    journal, archive, receipt, job, old_job, refs = growth(tmp_path)
    assert journal.acknowledge_capture_reuse(job.job_id, job.lease_token, model=MODEL, weights_digest=DIGEST)
    with journal._connect() as db:
        if damage == "missing_reuse":
            db.execute("UPDATE history_analysis_jobs SET sealed_reuse=NULL WHERE job_id=?", (job.job_id,))
        else:
            column, selected = {"reuse": ("sealed_reuse", job.job_id),
                "parent_ack": ("sealed_receipt", old_job.job_id),
                "parent_stage": ("sealed_extraction", old_job.job_id)}[damage]
            db.execute(f"UPDATE history_analysis_jobs SET {column}=zeroblob(length({column})) WHERE job_id=?",
                       (selected,))
    with pytest.raises(VaultIntegrityError):
        journal.get_analysis_job(job.job_id)
    with pytest.raises(VaultIntegrityError):
        journal.capture_window_status(receipt)
    with pytest.raises(VaultIntegrityError):
        journal.verify_all()


def test_stale_lease_and_durable_cancel_cannot_commit_reuse(tmp_path):
    journal, archive, receipt, job, old_job, refs = growth(tmp_path)
    assert not journal.acknowledge_capture_reuse(job.job_id, "stale", model=MODEL, weights_digest=DIGEST)
    assert journal.request_analysis_cancel(job.job_id)
    assert not journal.acknowledge_capture_reuse(job.job_id, job.lease_token, model=MODEL, weights_digest=DIGEST)
    assert journal.capture_window_status(receipt)["acknowledged"] == 0


def test_changed_generation_defaults_do_not_invalidate_historical_reuse(tmp_path, monkeypatch):
    from muninn.history import secure_analysis
    journal, archive, receipt, job, old_job, refs = growth(tmp_path)
    assert journal.acknowledge_capture_reuse(job.job_id, job.lease_token, model=MODEL, weights_digest=DIGEST)
    original = secure_analysis.Provider.request_body
    def changed(provider, messages):
        body = original(provider, messages)
        body["options"] = {"temperature": 0.9}
        return body
    monkeypatch.setattr(secure_analysis.Provider, "request_body", changed)
    assert journal.get_analysis_job(job.job_id)["state"] == "reused"
    assert journal.verify_all() == 0


def test_lease_is_rechecked_after_expensive_original_ledger_proof(tmp_path, monkeypatch):
    from muninn.history.memory_ledger import MemoryLedger
    journal, archive, receipt, job, old_job, refs = growth(tmp_path)
    original = MemoryLedger.verify_refs
    def expire(ledger, refs):
        checked = original(ledger, refs)
        with journal._connect() as db:
            db.execute("UPDATE history_analysis_jobs SET lease_until=0 WHERE job_id=?", (job.job_id,))
        return checked
    monkeypatch.setattr(MemoryLedger, "verify_refs", expire)
    assert not journal.acknowledge_capture_reuse(job.job_id, job.lease_token, model=MODEL, weights_digest=DIGEST)
    assert journal.capture_window_status(receipt)["acknowledged"] == 0


def test_reuse_receipt_and_source_counter_commit_atomically(tmp_path, monkeypatch):
    journal, archive, receipt, job, old_job, refs = growth(tmp_path)
    original = journal._seal_search
    def fail(value, job_id, purpose):
        if purpose.startswith("capture-reuse-v1:"):
            raise RuntimeError("Isolated interrupted reuse publication")
        return original(value, job_id, purpose)
    monkeypatch.setattr(journal, "_seal_search", fail)
    with pytest.raises(RuntimeError):
        journal.acknowledge_capture_reuse(job.job_id, job.lease_token, model=MODEL, weights_digest=DIGEST)
    assert journal.get_analysis_job(job.job_id)["state"] == "running"
    assert journal.capture_window_status(receipt)["acknowledged"] == 0


def test_changed_weights_at_final_guard_cannot_commit_reuse(tmp_path):
    journal, archive, receipt, job, old_job, refs = growth(tmp_path)
    assert not journal.acknowledge_capture_reuse(job.job_id, job.lease_token, model=MODEL,
        weights_digest=DIGEST, identity_guard=lambda: False)
    assert journal.capture_window_status(receipt)["acknowledged"] == 0


@pytest.mark.asyncio
async def test_capture_worker_reuses_actual_selected_contract_before_any_model_post(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager
    from muninn.history import secure_analysis
    from muninn.history.service import HistoryService
    journal, archive, receipt, job, old_job, refs = growth(tmp_path)
    before = CitedAnalysisSource(archive).ledger.verify_all()
    with journal._connect() as db:
        db.execute("UPDATE history_analysis_jobs SET state='pending',lease_token=NULL,lease_until=NULL "
                   "WHERE job_id=?", (job.job_id,))
    service = HistoryService(None, tmp_path / "service", home=tmp_path)
    service._capture_journal = journal
    monkeypatch.setattr(service, "_require_secure_archive", lambda: archive)
    @asynccontextmanager
    async def slot():
        yield
    monkeypatch.setattr("muninn.extraction.ollama_slot.async_ollama_slot", slot)
    monkeypatch.setattr(secure_analysis, "_select_local", lambda base: (MODEL, "local"))
    reads = []
    def digest(*args):
        reads.append(True)
        return DIGEST
    monkeypatch.setattr(secure_analysis, "_weights_digest", digest)
    monkeypatch.setattr(secure_analysis.httpx, "AsyncClient", lambda **kwargs: pytest.fail("Model POST attempted"))
    assert await service._process_secure_analysis_once(include_capture=True)
    assert journal.get_analysis_job(job.job_id)["state"] == "reused"
    assert reads == [True, True]
    assert CitedAnalysisSource(archive).ledger.verify_all() == before


@pytest.mark.asyncio
async def test_cancelled_worker_drains_reuse_writer_before_shutdown(tmp_path, monkeypatch):
    import asyncio
    import threading
    from contextlib import asynccontextmanager
    from muninn.history import secure_analysis
    from muninn.history.service import HistoryService
    journal, archive, receipt, job, old_job, refs = growth(tmp_path)
    with journal._connect() as db:
        db.execute("UPDATE history_analysis_jobs SET state='pending',lease_token=NULL,lease_until=NULL "
                   "WHERE job_id=?", (job.job_id,))
    service = HistoryService(None, tmp_path / "service", home=tmp_path)
    service._capture_journal = journal
    monkeypatch.setattr(service, "_require_secure_archive", lambda: archive)
    @asynccontextmanager
    async def slot():
        yield
    monkeypatch.setattr("muninn.extraction.ollama_slot.async_ollama_slot", slot)
    monkeypatch.setattr(secure_analysis, "_select_local", lambda base: (MODEL, "local"))
    monkeypatch.setattr(secure_analysis, "_weights_digest", lambda *args: DIGEST)
    monkeypatch.setattr(secure_analysis.httpx, "AsyncClient", lambda **kwargs: pytest.fail("Model POST attempted"))
    entered, release = threading.Event(), threading.Event()
    original = journal.acknowledge_capture_reuse
    def delayed(*args, **kwargs):
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("Isolated drain wait timed out")
        return original(*args, **kwargs)
    monkeypatch.setattr(journal, "acknowledge_capture_reuse", delayed)
    pending = asyncio.create_task(service._process_secure_analysis_once(include_capture=True))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        pending.cancel()
        await asyncio.sleep(0.02)
        assert not pending.done()  # Shutdown cannot report an undrained writer.
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, timeout=3)
    assert journal.capture_window_status(receipt)["acknowledged"] == 0
