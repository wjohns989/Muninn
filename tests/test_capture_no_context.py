"""Authenticated empty-window completion is not inference or model reuse."""
import os

import pytest

from muninn.history.capture_journal import CaptureJournal
from muninn.history.credential_crypto import VaultIntegrityError
from tests.test_capture_window_jobs import window_fixture


def queued(tmp_path, text="\n"):
    journal, archive, receipt = window_fixture(tmp_path, text=text)
    journal.queue_capture_windows(receipt, limit=4)
    job = journal.claim_analysis(include_capture=True, include_search=False)
    plans, entry = journal._capture_plan_source(receipt)
    window = plans.window_at(entry, receipt["version"], job.target["plan_attempt"], job.target["ordinal"])
    assert journal.bind_analysis_window(job.job_id, job.lease_token, window)
    return journal, archive, receipt, job


def test_empty_claim_completes_without_model_success_and_survives_reopen(tmp_path):
    journal, archive, receipt, job = queued(tmp_path)
    assert journal.acknowledge_capture_no_context(job.job_id, lease_token=job.lease_token)
    assert not journal.acknowledge_capture_no_context(job.job_id, lease_token=job.lease_token)
    state = journal.capture_window_status(receipt)
    assert state["state"] == "no_context"
    assert state["acknowledged"] == state["windows"] == 1
    assert state["jobs"] == {"no_context": 1}
    result = journal.get_analysis_job(job.job_id)
    assert result["state"] == "no_context" and result["result"] is None
    assert result["provider"] is None and result["model"] is None
    assert result["coverage_basis"] == "authenticated_whitespace"
    assert CaptureJournal(archive, recover=False).verify_all() == 0


def test_legacy_failed_empty_window_is_reconciled_without_retry(tmp_path):
    journal, archive, receipt, job = queued(tmp_path)
    assert journal.fail_analysis(job.job_id, job.lease_token, "insufficient_context")
    assert journal.reconcile_capture_no_context(limit=4) == 1
    assert journal.reconcile_capture_no_context(limit=4) == 0
    with journal._connect() as db:
        row = db.execute("SELECT attempt,state,remote_dispatched FROM history_analysis_jobs").fetchone()
        assert tuple(row) == (job.attempt, "no_context", 0)
    assert journal.pending_enrichment() == []
    assert CaptureJournal(archive, recover=False).capture_window_status(receipt)["state"] == "no_context"


@pytest.mark.parametrize("text", ["Meaningful text", "\nX\n"])
def test_nonempty_content_cannot_acquire_empty_coverage(tmp_path, text):
    journal, _, receipt, job = queued(tmp_path, text=text)
    with journal._connect() as db:
        before = tuple(db.execute("SELECT * FROM history_analysis_jobs").fetchone())
    assert not journal.acknowledge_capture_no_context(job.job_id, lease_token=job.lease_token)
    with journal._connect() as db:
        assert tuple(db.execute("SELECT * FROM history_analysis_jobs").fetchone()) == before
    assert journal.capture_window_status(receipt)["acknowledged"] == 0


@pytest.mark.parametrize("field,value", [("remote_dispatched", 1), ("publication_started", 1),
    ("sealed_extraction", b"retain"), ("sealed_receipt", b"retain"),
    ("sealed_reuse", b"retain"), ("sealed_result", b"retain"),
    ("cancel_requested", 1), ("state", "outcome_unknown")])
def test_sent_immutable_or_cancelled_work_is_unchanged(tmp_path, field, value):
    journal, _, _, job = queued(tmp_path)
    with journal._connect() as db:
        db.execute(f"UPDATE history_analysis_jobs SET {field}=?", (value,))
        before = tuple(db.execute("SELECT * FROM history_analysis_jobs").fetchone())
    assert not journal.acknowledge_capture_no_context(job.job_id, lease_token=job.lease_token)
    with journal._connect() as db:
        assert tuple(db.execute("SELECT * FROM history_analysis_jobs").fetchone()) == before


def test_bad_empty_proof_blocks_verification_and_status(tmp_path):
    journal, _, receipt, job = queued(tmp_path)
    assert journal.acknowledge_capture_no_context(job.job_id, lease_token=job.lease_token)
    with journal._connect() as db:
        db.execute("UPDATE history_analysis_jobs SET sealed_result=zeroblob(40)")
    with pytest.raises(VaultIntegrityError):
        journal.capture_window_status(receipt)
    with pytest.raises(VaultIntegrityError):
        journal.verify_all()


@pytest.mark.parametrize("limit", [0, 129, True, 1.5])
def test_reconciliation_is_bounded(tmp_path, limit):
    journal, _, _, _ = queued(tmp_path)
    with pytest.raises(ValueError):
        journal.reconcile_capture_no_context(limit=limit)


def test_paid_item_cannot_be_resolved_locally(tmp_path, monkeypatch):
    journal, _, _, job = queued(tmp_path)
    monkeypatch.setattr(journal, "_historical_batch_blocked_jobs", lambda db: {job.job_id})
    with journal._connect() as db:
        before = tuple(db.execute("SELECT * FROM history_analysis_jobs").fetchone())
    assert not journal.acknowledge_capture_no_context(job.job_id, lease_token=job.lease_token)
    with journal._connect() as db:
        assert tuple(db.execute("SELECT * FROM history_analysis_jobs").fetchone()) == before


def test_failed_reconciliation_fences_stale_attempt_and_target(tmp_path):
    journal, _, _, job = queued(tmp_path)
    assert journal.fail_analysis(job.job_id, job.lease_token, "insufficient_context")
    assert not journal.acknowledge_capture_no_context(job.job_id, expected_attempt=job.attempt - 1,
                                                    expected_target_sha256="a" * 64)
    assert not journal.acknowledge_capture_no_context(job.job_id, expected_attempt=job.attempt,
                                                    expected_target_sha256="a" * 64)
    assert journal.reconcile_capture_no_context() == 1


def test_proof_and_counters_roll_back_together(tmp_path, monkeypatch):
    journal, _, _, job = queued(tmp_path)
    with journal._connect() as db:
        before = tuple(db.execute("SELECT * FROM history_analysis_jobs").fetchone())
        source_before = tuple(db.execute("SELECT * FROM capture_enrichment_sources").fetchone())
    ack = journal._ack_capture_window
    def interrupted(db, row):
        ack(db, row)
        raise RuntimeError("isolated rollback")
    monkeypatch.setattr(journal, "_ack_capture_window", interrupted)
    with pytest.raises(RuntimeError):
        journal.acknowledge_capture_no_context(job.job_id, lease_token=job.lease_token)
    with journal._connect() as db:
        assert tuple(db.execute("SELECT * FROM history_analysis_jobs").fetchone()) == before
        assert tuple(db.execute("SELECT * FROM capture_enrichment_sources").fetchone()) == source_before


@pytest.mark.asyncio
async def test_new_whitespace_never_calls_analysis(monkeypatch, tmp_path):
    from tests.test_capture_automatic_service import enabled_service
    from muninn.history import secure_analysis
    service, _, source, now = enabled_service(monkeypatch, tmp_path)
    source.write_text('{"type":"event_msg","payload":{"type":"user_message","message":"\\n"}}\n')
    await service.capture(str(source), "codex")
    journal = service._require_capture_journal()
    receipt = journal.pending_enrichment()[0]
    journal.queue_capture_windows(receipt, limit=4)
    async def forbidden(*args, **kwargs):
        pytest.fail("whitespace must not invoke model routing or inference")
    monkeypatch.setattr(secure_analysis, "analyze_cited_window", forbidden)
    now[0] = 300.0
    assert await service._process_secure_analysis_once(include_capture=True, include_search=False)
    assert journal.capture_window_status(receipt)["state"] == "no_context"


def test_cleanup_gate_ignores_capacity_but_preserves_foreground_priority(tmp_path, monkeypatch):
    journal, _, _, _ = queued(tmp_path)
    monkeypatch.setattr(journal, "_capture_window_capacity", lambda db: 0)
    assert not journal.capture_planning_ready()
    assert journal.capture_planning_ready(require_capacity=False)
    journal.enqueue_search("isolated query")
    assert not journal.capture_planning_ready(require_capacity=False)


def test_damaged_original_plan_cannot_commit_empty_completion(tmp_path):
    journal, _, receipt, job = queued(tmp_path)
    plans, _ = journal._capture_plan_source(receipt)
    with plans._connect() as db:
        db.execute("UPDATE pages SET ciphertext=zeroblob(40)")
    with journal._connect() as db:
        before = tuple(db.execute("SELECT * FROM history_analysis_jobs").fetchone())
        source_before = tuple(db.execute("SELECT * FROM capture_enrichment_sources").fetchone())
    with pytest.raises(VaultIntegrityError):
        journal.acknowledge_capture_no_context(job.job_id, lease_token=job.lease_token)
    with journal._connect() as db:
        assert tuple(db.execute("SELECT * FROM history_analysis_jobs").fetchone()) == before
        assert tuple(db.execute("SELECT * FROM capture_enrichment_sources").fetchone()) == source_before


def test_empty_proof_survives_portable_archive_restore(tmp_path):
    from muninn.history.secure_archive import SecureHistoryArchive
    journal, archive, receipt, job = queued(tmp_path)
    assert journal.acknowledge_capture_no_context(job.job_id, lease_token=job.lease_token)
    # Portable passphrase restoration is supported on every host. The source
    # here is an isolated encrypted archive, not an unattended Windows backup.
    restored = SecureHistoryArchive.restore_from_backup(archive.root, tmp_path / "restored",
                                                       "test-only portable passphrase")
    restored_journal = CaptureJournal(restored, recover=False)
    assert restored_journal.verify_all() == 0
    assert restored_journal.get_analysis_job(job.job_id)["coverage_basis"] == "authenticated_whitespace"
    assert restored_journal.capture_window_status(receipt)["state"] == "no_context"


@pytest.mark.skipif(os.name != "nt", reason="Unattended backup requires Windows user protection")
def test_empty_proof_survives_actual_windows_backup_restore(tmp_path):
    from muninn.history.secure_archive import SecureHistoryArchive
    journal, archive, receipt, job = queued(tmp_path)
    assert journal.acknowledge_capture_no_context(job.job_id, lease_token=job.lease_token)
    archive.backup_to(tmp_path / "backup")
    restored = SecureHistoryArchive.restore_from_backup(tmp_path / "backup", tmp_path / "restored",
                                                       "test-only portable passphrase")
    restored_journal = CaptureJournal(restored, recover=False)
    assert restored_journal.verify_all() == 0
    assert restored_journal.get_analysis_job(job.job_id)["coverage_basis"] == "authenticated_whitespace"
    assert restored_journal.capture_window_status(receipt)["state"] == "no_context"


@pytest.mark.parametrize("field,value", [("lease_until", 0), ("lease_token", "a" * 32)])
def test_expired_or_replaced_lease_cannot_commit(tmp_path, field, value):
    journal, _, _, job = queued(tmp_path)
    with journal._connect() as db:
        db.execute(f"UPDATE history_analysis_jobs SET {field}=?", (value,))
    assert not journal.acknowledge_capture_no_context(job.job_id, lease_token=job.lease_token)


@pytest.mark.asyncio
async def test_service_repairs_old_empty_failures_at_full_capacity(monkeypatch, tmp_path):
    from tests.test_capture_automatic_service import enabled_service
    service, _, source, now = enabled_service(monkeypatch, tmp_path)
    source.write_text('{"type":"event_msg","payload":{"type":"user_message","message":"\\n"}}\n')
    await service.capture(str(source), "codex")
    journal = service._require_capture_journal()
    receipt = journal.pending_enrichment()[0]
    journal.queue_capture_windows(receipt, limit=4)
    job = journal.claim_analysis(include_capture=True, include_search=False)
    plans, entry = journal._capture_plan_source(receipt)
    window = plans.window_at(entry, receipt["version"], job.target["plan_attempt"], 0)
    assert journal.bind_analysis_window(job.job_id, job.lease_token, window)
    assert journal.fail_analysis(job.job_id, job.lease_token, "insufficient_context")
    monkeypatch.setattr(journal, "_capture_window_capacity", lambda db: 0)
    monkeypatch.setattr(journal, "next_capture_plan", lambda: pytest.fail("full queue must not plan"))
    now[0] = 300.0
    search = journal.enqueue_search("isolated query")
    assert not await service._process_capture_plan_once(automatic=True)
    assert journal.get_analysis_job(job.job_id)["state"] == "failed"
    assert journal.cancel_search(search)
    assert await service._process_capture_plan_once(automatic=True)
    assert journal.capture_window_status(receipt)["state"] == "no_context"


@pytest.mark.asyncio
async def test_cancelled_consumer_drains_empty_ack_writer(monkeypatch, tmp_path):
    import asyncio
    import threading
    from tests.test_capture_automatic_service import enabled_service
    service, _, source, now = enabled_service(monkeypatch, tmp_path)
    source.write_text('{"type":"event_msg","payload":{"type":"user_message","message":"\\n"}}\n')
    await service.capture(str(source), "codex")
    journal = service._require_capture_journal()
    receipt = journal.pending_enrichment()[0]
    journal.queue_capture_windows(receipt, limit=4)
    ack = journal.acknowledge_capture_no_context
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    def controlled(*args, **kwargs):
        started.set()
        assert release.wait(5)
        try:
            return ack(*args, **kwargs)
        finally:
            finished.set()
    monkeypatch.setattr(journal, "acknowledge_capture_no_context", controlled)
    now[0] = 300.0
    consumer = asyncio.create_task(service._process_secure_analysis_once(
        include_capture=True, include_search=False))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        consumer.cancel()
        await asyncio.sleep(0.02)
        assert not consumer.done() and not finished.is_set()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    assert finished.is_set()
    assert journal.capture_window_status(receipt)["state"] == "no_context"
