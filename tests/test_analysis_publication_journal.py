"""Real encrypted isolated stores; no inference, remote provider, or live state."""
import json
import sqlite3

import pytest

from muninn.history.blind_index import SecureHistoryBlindIndex
from muninn.history.capture_journal import CaptureJournal, SearchJobError
from muninn.history.cited_analysis_source import CitedAnalysisSource
from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.secure_archive import SecureHistoryArchive


def queued(tmp_path):
    archive = SecureHistoryArchive.create(tmp_path / "archive", "synthetic recovery passphrase")
    path = tmp_path / "chat.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [
        {"type": "session_meta", "payload": {"cwd": "C:/synthetic-project"}},
        {"type": "event_msg", "timestamp": "2026-09-30T12:00:00Z", "payload": {
            "type": "user_message", "message": "Keep needle citations."}},
    ]) + "\n", encoding="utf-8")
    archive.archive_file(path, "codex")
    entry = archive._load_manifest()["files"][str(path.resolve())][0]
    index = SecureHistoryBlindIndex(archive)
    cap = index._capability(entry, 0, "needle")
    source = CitedAnalysisSource(archive)
    window = source.prepare(cap)
    journal = CaptureJournal(archive)
    search = journal.enqueue_search("needle")
    worker = journal.claim_search()
    result = {"matches": [{"ref": "r", "provider": "codex", "kind": "transcript",
              "captured_day_utc": "2026-09-30", "size_bucket_kib": 1, "versions": 1,
              "fetch_capability": cap}], "total": 1, "ready": 1, "missing": 0,
              "overflow": 0, "complete": True, "truncated": False}
    journal.finish_search(search, worker.lease_token, result,
                           analysis_target=index._analysis_target(cap, ["needle"]))
    job = journal.claim_analysis()
    stage = {"format": 1, "window": window, "model_identity": "a" * 64,
             "proposals": [{"type": "decision", "text": "Keep citations.",
                            "quote": "Keep needle citations.", "start": 0}],
             "result": {"status": "ok", "provider": "ollama", "model": "fixture-model",
                        "analysis": {"summary": "Citations were requested.", "decisions": [],
                                     "open_items": [], "uncertainty": "Unverified."}}}
    return journal, archive, job, stage, source


def bind_stage(journal, job, stage):
    assert journal.bind_analysis_window(job.job_id, job.lease_token, stage["window"])
    assert journal.stage_analysis(job.job_id, job.lease_token, stage)


def expire(journal, job):
    with sqlite3.connect(journal.path) as db:
        db.execute("UPDATE history_analysis_jobs SET lease_until=0,due_at=0 WHERE job_id=?", (job.job_id,))


def publish(source, stage):
    return source.record_proposals(stage["window"], stage["proposals"], model_identity=stage["model_identity"])


def test_stage_is_immutable_encrypted_and_not_public(tmp_path):
    journal, archive, job, stage, source = queued(tmp_path)
    bind_stage(journal, job, stage)
    assert journal.stage_analysis(job.job_id, job.lease_token, stage)
    changed = {**stage, "result": {**stage["result"], "model": "different-model"}}
    with pytest.raises(SearchJobError):
        journal.stage_analysis(job.job_id, job.lease_token, changed)
    public = json.dumps(journal.get_analysis_job(job.job_id))
    assert "Keep needle citations" not in public and "window" not in public and "proposals" not in public
    assert b"Keep needle citations" not in journal.path.read_bytes()
    reopened = journal._analysis_row(journal._publication_row(job.job_id))
    assert reopened.extraction == stage and reopened.window == stage["window"]
    assert "Keep needle citations" not in repr(reopened)


def test_cancel_and_publication_admission_are_mutually_exclusive(tmp_path):
    journal, archive, job, stage, source = queued(tmp_path)
    bind_stage(journal, job, stage)
    assert journal.request_analysis_cancel(job.job_id)
    assert not journal.begin_publication(job.job_id, job.lease_token)
    assert source.ledger.verify_all()["candidates"] == 0
    assert journal.fail_analysis(job.job_id, job.lease_token, "cancelled")
    assert journal.get_analysis_job(job.job_id) is None


def test_admitted_publication_cannot_be_cancelled_and_can_acknowledge(tmp_path):
    journal, archive, job, stage, source = queued(tmp_path)
    bind_stage(journal, job, stage)
    assert journal.begin_publication(job.job_id, job.lease_token)
    assert not journal.request_analysis_cancel(job.job_id)
    assert not journal.cancel_analysis(job.job_id)
    # Ledger/source work is outside a journal transaction: a heartbeat and an
    # unrelated enqueue remain possible, even during publication.
    assert journal.heartbeat_analysis(job.job_id, job.lease_token)
    assert journal.enqueue_search("another")
    refs = publish(source, stage)
    assert journal.acknowledge_publication(job.job_id, job.lease_token, refs)
    visible = journal.get_analysis_job(job.job_id)
    assert visible["state"] == "succeeded" and visible["result"] == stage["result"]
    assert visible["memory_refs"] == refs


@pytest.mark.parametrize("remote", [False, True])
def test_crash_after_ledger_commit_replays_stage_not_inference(tmp_path, remote):
    journal, archive, job, stage, source = queued(tmp_path)
    bind_stage(journal, job, stage)
    if remote:
        assert journal.mark_remote_dispatched(job.job_id, job.lease_token)
    assert journal.begin_publication(job.job_id, job.lease_token)
    refs = publish(source, stage)
    expire(journal, job)
    reopened = CaptureJournal(archive)
    resumed = reopened.claim_analysis()
    assert resumed.state == "publishing" and resumed.extraction == stage
    assert resumed.lease_token != job.lease_token
    assert not reopened.acknowledge_publication(job.job_id, job.lease_token, refs)
    assert not reopened.request_analysis_cancel(job.job_id)
    assert publish(CitedAnalysisSource(archive), resumed.extraction) == refs
    assert reopened.acknowledge_publication(job.job_id, resumed.lease_token, refs)
    assert source.ledger.verify_all()["candidates"] == 1


def test_durable_reply_before_admission_replays_without_remote_redispatch(tmp_path):
    journal, archive, job, stage, source = queued(tmp_path)
    bind_stage(journal, job, stage)
    assert journal.mark_remote_dispatched(job.job_id, job.lease_token)
    expire(journal, job)
    resumed = CaptureJournal(archive).claim_analysis()
    assert resumed.extraction == stage and resumed.state == "running"
    assert not journal.mark_remote_dispatched(job.job_id, resumed.lease_token)
    assert journal.begin_publication(job.job_id, resumed.lease_token)


@pytest.mark.parametrize("mutation", ["window", "stage", "identity"])
def test_tampered_stage_or_window_cannot_be_resumed(tmp_path, mutation):
    journal, archive, job, stage, source = queued(tmp_path)
    bind_stage(journal, job, stage)
    with sqlite3.connect(journal.path) as db:
        column = {"window": "sealed_window", "stage": "sealed_extraction"}.get(mutation)
        if column:
            db.execute(f"UPDATE history_analysis_jobs SET {column}=zeroblob(length({column})) WHERE job_id=?", (job.job_id,))
        else:
            db.execute("UPDATE history_analysis_jobs SET extraction_id=? WHERE job_id=?", ("b" * 64, job.job_id))
    with pytest.raises(VaultIntegrityError):
        journal.get_analysis_job(job.job_id)
    assert source.ledger.verify_all()["candidates"] == 0


def test_invalid_quote_or_stale_lease_cannot_stage_or_publish(tmp_path):
    journal, archive, job, stage, source = queued(tmp_path)
    assert not journal.bind_analysis_window(job.job_id, "wrong", stage["window"])
    assert journal.bind_analysis_window(job.job_id, job.lease_token, stage["window"])
    bad = {**stage, "proposals": [{**stage["proposals"][0], "quote": "invented source quote"}]}
    with pytest.raises(ValueError):
        journal.stage_analysis(job.job_id, job.lease_token, bad)
    assert journal._publication_row(job.job_id)["sealed_extraction"] is None
    assert not journal.stage_analysis(job.job_id, "wrong", stage)
    assert not journal.begin_publication(job.job_id, "wrong")
    assert source.ledger.verify_all()["candidates"] == 0


@pytest.mark.parametrize("remote", [False, True])
def test_durable_cancel_survives_worker_crash_before_publication(tmp_path, remote):
    journal, archive, job, stage, source = queued(tmp_path)
    bind_stage(journal, job, stage)
    if remote:
        assert journal.mark_remote_dispatched(job.job_id, job.lease_token)
    assert journal.request_analysis_cancel(job.job_id)
    expire(journal, job)
    reopened = CaptureJournal(archive)
    assert reopened.claim_analysis() is None
    assert reopened.get_analysis_job(job.job_id) is None
    assert not reopened.begin_publication(job.job_id, job.lease_token)
    assert source.ledger.verify_all()["candidates"] == 0


def test_publication_only_retry_preserves_cutoff_and_stage(tmp_path):
    journal, archive, job, stage, source = queued(tmp_path)
    bind_stage(journal, job, stage)
    assert journal.begin_publication(job.job_id, job.lease_token)
    assert journal.defer_publication(job.job_id, job.lease_token)
    assert not journal.request_analysis_cancel(job.job_id)
    assert journal.get_analysis_job(job.job_id)["state"] == "publication_pending"
    with sqlite3.connect(journal.path) as db:
        db.execute("UPDATE history_analysis_jobs SET due_at=0 WHERE job_id=?", (job.job_id,))
    resumed = journal.claim_analysis()
    assert resumed.state == "publishing" and resumed.extraction == stage
    assert journal.acknowledge_publication(job.job_id, resumed.lease_token, publish(source, stage))


def test_cited_queue_state_and_receipt_survive_portable_recovery(tmp_path):
    journal, archive, job, stage, source = queued(tmp_path)
    bind_stage(journal, job, stage)
    assert journal.begin_publication(job.job_id, job.lease_token)
    refs = publish(source, stage)
    assert journal.acknowledge_publication(job.job_id, job.lease_token, refs)
    restored = SecureHistoryArchive.restore_from_backup(archive.root, tmp_path / "restored",
                                                         "synthetic recovery passphrase")
    visible = CaptureJournal(restored).get_analysis_job(job.job_id)
    assert visible["memory_refs"] == refs and visible["result"] == stage["result"]
    assert CitedAnalysisSource(restored).ledger.get(refs[0])["proposal_origin"] == "model"


def test_window_and_stage_binding_prevent_repointing_to_another_input(tmp_path):
    journal, archive, job, stage, source = queued(tmp_path)
    bind_stage(journal, job, stage)
    changed = {**stage["window"], "input_sha256": "b" * 64}
    with pytest.raises(ValueError):
        journal.bind_analysis_window(job.job_id, job.lease_token, changed)
    with pytest.raises(ValueError):
        journal.stage_analysis(job.job_id, job.lease_token, {**stage, "window": changed})
    assert journal._analysis_row(journal._publication_row(job.job_id)).extraction == stage


def test_acknowledgment_requires_the_exact_durable_memory_records(tmp_path):
    journal, archive, job, stage, source = queued(tmp_path)
    bind_stage(journal, job, stage)
    assert journal.begin_publication(job.job_id, job.lease_token)
    with pytest.raises(SearchJobError):
        journal.acknowledge_publication(job.job_id, job.lease_token, ["b" * 64])
    assert journal.get_analysis_job(job.job_id)["state"] == "publishing"
    assert source.ledger.verify_all()["candidates"] == 0


def test_computed_identity_without_ledger_commit_is_not_success(tmp_path):
    journal, archive, job, stage, source = queued(tmp_path)
    bind_stage(journal, job, stage)
    assert journal.begin_publication(job.job_id, job.lease_token)
    refs = source.expected_refs(stage["window"], stage["proposals"], model_identity=stage["model_identity"])
    with pytest.raises(SearchJobError):
        journal.acknowledge_publication(job.job_id, job.lease_token, refs)
    assert journal.get_analysis_job(job.job_id)["state"] == "publishing"


def test_ledger_corruption_blocks_acknowledgment(tmp_path):
    journal, archive, job, stage, source = queued(tmp_path)
    bind_stage(journal, job, stage)
    assert journal.begin_publication(job.job_id, job.lease_token)
    refs = publish(source, stage)
    with source.ledger._connect() as db:
        db.execute("UPDATE events SET ciphertext=zeroblob(length(ciphertext))")
    with pytest.raises(RuntimeError):
        journal.acknowledge_publication(job.job_id, job.lease_token, refs)
    assert journal.get_analysis_job(job.job_id)["state"] == "publishing"


def test_lease_is_rechecked_after_ledger_verification(tmp_path, monkeypatch):
    from muninn.history.memory_ledger import MemoryLedger
    journal, archive, job, stage, source = queued(tmp_path)
    bind_stage(journal, job, stage)
    assert journal.begin_publication(job.job_id, job.lease_token)
    refs = publish(source, stage)
    original = MemoryLedger.verify_refs
    def expire_after_proof(ledger, refs):
        result = original(ledger, refs)
        # This separate writer would fail if ACK held the journal transaction.
        expire(journal, job)
        return result
    monkeypatch.setattr(MemoryLedger, "verify_refs", expire_after_proof)
    assert not journal.acknowledge_publication(job.job_id, job.lease_token, refs)
    assert journal.get_analysis_job(job.job_id)["state"] == "publishing"


def test_acknowledgment_does_not_hold_writer_lock_during_verification(tmp_path, monkeypatch):
    from muninn.history.memory_ledger import MemoryLedger
    journal, archive, job, stage, source = queued(tmp_path)
    bind_stage(journal, job, stage)
    assert journal.begin_publication(job.job_id, job.lease_token)
    refs = publish(source, stage)
    original = MemoryLedger.verify_refs
    def unrelated_write(ledger, refs):
        assert journal.heartbeat_analysis(job.job_id, job.lease_token)
        assert journal.enqueue_search("independent")
        return original(ledger, refs)
    monkeypatch.setattr(MemoryLedger, "verify_refs", unrelated_write)
    assert journal.acknowledge_publication(job.job_id, job.lease_token, refs)
