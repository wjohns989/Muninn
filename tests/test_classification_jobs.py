"""Durable encrypted ACK/stage recovery; no live models or credentials."""
import time

import pytest

from muninn.history.capture_journal import CaptureJournal
from muninn.history.memory_classification import prepare_classification
from muninn.history.memory_ledger import MemoryLedger
from tests.test_analysis_publication_journal import queued, bind_stage, publish
from tests.test_memory_classification import reply


@pytest.fixture(autouse=True)
def isolated_dispatch_admission(monkeypatch):
    # Tests below isolate encrypted job ownership, not real provider billing.
    monkeypatch.setattr("muninn.history.remote_accounting.unowned_unknown_response", lambda *a, **k: True)
    monkeypatch.setattr("muninn.history.remote_accounting.classification_admission_state", lambda *a, **k: "unknown")


def acknowledged(tmp_path):
    journal, archive, analysis, stage, source = queued(tmp_path)
    bind_stage(journal, analysis, stage)
    assert journal.begin_publication(analysis.job_id, analysis.lease_token)
    refs = publish(source, stage)
    assert journal.acknowledge_publication(analysis.job_id, analysis.lease_token, refs)
    return journal, archive, refs


def test_ack_discovery_is_encrypted_idempotent_and_survives_restart(tmp_path):
    journal, archive, refs = acknowledged(tmp_path)
    assert journal.discover_classifications() == {"acks": 1, "jobs": 1}
    reopened = CaptureJournal(archive, recover=False)
    assert reopened.discover_classifications() == {"acks": 0, "jobs": 0}
    job = reopened.claim_classification()
    assert job["refs"] == refs and job["state"] == "running"
    assert b"Keep needle" not in journal.path.read_bytes()
    assert reopened.verify_classifications() == 1


def test_sent_claim_expires_unknown_without_redispatch(tmp_path):
    journal, archive, refs = acknowledged(tmp_path)
    journal.discover_classifications()
    job = journal.claim_classification()
    plan = prepare_classification(MemoryLedger(archive, read_only=True), refs)
    journal.prepare_classification_job(job["job_id"], job["lease"], plan)
    journal.mark_classification_dispatch(job["job_id"], job["lease"], "c" * 32, 1)
    assert journal.claim_classification(now=time.time() + 200) is None
    assert journal.classification_status() == {"outcome_unknown": 1}
    assert CaptureJournal(archive, recover=False).claim_classification() is None


def test_stage_requires_verified_settlement_and_crash_recovers_without_model(tmp_path, monkeypatch):
    journal, archive, refs = acknowledged(tmp_path)
    journal.discover_classifications()
    job = journal.claim_classification()
    plan = prepare_classification(MemoryLedger(archive, read_only=True), refs)
    journal.prepare_classification_job(job["job_id"], job["lease"], plan)
    journal.mark_classification_dispatch(job["job_id"], job["lease"], "c" * 32, 1)
    monkeypatch.setattr("muninn.history.remote_accounting.settled_response", lambda *a, **k: False)
    with pytest.raises(ValueError, match="settlement missing"):
        journal.stage_classification(job["job_id"], job["lease"], reply(plan), model="synthetic-luna")
    # Synthetic settlement stub isolates stage/replay properties, not billing.
    monkeypatch.setattr("muninn.history.remote_accounting.settled_response", lambda *a, **k: True)
    journal.stage_classification(job["job_id"], job["lease"], reply(plan), model="synthetic-luna")
    reopened = CaptureJournal(archive, recover=False)
    recovered = reopened.claim_classification()
    assert recovered["state"] == "staged" and recovered["attempt"] == 1
    stage = recovered["stage"]
    # Crash after ledger append but before journal publication ACK.
    MemoryLedger(archive).commit_classification(plan, stage["raw"],
        model_identity=stage["model_identity"], receipt=stage["receipt"])
    MemoryLedger(archive).resolve_review(refs[0], state="rejected", expected_state="provisional", reason="user_rejected")
    assert reopened.publish_classification(job["job_id"]) == refs
    assert reopened.classification_status() == {"published": 1}
    assert reopened.claim_classification() is None
    assert reopened.verify_classifications() == 1
    assert MemoryLedger(archive).get(refs[0])["state"] == "rejected"
    assert MemoryLedger(archive).get(refs[0])["placement"]["status"] == "stale"


def test_prepare_rejects_wrong_candidate_even_with_valid_input_identity(tmp_path):
    from tests.test_memory_classification import peers
    journal, archive, refs = acknowledged(tmp_path)
    journal.discover_classifications()
    job = journal.claim_classification()
    separate = tmp_path / "separate"
    separate.mkdir()
    _writer, reader, other = peers(separate)
    with pytest.raises(ValueError, match="preparation lease changed"):
        journal.prepare_classification_job(job["job_id"], job["lease"], prepare_classification(reader, other[:1]))


def test_already_settled_or_batch_owned_admission_cannot_be_substituted(tmp_path, monkeypatch):
    journal, archive, refs = acknowledged(tmp_path)
    journal.discover_classifications()
    job = journal.claim_classification()
    journal.prepare_classification_job(job["job_id"], job["lease"],
        prepare_classification(MemoryLedger(archive, read_only=True), refs))
    monkeypatch.setattr("muninn.history.remote_accounting.unowned_unknown_response", lambda *a, **k: False)
    with pytest.raises(ValueError, match="fresh unowned dispatch"):
        journal.mark_classification_dispatch(job["job_id"], job["lease"], "c" * 32, 1)
    assert journal.classification_status() == {"running": 1}


def test_charged_invalid_reply_never_becomes_dispatch_retry(tmp_path, monkeypatch):
    journal, archive, refs = acknowledged(tmp_path)
    journal.discover_classifications()
    job = journal.claim_classification()
    plan = prepare_classification(MemoryLedger(archive, read_only=True), refs)
    journal.prepare_classification_job(job["job_id"], job["lease"], plan)
    journal.mark_classification_dispatch(job["job_id"], job["lease"], "c" * 32, 1)
    monkeypatch.setattr("muninn.history.remote_accounting.settled_response", lambda *a, **k: True)
    with pytest.raises(ValueError, match="reply_invalid"):
        journal.stage_classification(job["job_id"], job["lease"], "malformed", model="synthetic-luna")
    journal.stop_classification(job["job_id"], job["lease"], reason="reply_invalid")
    assert journal.classification_status() == {"needs_user": 1} and journal.claim_classification() is None


def test_stale_stage_is_consultation_and_retained_without_redispatch(tmp_path, monkeypatch):
    journal, archive, refs = acknowledged(tmp_path)
    journal.discover_classifications()
    job = journal.claim_classification()
    plan = prepare_classification(MemoryLedger(archive, read_only=True), refs)
    journal.prepare_classification_job(job["job_id"], job["lease"], plan)
    journal.mark_classification_dispatch(job["job_id"], job["lease"], "c" * 32, 1)
    monkeypatch.setattr("muninn.history.remote_accounting.settled_response", lambda *a, **k: True)
    stage_id = journal.stage_classification(job["job_id"], job["lease"], reply(plan), model="synthetic-luna")
    MemoryLedger(archive).resolve_review(refs[0], state="filed", expected_state="provisional", reason="user_confirmed")
    with pytest.raises(ValueError, match="human_or_terminal"):
        journal.publish_classification(job["job_id"])
    journal.stop_classification(job["job_id"], None, reason="input_changed")
    assert journal.classification_status() == {"needs_user": 1} and journal.claim_classification() is None
    with journal._connect() as db:
        value = journal._classification_job(db.execute("SELECT * FROM memory_classification_jobs").fetchone())
    assert value["stage"]["receipt"]["stage_id"] == stage_id
    assert journal.verify_classifications() == 1


def test_late_old_job_ack_and_duplicate_results_do_not_lose_or_multiply_work(tmp_path):
    from muninn.history.blind_index import SecureHistoryBlindIndex
    journal, archive, old, stage, source = queued(tmp_path)
    with journal._connect() as db:
        original = db.execute("SELECT * FROM history_search_jobs").fetchone()
        result = journal._open_search(original["sealed_result"], original["job_id"], "result")
    new_search = journal.enqueue_search("needle citations")
    search_worker = journal.claim_search()
    cap = result["matches"][0]["fetch_capability"]
    journal.finish_search(new_search, search_worker.lease_token, result,
        analysis_target=SecureHistoryBlindIndex(archive)._analysis_target(cap, ["needle", "citations"]))
    journal.defer_analysis(old.job_id, old.lease_token, "deferred")
    newer = journal.claim_analysis()
    assert newer.job_id != old.job_id
    bind_stage(journal, newer, stage)
    journal.begin_publication(newer.job_id, newer.lease_token)
    refs = publish(source, stage)
    journal.acknowledge_publication(newer.job_id, newer.lease_token, refs)
    assert journal.discover_classifications() == {"acks": 1, "jobs": 1}
    with journal._connect() as db:
        db.execute("UPDATE history_analysis_jobs SET due_at=0 WHERE job_id=?", (old.job_id,))
    late = journal.claim_analysis()
    assert late.job_id == old.job_id
    old = late
    bind_stage(journal, old, stage)
    journal.begin_publication(old.job_id, old.lease_token)
    assert publish(source, stage) == refs
    journal.acknowledge_publication(old.job_id, old.lease_token, refs)
    assert journal.discover_classifications() == {"acks": 1, "jobs": 0}
    assert journal.classification_status() == {"pending": 1}
    assert journal.verify_classifications() == 1


def test_portable_recovery_preserves_staged_classification_without_dispatch(tmp_path, monkeypatch):
    from muninn.history.secure_archive import SecureHistoryArchive
    journal, archive, refs = acknowledged(tmp_path)
    journal.discover_classifications()
    job = journal.claim_classification()
    plan = prepare_classification(MemoryLedger(archive, read_only=True), refs)
    journal.prepare_classification_job(job["job_id"], job["lease"], plan)
    journal.mark_classification_dispatch(job["job_id"], job["lease"], "c" * 32, 1)
    monkeypatch.setattr("muninn.history.remote_accounting.settled_response", lambda *a, **k: True)
    journal.stage_classification(job["job_id"], job["lease"], reply(plan), model="synthetic-luna")
    archive.backup_to(tmp_path / "backup")
    restored = SecureHistoryArchive.restore_from_backup(tmp_path / "backup", tmp_path / "restore", "synthetic recovery passphrase")
    reopened = CaptureJournal(restored, recover=False)
    assert reopened.classification_status() == {"staged": 1}
    assert reopened.claim_classification()["attempt"] == 1
    assert reopened.publish_classification(job["job_id"]) == refs
    assert reopened.verify_classifications() == 1


def test_expired_stage_worker_is_fenced_without_recovery_claim(tmp_path, monkeypatch):
    journal, archive, refs = acknowledged(tmp_path)
    journal.discover_classifications()
    job = journal.claim_classification()
    plan = prepare_classification(MemoryLedger(archive, read_only=True), refs)
    journal.prepare_classification_job(job["job_id"], job["lease"], plan)
    journal.mark_classification_dispatch(job["job_id"], job["lease"], "c" * 32, 1)
    with journal._connect() as db:
        value = journal._classification_job(db.execute("SELECT * FROM memory_classification_jobs").fetchone())
        value["lease_until"] = time.time() - 1
        journal._save_classification(db, job["job_id"], value)
    monkeypatch.setattr("muninn.history.remote_accounting.settled_response", lambda *a, **k: True)
    with pytest.raises(ValueError, match="stage lease changed"):
        journal.stage_classification(job["job_id"], job["lease"], reply(plan), model="synthetic-luna")


def test_backup_rejects_orphan_ledger_event_but_allows_commit_before_ack(tmp_path, monkeypatch):
    from muninn.history.credential_crypto import VaultIntegrityError
    journal, archive, refs = acknowledged(tmp_path)
    journal.discover_classifications()
    job = journal.claim_classification()
    plan = prepare_classification(MemoryLedger(archive, read_only=True), refs)
    journal.prepare_classification_job(job["job_id"], job["lease"], plan)
    journal.mark_classification_dispatch(job["job_id"], job["lease"], "c" * 32, 1)
    monkeypatch.setattr("muninn.history.remote_accounting.settled_response", lambda *a, **k: True)
    journal.stage_classification(job["job_id"], job["lease"], reply(plan), model="synthetic-luna")
    with journal._connect() as db:
        value = journal._classification_job(db.execute("SELECT * FROM memory_classification_jobs").fetchone())
    MemoryLedger(archive).commit_classification(plan, value["stage"]["raw"],
        model_identity=value["stage"]["model_identity"], receipt=value["stage"]["receipt"])
    assert journal.verify_classifications() == 1
    with journal._connect() as db:
        db.execute("DELETE FROM memory_classification_jobs")  # isolated corrupt-copy counterexample
    with pytest.raises(VaultIntegrityError, match="membership"):
        journal.verify_classifications()


def test_real_accounting_binding_survives_encrypted_portable_recovery(tmp_path, monkeypatch):
    from muninn.history.remote_accounting import reserve, settled_response
    from muninn.history.secure_archive import SecureHistoryArchive
    from tests.test_remote_accounting import policy, READY
    monkeypatch.undo()  # No settlement/ownership mocks in this recovery test.
    journal, archive, refs = acknowledged(tmp_path)
    journal.discover_classifications()
    job = journal.claim_classification()
    plan = prepare_classification(MemoryLedger(archive, read_only=True), refs)
    journal.prepare_classification_job(job["job_id"], job["lease"], plan)
    generation = policy(tmp_path).generation
    admission = reserve(tmp_path, generation, READY, classification_job=job["job_id"],
                        classification_input=plan.input_sha256)
    admission.mark_unknown()
    journal.mark_classification_dispatch(job["job_id"], job["lease"], admission.identifier, generation)
    assert admission.settle_response({"usage": {"cost": 0.003}})  # synthetic response, real local ledger
    journal.stage_classification(job["job_id"], job["lease"], reply(plan), model="synthetic-luna")
    archive.backup_to(tmp_path / "backup")
    restored = SecureHistoryArchive.restore_from_backup(tmp_path / "backup", tmp_path / "restore", "synthetic recovery passphrase")
    root = restored.root.parent
    assert settled_response(root, admission.identifier, generation, require_unowned=True,
        classification_job=job["job_id"], classification_input=plan.input_sha256)
    assert not settled_response(root, admission.identifier, generation)
    reopened = CaptureJournal(restored, recover=False)
    assert reopened.publish_classification(job["job_id"]) == refs
    assert reopened.verify_classifications() == 1


def test_operator_reconciled_unknown_retains_backup_without_publication_or_retry(tmp_path, monkeypatch):
    from muninn.history.remote_accounting import reserve, _finish, settled_response
    from muninn.history.secure_archive import SecureHistoryArchive
    from tests.test_remote_accounting import policy, READY
    monkeypatch.undo()  # Exercise actual local accounting, not provider billing.
    journal, archive, refs = acknowledged(tmp_path)
    journal.discover_classifications()
    job = journal.claim_classification()
    plan = prepare_classification(MemoryLedger(archive, read_only=True), refs)
    journal.prepare_classification_job(job["job_id"], job["lease"], plan)
    generation = policy(tmp_path).generation
    admission = reserve(tmp_path, generation, READY, classification_job=job["job_id"],
                        classification_input=plan.input_sha256)
    admission.mark_unknown()
    journal.mark_classification_dispatch(job["job_id"], job["lease"], admission.identifier, generation)
    _finish(tmp_path, admission.identifier, 3000, "operator")
    assert not settled_response(tmp_path, admission.identifier, generation, require_unowned=True,
        classification_job=job["job_id"], classification_input=plan.input_sha256)
    with pytest.raises(ValueError, match="settlement missing"):
        journal.stage_classification(job["job_id"], job["lease"], reply(plan), model="synthetic-luna")
    assert journal.claim_classification(now=time.time() + 200) is None
    assert journal.classification_status() == {"outcome_unknown": 1}
    assert journal.verify_classifications() == 1
    archive.backup_to(tmp_path / "backup")
    restored = SecureHistoryArchive.restore_from_backup(tmp_path / "backup", tmp_path / "restore", "synthetic recovery passphrase")
    reopened = CaptureJournal(restored, recover=False)
    assert reopened.verify_classifications() == 1
    assert reopened.classification_status() == {"outcome_unknown": 1}
    assert reopened.claim_classification() is None


def test_correctly_sealed_invalid_pending_admission_and_foreign_owner_are_rejected(tmp_path, monkeypatch):
    from muninn.history.credential_crypto import VaultIntegrityError
    journal, archive, refs = acknowledged(tmp_path)
    journal.discover_classifications()
    job = journal.claim_classification()
    plan = prepare_classification(MemoryLedger(archive, read_only=True), refs)
    journal.prepare_classification_job(job["job_id"], job["lease"], plan)
    journal.mark_classification_dispatch(job["job_id"], job["lease"], "c" * 32, 1)
    with journal._connect() as db:
        original = journal._classification_job(db.execute("SELECT * FROM memory_classification_jobs").fetchone())
        journal._save_classification(db, job["job_id"], {**original, "state": "pending", "lease": None, "lease_until": None})
    with pytest.raises(VaultIntegrityError, match="journal authentication"):
        journal.verify_classifications()
    with journal._connect() as db:
        journal._save_classification(db, job["job_id"], original)
    monkeypatch.setattr("muninn.history.remote_accounting.classification_admission_state", lambda *a, **k: None)
    with pytest.raises(VaultIntegrityError, match="journal authentication"):
        journal.verify_classifications()


def test_pre_ack_event_must_match_entire_retained_stage(tmp_path, monkeypatch):
    from muninn.history.credential_crypto import VaultIntegrityError
    journal, archive, refs = acknowledged(tmp_path)
    journal.discover_classifications()
    job = journal.claim_classification()
    plan = prepare_classification(MemoryLedger(archive, read_only=True), refs)
    journal.prepare_classification_job(job["job_id"], job["lease"], plan)
    journal.mark_classification_dispatch(job["job_id"], job["lease"], "c" * 32, 1)
    monkeypatch.setattr("muninn.history.remote_accounting.settled_response", lambda *a, **k: True)
    journal.stage_classification(job["job_id"], job["lease"], reply(plan), model="synthetic-luna")
    with journal._connect() as db:
        stage = journal._classification_job(db.execute("SELECT * FROM memory_classification_jobs").fetchone())["stage"]
    MemoryLedger(archive).commit_classification(plan, reply(plan, bucket="procedure"),
        model_identity=stage["model_identity"], receipt=stage["receipt"])
    assert MemoryLedger(archive).verify_all()["decisions"] == 1
    with pytest.raises(VaultIntegrityError, match="durable publication"):
        journal.verify_classifications()
