"""Durable batch ownership on isolated encrypted archives; no network/inference."""
import json
import sqlite3
from copy import deepcopy

import pytest

from muninn.history.capture_journal import CaptureJournal
from muninn.history.cited_analysis_source import CitedAnalysisSource
from muninn.history.cited_windows import CitedWindowPlanStore
from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.historical_batch import MODEL, BatchError, BatchOutbox, prepare_items, validate_item
from muninn.history.remote_accounting import reserve
from muninn.history.remote_policy import write_policy
from tests.test_capture_window_jobs import window_fixture


def fixture(tmp_path):
    journal, archive, receipt = window_fixture(tmp_path, text="An ordinary cited observation. " * 140)
    write_policy(journal.policy_root, enabled=True, daily_usd=5, monthly_usd=50,
                 override_ceiling=False, fallback=lambda: (False, 1, 30, False))
    journal.queue_capture_windows(receipt, limit=2, remote_policy_generation=1)
    plans = CitedWindowPlanStore(archive)
    bindings = []
    with journal._connect() as db:
        for row in db.execute("SELECT * FROM history_analysis_jobs ORDER BY created_at,job_id"):
            target = journal._validated_analysis_target(row, db)
            entry = plans.source.ledger._entries[(target["blob"], target["version"])]
            window = plans.window_at(entry, target["version"], target["plan_attempt"], target["ordinal"])
            bindings.append((row["job_id"], window))
    assert len(bindings) == 2
    outbox = BatchOutbox(archive)
    ident = outbox.prepare(prepare_items(CitedAnalysisSource(archive), bindings), consent_generation=1)
    return journal, archive, outbox, ident, bindings


def admission(journal, batch_owner=None):
    admitted = reserve(journal.policy_root, 1, {"admission_ready": True,
                     "usage_daily_usd": 0, "usage_monthly_usd": 0}, batch_owner=batch_owner)
    admitted.mark_unknown()
    return admitted


def test_ownership_survives_restart_and_blocks_ordinary_claims(tmp_path):
    journal, archive, _outbox, ident, bindings = fixture(tmp_path)
    journal.reserve_historical_batch(ident)
    reopened = CaptureJournal(archive)
    assert reopened.claim_analysis(include_capture=True) is None
    assert reopened.historical_batch_owner()["id"] == ident
    assert reopened.historical_batch_owner()["phase"] == "owned"
    with reopened._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM history_analysis_jobs WHERE lease_until IS NOT NULL").fetchone()[0] == 0
    assert reopened.verify_all() == 0
    assert b"ordinary cited observation" not in reopened.path.read_bytes()


def test_only_one_owner_and_unresolved_checkpoint_cannot_pass(tmp_path):
    journal, _archive, outbox, ident, _bindings = fixture(tmp_path)
    journal.reserve_historical_batch(ident)
    another = outbox.prepare(outbox.read(ident)["items"], consent_generation=1)
    with pytest.raises(BatchError):
        journal.reserve_historical_batch(another)
    assert not journal.finish_historical_batch(ident)
    assert journal.historical_batch_owner()["phase"] == "owned"


@pytest.mark.parametrize("sent", [False, True])
def test_normal_cancellation_cannot_detach_owned_member(tmp_path, sent):
    journal, _archive, outbox, ident, bindings = fixture(tmp_path)
    journal.reserve_historical_batch(ident)
    if sent:
        paid = admission(journal, ident)
        outbox.begin_submission(ident, 0)
        journal.mark_historical_batch_dispatched(ident, paid.identifier)
    for job_id, _window in bindings:
        assert not journal.cancel_analysis(job_id)
        assert not journal.request_analysis_cancel(job_id)
    assert journal.claim_analysis(include_capture=True) is None


def test_portable_archive_restore_keeps_unsent_ownership_and_outbox(tmp_path):
    from muninn.history.secure_archive import SecureHistoryArchive

    journal, archive, outbox, ident, _bindings = fixture(tmp_path)
    journal.reserve_historical_batch(ident)
    destination = tmp_path / "portable-backup"
    report = archive.backup_to(destination)
    assert report["historical_batches_verified"] == 1
    restored = SecureHistoryArchive.restore_from_backup(destination, tmp_path / "restored",
                                                       "test-only portable passphrase")
    recovered = CaptureJournal(restored)
    assert recovered.historical_batch_owner()["id"] == ident
    assert recovered.claim_analysis(include_capture=True) is None
    assert BatchOutbox(restored).read(ident)["items"] == outbox.read(ident)["items"]
    assert BatchOutbox(restored).verify_all() == {"batches": 1}


@pytest.mark.parametrize("replay,backup_damage", [(False, None), (True, None),
                                                (False, "earlier_outbox"), (False, "different_reply")])
def test_checkpoint_requires_every_durable_publication_not_provider_success(tmp_path, replay, backup_damage):
    journal, archive, outbox, ident, bindings = fixture(tmp_path)
    prepared_record = outbox.read(ident)
    source = CitedAnalysisSource(archive)
    journal.reserve_historical_batch(ident)
    paid = admission(journal, ident)
    outbox.begin_submission(ident, 0)
    journal.mark_historical_batch_dispatched(ident, paid.identifier)
    submitted = {"id": "batch_fixture", "model": MODEL, "endpoint": "/v1/chat/completions",
                 "completion_window": "24h", "status": "validating", "request_counts": {"total": 2}}
    outbox.save_submission(ident, 1, submitted)
    items = outbox.read(ident)["items"]
    rows = []
    for item in items:
        text = source.reopen(item["window"])["text"]
        output = {"summary": "An observation.", "decisions": [], "open_items": [], "uncertainty": "",
                  "proposals": [{"type": "fact", "text": "An ordinary cited observation.",
                                 "quote": text[:64], "start": 0}]}
        rows.append({"custom_id": item["custom_id"], "error": None, "response": {
            "status_code": 200, "body": {"model": MODEL, "choices": [{"finish_reason": "stop",
                "message": {"content": json.dumps(output)}}]}}})
    completed = {**submitted, "status": "completed", "results": rows,
                 "request_counts": {"total": 2, "completed": 2, "failed": 0},
                 "usage": {"cost": 0.001, "is_byok": False}}
    outbox.save_terminal(ident, 2, completed)
    assert paid.settle_response(completed)
    assert not journal.finish_historical_batch(ident)
    for index, (item, row) in enumerate(zip(items, rows)):
        checked = validate_item(source, item, row)
        stage = {**checked["extraction"], "admission_id": paid.identifier}
        job = journal.claim_historical_batch_result(ident, item["job_id"])
        assert job and journal.stage_analysis(job.job_id, job.lease_token, stage)
        assert journal.begin_publication(job.job_id, job.lease_token)
        refs = source.record_proposals(stage["window"], stage["proposals"], model_identity=stage["model_identity"])
        assert journal.acknowledge_publication(job.job_id, job.lease_token, refs)
        assert journal.finish_historical_batch(ident) == (index == 1)
        assert journal.claim_historical_batch_result(ident, item["job_id"]) is None
    assert journal.historical_batch_owner()["phase"] == "passed"
    assert journal.verify_all() == 0
    assert journal.verify_publications() == 2
    if backup_damage:
        record = prepared_record if backup_damage == "earlier_outbox" else outbox.read(ident)
        if backup_damage == "different_reply":
            message = record["terminal"]["results"][0]["response"]["body"]["choices"][0]["message"]
            content = json.loads(message["content"])
            content["summary"] = "Another valid summary."
            message["content"] = json.dumps(content)
        # Simulate individually authentic but mismatched cross-store snapshots.
        with sqlite3.connect(outbox.path) as db:
            db.execute("UPDATE batches SET revision=?,state=?,sealed=? WHERE id=?", (
                record["revision"], record["state"], outbox._seal(record), ident))
        with pytest.raises(VaultIntegrityError):
            journal.verify_all()
        return
    with journal._connect() as db:
        passed_head = db.execute("SELECT sealed_head FROM historical_batch_control").fetchone()[0]
    later = tmp_path / "later.jsonl"
    later.write_text(json.dumps({"type": "event_msg", "payload": {
        "type": "user_message", "message": "A subsequent independent observation."}}) + "\n", encoding="utf-8")
    receipt = archive.archive_file(later, "codex", include_snapshot_receipt=True)["snapshot_receipt"]
    journal.enqueue_enrichment_receipt(receipt)
    journal.queue_capture_windows(receipt, limit=1, remote_policy_generation=1)
    plans = CitedWindowPlanStore(archive)
    with journal._connect() as db:
        row = db.execute("SELECT * FROM history_analysis_jobs WHERE state='pending'").fetchone()
        target = journal._validated_analysis_target(row, db)
    entry = plans.source.ledger._entries[(target["blob"], target["version"])]
    descriptor = plans.window_at(entry, target["version"], target["plan_attempt"], target["ordinal"])
    next_ident = outbox.prepare(prepare_items(CitedAnalysisSource(archive), [(row["job_id"], descriptor)]),
                               consent_generation=1)
    journal.reserve_historical_batch(next_ident)
    assert journal.verify_all() == 0
    if replay:
        with journal._connect() as db:
            db.execute("UPDATE historical_batch_control SET sealed_head=?", (passed_head,))
        with pytest.raises(VaultIntegrityError):
            journal.claim_analysis(include_capture=True)
    else:
        assert journal.claim_analysis(include_capture=True) is None


def test_unrelated_unknown_admission_cannot_be_attached_to_a_batch(tmp_path):
    journal, _archive, outbox, ident, _bindings = fixture(tmp_path)
    journal.reserve_historical_batch(ident)
    unrelated = admission(journal)
    outbox.begin_submission(ident, 0)
    with pytest.raises(BatchError, match="admission_unresolved"):
        journal.mark_historical_batch_dispatched(ident, unrelated.identifier)
    assert journal.historical_batch_owner()["phase"] == "owned"
    assert journal.claim_analysis(include_capture=True) is None


@pytest.mark.parametrize("defect", ["claimed", "generation", "cancelled", "wrong_body", "duplicate"])
def test_reservation_rechecks_jobs_and_exact_screened_input(tmp_path, defect):
    journal, _archive, outbox, ident, bindings = fixture(tmp_path)
    if defect == "claimed":
        assert journal.claim_analysis(include_capture=True)
    elif defect in {"generation", "cancelled"}:
        with journal._connect() as db:
            db.execute("UPDATE history_analysis_jobs SET " + (
                "remote_policy_generation=2" if defect == "generation" else "cancel_requested=1"))
    else:
        items = deepcopy(outbox.read(ident)["items"])
        if defect == "wrong_body":
            items[0]["body"]["messages"][0]["content"] = "different source"
        else:
            items[1]["job_id"] = bindings[0][0]
        if defect == "duplicate":
            with pytest.raises(BatchError):
                outbox.prepare(items, consent_generation=1)
            return
        ident = outbox.prepare(items, consent_generation=1)
    with pytest.raises((BatchError, VaultIntegrityError)):
        journal.reserve_historical_batch(ident)
    assert journal.historical_batch_owner() is None


def test_sent_owner_never_becomes_runnable_after_lease_timeout(tmp_path):
    journal, archive, outbox, ident, bindings = fixture(tmp_path)
    journal.reserve_historical_batch(ident)
    paid = admission(journal, ident)
    with pytest.raises(BatchError):
        journal.mark_historical_batch_dispatched(ident, paid.identifier)
    outbox.begin_submission(ident, 0)
    journal.mark_historical_batch_dispatched(ident, paid.identifier)
    reopened = CaptureJournal(archive)
    assert reopened.claim_analysis(include_capture=True) is None
    assert reopened.historical_batch_owner()["phase"] == "sent"
    assert reopened.claim_historical_batch_result(ident, bindings[0][0]) is None
    assert paid.settle_response({"usage": {"cost": 0.001}})
    job = reopened.claim_historical_batch_result(ident, bindings[0][0])
    assert job and job.window == bindings[0][1] and job.lease_token
    with reopened._connect() as db:
        db.execute("UPDATE history_analysis_jobs SET lease_until=0 WHERE job_id=?", (job.job_id,))
    again = CaptureJournal(archive)
    assert again.claim_analysis(include_capture=True) is None
    recovered = again.claim_historical_batch_result(ident, job.job_id)
    assert recovered and recovered.lease_token != job.lease_token
    assert not again.finish_historical_batch(ident)


@pytest.mark.parametrize("damage", ["head", "owner", "missing", "window"])
def test_tampering_fails_closed_before_ordinary_claim(tmp_path, damage):
    journal, _archive, _outbox, ident, _bindings = fixture(tmp_path)
    journal.reserve_historical_batch(ident)
    with sqlite3.connect(journal.path) as db:
        if damage == "head":
            db.execute("UPDATE historical_batch_control SET sealed_head=?", (b"damaged",))
        elif damage == "owner":
            db.execute("UPDATE historical_batch_owners SET sealed_owner=?", (b"damaged",))
        elif damage == "missing":
            db.execute("DELETE FROM historical_batch_owners")
        else:
            db.execute("UPDATE history_analysis_jobs SET sealed_window=?", (b"damaged",))
    with pytest.raises(VaultIntegrityError):
        journal.claim_analysis(include_capture=True)
