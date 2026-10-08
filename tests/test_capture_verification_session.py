"""Offline verifier reuse stays invocation-local; synthetic encrypted sources only."""
import json

import pytest

from muninn.history.cited_windows import CitedWindowPlanStore
from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.memory_ledger import MemoryLedger, MemoryLedgerIntegrityError
from muninn.history.secure_analysis import _cited_model_identity
from tests.test_capture_window_jobs import window_fixture
from tests.test_capture_window_reuse import MODEL, DIGEST


def two_reuses(tmp_path):
    journal, archive, receipt = window_fixture(tmp_path)
    journal.queue_capture_windows(receipt, limit=2)
    plans = CitedWindowPlanStore(archive)
    source = plans.source
    for ordinal in range(2):
        job = journal.claim_analysis(include_capture=True)
        assert job.target["ordinal"] == ordinal
        entry = source.ledger._entries[(receipt["blob"], 0)]
        descriptor = plans.window_at(entry, 0, job.target["plan_attempt"], ordinal)
        window = source.reopen(descriptor)
        quote = "local capture observation"
        start = window["text"].index(quote)
        stage = {"format": 1, "window": descriptor,
            "model_identity": _cited_model_identity(window, "ollama", MODEL, DIGEST),
            "proposals": [{"type": "observation", "text": "Local capture observation.",
                           "quote": quote, "start": start}],
            "result": {"status": "ok", "provider": "ollama", "model": MODEL,
                "analysis": {"summary": "Synthetic plumbing fixture.", "decisions": [],
                             "open_items": [], "uncertainty": "Not model quality evidence."}}}
        assert journal.bind_analysis_window(job.job_id, job.lease_token, descriptor)
        assert journal.stage_analysis(job.job_id, job.lease_token, stage)
        assert journal.begin_publication(job.job_id, job.lease_token)
        refs = source.record_proposals(descriptor, stage["proposals"], model_identity=stage["model_identity"])
        assert journal.acknowledge_publication(job.job_id, job.lease_token, refs)
    path = tmp_path / "session.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "event_msg", "payload": {
            "type": "user_message", "message": "More synthetic context."}}) + "\n")
    current = archive.archive_file(path, "codex", include_snapshot_receipt=True)["snapshot_receipt"]
    assert journal.enqueue_enrichment_receipt(current) == "queued"
    journal.queue_capture_windows(current, limit=2)
    plans = CitedWindowPlanStore(archive)
    jobs = []
    for ordinal in range(2):
        job = journal.claim_analysis(include_capture=True)
        descriptor = plans.window_at(plans.source.ledger._entries[(current["blob"], 1)],
                                     1, job.target["plan_attempt"], ordinal)
        assert journal.bind_analysis_window(job.job_id, job.lease_token, descriptor)
        assert journal.acknowledge_capture_reuse(job.job_id, job.lease_token,
                                                model=MODEL, weights_digest=DIGEST)
        jobs.append(job.job_id)
    return journal, archive, jobs


def verify_windows(journal):
    with journal._connect() as db:
        journal._verify_capture_window_jobs(db)


def test_multiple_reuses_authenticate_ledger_once_per_invocation(tmp_path, monkeypatch):
    journal, _archive, _jobs = two_reuses(tmp_path)
    calls = []
    original = MemoryLedger._snapshot
    def counted(ledger, db):
        calls.append(ledger)
        return original(ledger, db)
    monkeypatch.setattr(MemoryLedger, "_snapshot", counted)
    verify_windows(journal)
    assert len(calls) == 1
    assert calls[0].read_only is True
    verify_windows(journal)
    assert len(calls) == 2  # No cache survives the verification invocation.
    assert calls[1] is not calls[0] and calls[1].read_only is True


def test_later_reuse_tamper_is_not_hidden_by_first_success(tmp_path):
    journal, _archive, jobs = two_reuses(tmp_path)
    verify_windows(journal)
    with journal._connect() as db:
        db.execute("UPDATE history_analysis_jobs SET sealed_reuse=zeroblob(length(sealed_reuse)) WHERE job_id=?",
                   (jobs[1],))
    with pytest.raises(VaultIntegrityError):
        verify_windows(journal)


def test_next_invocation_reauthenticates_changed_ledger(tmp_path):
    journal, archive, _jobs = two_reuses(tmp_path)
    verify_windows(journal)
    ledger = MemoryLedger(archive)
    with ledger._connect() as db:
        db.execute("UPDATE events SET ciphertext=zeroblob(length(ciphertext)) WHERE seq=2")
    with pytest.raises(MemoryLedgerIntegrityError):
        verify_windows(journal)
