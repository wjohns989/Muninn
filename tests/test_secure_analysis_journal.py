import sqlite3

import pytest

from muninn.history.capture_journal import CaptureJournal, SearchJobError
from muninn.history.secure_archive import SecureHistoryArchive


def _journal(tmp_path):
    archive = SecureHistoryArchive.create(tmp_path / "archive", "test-only passphrase")
    return CaptureJournal(archive), archive


def _target(archive):
    return {"vault_id": archive.vault_id, "blob": "a" * 32, "sha256": "b" * 64, "version": 0, "terms": ["needle"]}


def _result():
    return {
        "matches": [
            {
                "ref": "r",
                "provider": "p",
                "kind": "k",
                "captured_day_utc": "2026-01-01",
                "size_bucket_kib": 1,
                "versions": 1,
                "fetch_capability": "c",
            }
        ],
        "total": 1,
        "ready": 1,
        "missing": 0,
        "overflow": 0,
        "complete": True,
        "truncated": False,
    }


def test_analysis_link_dedup_and_target_not_plaintext(tmp_path):
    journal, archive = _journal(tmp_path)
    first = journal.enqueue_search("needle")
    worker = journal.claim_search()
    assert worker and journal.finish_search(first, worker.lease_token, _result(), analysis_target=_target(archive))
    second = journal.enqueue_search("needle")
    worker = journal.claim_search()
    assert worker and journal.finish_search(second, worker.lease_token, _result(), analysis_target=_target(archive))
    row = journal.get_search_job(first)
    row2 = journal.get_search_job(second)
    assert row["analysis_job_id"] == row2["analysis_job_id"]
    assert b"needle" not in (archive.root / "capture-jobs.db").read_bytes()


def test_analysis_lease_fence_and_result_allowlist(tmp_path):
    journal, archive = _journal(tmp_path)
    search = journal.enqueue_search("needle")
    worker = journal.claim_search()
    journal.finish_search(search, worker.lease_token, _result(), analysis_target=_target(archive))
    analysis = journal.claim_analysis()
    assert analysis
    valid = {"status": "ok", "provider": "ollama", "model": "m", "analysis": {
        "summary": "s", "decisions": [], "open_items": [], "uncertainty": "u"}}
    assert not journal.finish_analysis(analysis.job_id, "wrong", valid)
    assert journal.finish_analysis(
        analysis.job_id,
        analysis.lease_token,
        {
            "status": "ok",
            "provider": "ollama",
            "model": "m",
            "analysis": {"summary": "s", "decisions": [], "open_items": [], "uncertainty": "u"},
        },
    )
    visible = journal.get_analysis_job(analysis.job_id)
    assert visible["provider"] == "ollama"
    assert visible["model"] == "m"
    assert visible["provisional"] is True
    assert visible["result"]["analysis"]["summary"] == "s"


@pytest.mark.parametrize("bad", [
    {"status": "ok", "provider": "ollama", "model": "m"},
    {"status": "deferred", "provider": "ollama", "model": "m", "analysis": {}},
    {"status": "ok", "provider": "local", "model": "m", "analysis": {}},
    {"status": "ok", "provider": "ollama", "model": "m", "analysis": {
        "summary": "s" * 1201, "decisions": [], "open_items": [], "uncertainty": "u"}},
])
def test_analysis_result_rejects_unbounded_or_incomplete_output(bad):
    with pytest.raises(SearchJobError):
        CaptureJournal._allow_analysis_result(bad)


def test_resource_deferral_remains_retryable_and_remote_failure_is_uncertain(tmp_path):
    journal, archive = _journal(tmp_path)
    search = journal.enqueue_search("needle")
    worker = journal.claim_search()
    journal.finish_search(search, worker.lease_token, _result(), analysis_target=_target(archive))
    job = journal.claim_analysis()
    for _ in range(4):
        assert journal.defer_analysis(job.job_id, job.lease_token, "daily_zdr_cap_unverified")
        row = journal.get_analysis_job(job.job_id)
        assert row["state"] == "retry" and row["due_at"] > 0
        with sqlite3.connect(journal.path) as db:
            db.execute("UPDATE history_analysis_jobs SET due_at=0 WHERE job_id=?", (job.job_id,))
        job = journal.claim_analysis()
    assert journal.mark_remote_dispatched(job.job_id, job.lease_token)
    assert not journal.mark_remote_dispatched(job.job_id, job.lease_token)
    assert journal.fail_analysis(job.job_id, job.lease_token, "model_unavailable")
    assert journal.get_analysis_job(job.job_id)["state"] == "outcome_unknown"


def test_disabled_analysis_is_reported_without_queue(tmp_path):
    journal, _ = _journal(tmp_path)
    search = journal.enqueue_search("needle")
    worker = journal.claim_search()
    assert journal.finish_search(search, worker.lease_token, _result(), analysis_reason="disabled")
    row = journal.get_search_job(search)
    assert "analysis_job_id" not in row
    assert row["analysis_state"] == "not_queued"
    assert row["analysis_reason"] == "disabled"


def test_remote_dispatch_recovery_is_outcome_unknown(tmp_path):
    journal, archive = _journal(tmp_path)
    search = journal.enqueue_search("needle")
    worker = journal.claim_search()
    journal.finish_search(search, worker.lease_token, _result(), analysis_target=_target(archive))
    analysis = journal.claim_analysis()
    assert analysis and journal.mark_remote_dispatched(analysis.job_id, analysis.lease_token)
    with sqlite3.connect(journal.path) as db:
        db.execute("UPDATE history_analysis_jobs SET lease_until=0 WHERE job_id=?", (analysis.job_id,))
    recovered = CaptureJournal(archive).get_analysis_job(analysis.job_id)
    assert recovered["state"] == "outcome_unknown"
