# ruff: noqa: E501
import sqlite3
import time

import pytest

from muninn.history.capture_journal import CaptureJournal, SearchJobError
from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.secure_archive import SecureHistoryArchive


def journal(tmp_path):
    archive = SecureHistoryArchive.create(tmp_path / "archive", "test-only passphrase")
    return CaptureJournal(archive), archive


def test_commit_restart_and_no_plaintext(tmp_path):
    j, archive = journal(tmp_path)
    job_id = j.enqueue_search("needle", 10)
    worker = CaptureJournal(archive).claim_search()
    assert worker and worker.job_id == job_id
    assert worker.lease_token
    result = {"matches": [], "total": 1, "ready": 1, "missing": 0, "overflow": 0, "complete": True, "truncated": False}
    assert j.finish_search(job_id, worker.lease_token, result)
    assert CaptureJournal(archive).get_search_job(job_id)["result"]["total"] == 1
    raw = (archive.root / "capture-jobs.db").read_bytes()
    assert b"needle" not in raw


def test_lease_fencing_and_expiry(tmp_path):
    j, archive = journal(tmp_path)
    job_id = j.enqueue_search("quick search")
    first = j.claim_search()
    assert first
    with sqlite3.connect(j.path) as db:
        db.execute("UPDATE history_search_jobs SET lease_until=? WHERE job_id=?", (time.time() - 1, job_id))
    assert CaptureJournal(archive).claim_search() is not None
    assert not j.finish_search(
        job_id,
        first.lease_token,
        {"matches": [], "total": 0, "ready": 0, "missing": 0, "overflow": 0, "complete": True, "truncated": False},
    )


def test_cancel_race_and_expired_result(tmp_path):
    j, _ = journal(tmp_path)
    job_id = j.enqueue_search("quick search")
    worker = j.claim_search()
    assert worker and j.cancel_search(job_id)
    assert not j.finish_search(
        job_id,
        worker.lease_token,
        {"matches": [], "total": 0, "ready": 0, "missing": 0, "overflow": 0, "complete": True, "truncated": False},
    )
    assert j.get_search_job(job_id) is None


def test_validation_and_malformed_sealed_row(tmp_path):
    j, _ = journal(tmp_path)
    with pytest.raises(SearchJobError):
        j.enqueue_search("x" * 513)
    with pytest.raises(SearchJobError):
        j.enqueue_search("   ", 1)
    with pytest.raises(SearchJobError):
        j.enqueue_search("\ud800")
    job_id = j.enqueue_search("safe")
    with sqlite3.connect(j.path) as db:
        db.execute("UPDATE history_search_jobs SET sealed_query=? WHERE job_id=?", (b"bad", job_id))
    with pytest.raises(VaultIntegrityError):
        j.verify_all()


def test_search_query_repr_and_typed_terminal_failure(tmp_path):
    j, _ = journal(tmp_path)
    job_id = j.enqueue_search("private project needle")
    worker = j.claim_search()
    assert worker and "private project needle" not in repr(worker)
    assert j.fail_search(job_id, worker.lease_token, "vault_integrity")
    assert j.get_search_job(job_id) == {
        "job_id": job_id, "state": "failed", "result": None, "error_code": "vault_integrity",
    }
    assert j.claim_search() is None


def test_transient_failure_retries_and_stale_lease_cannot_fail(tmp_path):
    j, _ = journal(tmp_path)
    job_id = j.enqueue_search("recoverable needle")
    first = j.claim_search()
    assert first and j.fail_search(job_id, first.lease_token, "locked")
    assert j.get_search_job(job_id)["state"] == "retry"
    with sqlite3.connect(j.path) as db:
        db.execute("UPDATE history_search_jobs SET due_at=? WHERE job_id=?", (time.time() - 1, job_id))
    second = j.claim_search()
    assert second and second.lease_token != first.lease_token
    assert not j.fail_search(job_id, first.lease_token, "unknown")
    assert j.fail_search(job_id, second.lease_token, "unknown")
    assert j.get_search_job(job_id)["error_code"] == "unknown"


def test_search_result_allowlist_rejects_oversize(tmp_path):
    j, _ = journal(tmp_path)
    job_id = j.enqueue_search("result needle")
    worker = j.claim_search()
    assert worker
    result = {"matches": [{"ref": "x", "provider": "p", "kind": "k", "captured_day_utc": "2026-09-28",
                          "size_bucket_kib": 1, "versions": 1, "fetch_capability": "x" * 513}],
              "total": 1, "ready": 1, "missing": 0, "overflow": 0, "complete": True, "truncated": False}
    with pytest.raises(SearchJobError):
        j.finish_search(job_id, worker.lease_token, result)


def test_backup_restore_preserves_search_job(tmp_path):
    j, archive = journal(tmp_path)
    job_id = j.enqueue_search("safe")
    backup = tmp_path / "backup"
    archive.backup_to(backup)
    restored = SecureHistoryArchive.restore_from_backup(backup, tmp_path / "restored", "test-only passphrase")
    assert CaptureJournal(restored).get_search_job(job_id)["state"] == "pending"
