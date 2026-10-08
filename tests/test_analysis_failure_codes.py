"""Failure diagnostics use real isolated journals, never real model calls."""
import json

import pytest

from muninn.history.capture_journal import CaptureJournal
from tests.test_capture_window_jobs import window_fixture


@pytest.mark.asyncio
@pytest.mark.parametrize("reason,subcode,expected,state", [
    ("local_output_invalid", "json", "local_output_json", "failed"),
    ("local_output_invalid", "cited_schema", "local_output_cited_schema", "failed"),
    ("local_output_invalid", "citation", "local_output_citation", "failed"),
    ("local_output_invalid", "quote_missing_or_ambiguous", "local_output_quote", "failed"),
    ("local_output_invalid", "analysis_schema", "local_output_analysis_schema", "failed"),
    ("local_output_invalid", None, "local_output_invalid", "failed"),
    ("local_output_invalid", [], "local_output_invalid", "failed"),
    ("local_output_invalid", {}, "local_output_invalid", "failed"),
    ("local_output_invalid", "REJECTED_PRIVATE_TEXT", "local_output_invalid", "failed"),
    ("gpu_busy", "json", "gpu_busy", "retry"),
])
async def test_worker_preserves_only_fixed_failure_categories(
        tmp_path, monkeypatch, reason, subcode, expected, state):
    journal, archive, receipt = window_fixture(tmp_path, text="A bounded ordinary observation.")
    assert journal.queue_capture_windows(receipt, limit=1)["queued"] == 1
    with journal._connect() as db:
        job_id = db.execute("SELECT job_id FROM history_analysis_jobs").fetchone()[0]
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(archive.root))
    from muninn.history import secure_analysis
    from muninn.history.service import HistoryService
    service = HistoryService(None, tmp_path / "unused", home=tmp_path,
                             archive_passphrase="test-only portable passphrase")

    def forbidden(*args, **kwargs):
        pytest.fail("Local capture must not consult remote policy")

    monkeypatch.setattr("muninn.history.auto_routing.remote_policy_snapshot", forbidden)

    async def analyze(history, source, descriptor, *, allow_remote, should_cancel,
                      before_remote, remote_not_sent, expected_remote_generation,
                      prefer_remote=False, remote_gate=None):
        assert allow_remote is False and expected_remote_generation == -1
        assert prefer_remote is False
        assert await before_remote() is False
        return {"status": "deferred", "reason": reason, "output_failure": subcode,
                "content": "REJECTED_PRIVATE_TEXT"}

    monkeypatch.setattr(secure_analysis, "analyze_cited_window", analyze)
    assert await service._process_secure_analysis_once(include_capture=True)
    reopened = CaptureJournal(archive)
    status = reopened.get_analysis_job(job_id)
    assert status["state"] == state
    assert status["error_code"] == expected
    assert status["result"] is None and "memory_refs" not in status
    assert "REJECTED_PRIVATE_TEXT" not in json.dumps(status)
    assert b"REJECTED_PRIVATE_TEXT" not in reopened.path.read_bytes()
    assert reopened.capture_window_status(receipt)["acknowledged"] == 0
    assert reopened.verify_all() == 0
    if state == "failed":
        assert reopened.claim_analysis(include_capture=True) is None


@pytest.mark.usefixtures("fake_strict_remote_admission")
def test_typed_local_failure_does_not_override_unknown_remote_outcome(tmp_path):
    from tests.test_secure_analysis_journal import _journal, _result, _target
    journal, archive = _journal(tmp_path)
    search = journal.enqueue_search("needle")
    worker = journal.claim_search()
    assert journal.finish_search(search, worker.lease_token, _result(),
                                 analysis_target=_target(archive), remote_policy_generation=7)
    job = journal.claim_analysis()
    assert journal.mark_remote_dispatched(job.job_id, job.lease_token)
    assert journal.defer_analysis(job.job_id, job.lease_token, "local_output_json")
    status = CaptureJournal(archive).get_analysis_job(job.job_id)
    assert status["state"] == status["error_code"] == "outcome_unknown"
