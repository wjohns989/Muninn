"""Typed automatic window work; isolated encrypted fixtures, no provider calls."""
import json

import pytest

from muninn.history.capture_journal import CaptureJournal
from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.secure_archive import SecureHistoryArchive
from tests.test_secure_capture_journal import _journal


def window_fixture(tmp_path, *, text="A local capture observation. " * 180):
    journal, archive = _journal(tmp_path)
    journal.configure_enrichment(0)
    source = tmp_path / "session.jsonl"
    source.write_text(json.dumps({"type": "event_msg", "payload": {
        "type": "user_message", "message": text}}) + "\n", encoding="utf-8")
    receipt = archive.archive_file(source, "codex", include_snapshot_receipt=True)["snapshot_receipt"]
    assert journal.enqueue_enrichment_receipt(receipt) == "queued"
    return journal, archive, receipt


def test_typed_window_queue_resumes_atomically_without_fake_query(tmp_path):
    journal, archive, receipt = window_fixture(tmp_path)
    first = journal.queue_capture_windows(receipt, limit=1)
    assert first["queued"] == 1 and first["next_ordinal"] == 1 and first["windows"] >= 2
    assert journal.claim_analysis() is None  # Existing search consumer doesn't activate this lane.
    reopened = CaptureJournal(archive)
    second = reopened.queue_capture_windows(receipt, limit=4)
    assert second["next_ordinal"] == second["windows"]
    assert reopened.queue_capture_windows(receipt, limit=4)["queued"] == 0
    job = reopened.claim_analysis(include_capture=True)
    assert job.lane == 1 and job.target["kind"] == "capture_window"
    assert "terms" not in job.target and job.remote_policy_generation == -1
    assert job.target["ordinal"] == 0
    assert reopened.capture_window_status(receipt)["state"] == "processing"
    assert b"local capture observation" not in reopened.path.read_bytes()
    assert reopened.verify_all() == 0


def test_automatic_capacity_reserves_foreground_slots_without_consuming_cursor(tmp_path):
    journal, _archive, receipt = window_fixture(tmp_path, text="Long ordinary message. " * 4500)
    first = journal.queue_capture_windows(receipt, limit=32)
    assert first["queued"] == 24 and first["next_ordinal"] == 24
    second = journal.queue_capture_windows(receipt, limit=32)
    assert second["state"] == "queue_full" and second["next_ordinal"] == 24
    with journal._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM history_analysis_jobs WHERE lane=1").fetchone()[0] == 24


def test_receipt_not_in_authenticated_outbox_cannot_create_automatic_jobs(tmp_path):
    journal, _archive, receipt = window_fixture(tmp_path)
    forged = {**receipt, "blob": "f" * 32}
    with pytest.raises(VaultIntegrityError):
        journal.queue_capture_windows(forged)
    with journal._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM history_analysis_jobs").fetchone()[0] == 0


def test_lane_tamper_is_rejected_at_claim_and_publication_binding(tmp_path):
    journal, _archive, receipt = window_fixture(tmp_path)
    journal.queue_capture_windows(receipt, limit=1)
    with journal._connect() as db:
        db.execute("UPDATE history_analysis_jobs SET lane=0")
    with pytest.raises(VaultIntegrityError):
        journal.claim_analysis()
    with journal._connect() as db:
        row = db.execute("SELECT * FROM history_analysis_jobs").fetchone()
        assert row["state"] == "pending"
    with pytest.raises(VaultIntegrityError):
        journal._window_purpose(row)


def test_automatic_target_is_not_accepted_by_search_enqueue_validator(tmp_path):
    journal, _archive, receipt = window_fixture(tmp_path)
    journal.queue_capture_windows(receipt, limit=1)
    job = journal.claim_analysis(include_capture=True)
    assert journal._analysis_target(job.target, ["observation"], journal.archive.vault_id) is None
    assert not journal.mark_remote_dispatched(job.job_id, job.lease_token)


def test_remote_capture_job_binds_generation_and_recovers_without_repost(tmp_path):
    journal, archive, receipt = window_fixture(tmp_path)
    journal.queue_capture_windows(receipt, limit=1, remote_policy_generation=2)
    job = journal.claim_analysis(include_capture=True)
    assert job.remote_policy_generation == 2
    assert job.target["remote_policy_generation"] == 2
    assert journal.mark_remote_dispatched(job.job_id, job.lease_token)
    with journal._connect() as db:
        db.execute("UPDATE history_analysis_jobs SET lease_until=0 WHERE job_id=?", (job.job_id,))
    assert journal.claim_analysis(include_capture=True) is None
    assert journal.get_analysis_job(job.job_id)["state"] == "outcome_unknown"
    assert journal.capture_window_status(receipt)["acknowledged"] == 0
    assert journal.verify_all() == 0


def test_remote_capture_stage_requires_dispatch_marker(tmp_path):
    journal, archive, receipt = window_fixture(tmp_path)
    from muninn.history.remote_policy import write_policy
    from muninn.history.remote_accounting import reserve
    write_policy(journal.policy_root, enabled=True, daily_usd=5, monthly_usd=50,
                 override_ceiling=False, fallback=lambda: (False, 1, 30, False))
    journal.queue_capture_windows(receipt, limit=1, remote_policy_generation=1)
    job = journal.claim_analysis(include_capture=True)
    from muninn.history.cited_windows import CitedWindowPlanStore
    plans = CitedWindowPlanStore(archive)
    entry = plans.source.ledger._entries[(receipt["blob"], receipt["version"])]
    descriptor = plans.window_at(entry, receipt["version"], job.target["plan_attempt"], 0)
    assert journal.bind_analysis_window(job.job_id, job.lease_token, descriptor)
    result = {"status": "ok", "provider": "openrouter", "model": "fixture-zdr",
              "analysis": {"summary": "No supported claim.", "decisions": [],
                           "open_items": [], "uncertainty": ""}}
    stage = {"format": 1, "window": descriptor, "proposals": [],
             "model_identity": "a" * 64, "result": result}
    from muninn.history.capture_journal import SearchJobError
    with pytest.raises(SearchJobError, match="provider is not authorized"):
        journal.stage_analysis(job.job_id, job.lease_token, stage)
    assert journal.mark_remote_dispatched(job.job_id, job.lease_token)
    with pytest.raises(SearchJobError, match="settlement is missing"):
        journal.stage_analysis(job.job_id, job.lease_token, stage)
    admission = reserve(journal.policy_root, 1, {"admission_ready": True,
        "usage_daily_usd": 0, "usage_monthly_usd": 0})
    admission.mark_unknown()
    stage["admission_id"] = admission.identifier
    with pytest.raises(SearchJobError, match="not verified"):
        journal.stage_analysis(job.job_id, job.lease_token, stage)
    assert admission.settle_response({"usage": {"cost": 0.001}})
    assert journal.stage_analysis(job.job_id, job.lease_token, stage)
    assert journal.begin_publication(job.job_id, job.lease_token)
    assert journal.acknowledge_publication(job.job_id, job.lease_token, [])
    assert journal.get_analysis_job(job.job_id)["state"] == "succeeded"
    assert journal.capture_window_status(receipt)["acknowledged"] == 1
    assert journal.verify_all() == 0


def test_queue_transaction_rolls_back_jobs_mapping_and_cursor_together(tmp_path, monkeypatch):
    journal, archive, receipt = window_fixture(tmp_path)
    original = journal._seal_search

    def fail_at_binding(value, job_id, purpose):
        if purpose == "capture-window-binding-v1":
            raise RuntimeError("Simulated journal interruption")
        return original(value, job_id, purpose)

    monkeypatch.setattr(journal, "_seal_search", fail_at_binding)
    with pytest.raises(RuntimeError):
        journal.queue_capture_windows(receipt, limit=1)
    with journal._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM history_analysis_jobs").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM capture_enrichment_windows").fetchone()[0] == 0
    reopened = CaptureJournal(archive)
    retry = reopened.queue_capture_windows(receipt, limit=1)
    assert retry["queued"] == 1 and retry["next_ordinal"] == 1


def test_zero_window_plan_is_no_context_not_completed_analysis(tmp_path):
    journal, archive = _journal(tmp_path)
    journal.configure_enrichment(0)
    source = tmp_path / "metadata.jsonl"
    source.write_text(json.dumps({"type": "session_meta", "payload": {"cwd": "C:/sample"}}) + "\n")
    receipt = archive.archive_file(source, "codex", include_snapshot_receipt=True)["snapshot_receipt"]
    journal.enqueue_enrichment_receipt(receipt)
    outcome = journal.queue_capture_windows(receipt)
    assert outcome["state"] == "no_context" and outcome["windows"] == 0
    assert journal.capture_window_status(receipt)["state"] == "no_context"
    assert journal.claim_analysis(include_capture=True) is None
    assert journal.enrichment_status()["pending_sources"] == 0
    assert journal.pending_enrichment() == []
    with journal._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM capture_enrichment_sources").fetchone()[0] == 1
    assert journal.verify_all() == 0


def test_portable_restore_keeps_window_jobs_and_scheduling_cursor(tmp_path):
    journal, archive, receipt = window_fixture(tmp_path)
    journal.queue_capture_windows(receipt, limit=1)
    restored = SecureHistoryArchive.restore_from_backup(
        archive.root, tmp_path / "restored", "test-only portable passphrase")
    recovered = CaptureJournal(restored)
    outcome = recovered.queue_capture_windows(receipt, limit=4)
    assert outcome["next_ordinal"] == outcome["windows"]
    with recovered._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM history_analysis_jobs").fetchone()[0] == outcome["windows"]
    assert recovered.verify_all() == 0


def test_pending_foreground_search_prevents_automatic_claim(tmp_path):
    journal, _archive, receipt = window_fixture(tmp_path)
    journal.queue_capture_windows(receipt, limit=1)
    search_id = journal.enqueue_search("observation")
    assert journal.claim_analysis(include_capture=True) is None
    assert journal.cancel_search(search_id)
    assert journal.claim_analysis(include_capture=True).lane == 1


def test_search_analysis_has_priority_and_cannot_manufacture_capture_target(tmp_path):
    journal, archive, receipt = window_fixture(tmp_path)
    journal.queue_capture_windows(receipt, limit=1)
    capture_job = journal.claim_analysis(include_capture=True)
    with journal._connect() as db:
        db.execute("UPDATE history_analysis_jobs SET state='pending',lease_token=NULL,lease_until=NULL")
    from tests.test_secure_analysis_journal import _result, _target
    search_id = journal.enqueue_search("needle")
    search = journal.claim_search()
    assert journal.finish_search(search_id, search.lease_token, _result(), analysis_target=capture_job.target)
    search_status = journal.get_search_job(search_id)
    assert "analysis_job_id" not in search_status
    assert search_status["analysis_reason"] == "target_unavailable"
    assert journal.capture_window_status(receipt)["queued"] == 1
    assert journal.capture_window_status(receipt)["acknowledged"] == 0

    search_id = journal.enqueue_search("needle")
    search = journal.claim_search()
    assert journal.finish_search(search_id, search.lease_token, _result(), analysis_target=_target(archive))
    next_job = journal.claim_analysis(include_capture=True)
    assert next_job.lane == 0 and next_job.target["terms"] == ["needle"]


def test_legacy_result_only_ack_cannot_complete_automatic_window(tmp_path):
    journal, _archive, receipt = window_fixture(tmp_path)
    journal.queue_capture_windows(receipt, limit=1)
    job = journal.claim_analysis(include_capture=True)
    result = {"status": "ok", "provider": "ollama", "model": "fixture",
              "analysis": {"summary": "Result without durable publication.",
                           "decisions": [], "open_items": [], "uncertainty": ""}}
    assert journal.finish_analysis(job.job_id, job.lease_token, result) is False
    assert journal.capture_window_status(receipt)["acknowledged"] == 0


def test_concurrent_planner_cannot_overwrite_advanced_cursor(tmp_path, monkeypatch):
    journal, archive, receipt = window_fixture(tmp_path)
    from muninn.history.cited_windows import CitedWindowPlanStore
    original = CitedWindowPlanStore.window_at
    raced = False

    def race(plans, *args, **kwargs):
        nonlocal raced
        descriptor = original(plans, *args, **kwargs)
        if not raced:
            raced = True
            assert CaptureJournal(archive).queue_capture_windows(receipt, limit=2)["queued"] == 2
        return descriptor

    monkeypatch.setattr(CitedWindowPlanStore, "window_at", race)
    assert journal.queue_capture_windows(receipt, limit=1) == {"state": "concurrent_advance", "queued": 0}
    with journal._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM history_analysis_jobs").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM capture_enrichment_windows").fetchone()[0] == 2


@pytest.mark.parametrize("damage", ["binding", "mapping", "hint"])
def test_inconsistent_window_coverage_blocks_portable_restore(tmp_path, damage):
    journal, archive, receipt = window_fixture(tmp_path)
    journal.queue_capture_windows(receipt, limit=2)
    with journal._connect() as db:
        if damage == "binding":
            db.execute("UPDATE capture_enrichment_windows SET sealed_binding=zeroblob(length(sealed_binding))")
        elif damage == "mapping":
            db.execute("DELETE FROM capture_enrichment_windows WHERE ordinal=0")
        else:
            db.execute("UPDATE capture_enrichment_sources SET resolved=1")
    with pytest.raises(VaultIntegrityError):
        journal.verify_all()
    with pytest.raises(VaultIntegrityError):
        SecureHistoryArchive.restore_from_backup(
            archive.root, tmp_path / "restore-bad", "test-only portable passphrase")


def test_round_robin_planner_does_not_starve_later_sources(tmp_path):
    journal, archive, first = window_fixture(tmp_path, text="Long observation. " * 900)
    source = tmp_path / "other.jsonl"
    source.write_text(json.dumps({"type": "event_msg", "payload": {
        "type": "user_message", "message": "Another long observation. " * 900}}) + "\n", encoding="utf-8")
    second = archive.archive_file(source, "codex", include_snapshot_receipt=True)["snapshot_receipt"]
    journal.enqueue_enrichment_receipt(second)
    assert journal.next_capture_plan() == first
    journal.queue_capture_windows(first, limit=1)
    assert journal.next_capture_plan() == second
    journal.queue_capture_windows(second, limit=1)
    assert journal.next_capture_plan() == first


@pytest.mark.parametrize("hint", ["planning_complete", "resolved"])
def test_tampered_exclusion_hint_cannot_hide_unfinished_source(tmp_path, hint):
    journal, _archive, _receipt = window_fixture(tmp_path)
    with journal._connect() as db:
        db.execute(f"UPDATE capture_enrichment_sources SET {hint}=1")
    if hint == "planning_complete":
        with pytest.raises(VaultIntegrityError):
            journal.next_capture_plan()
    else:
        with pytest.raises(VaultIntegrityError):
            journal.enrichment_status()
        with pytest.raises(VaultIntegrityError):
            journal.pending_enrichment()


@pytest.mark.parametrize("damage", ["ciphertext", "missing_row"])
def test_scheduler_totals_fail_closed_without_silent_rebootstrap(tmp_path, damage):
    journal, archive, _receipt = window_fixture(tmp_path)
    with journal._connect() as db:
        if damage == "ciphertext":
            db.execute("UPDATE capture_enrichment_schedule SET sealed_totals=zeroblob(length(sealed_totals))")
        else:
            db.execute("DELETE FROM capture_enrichment_schedule")
    with pytest.raises(VaultIntegrityError):
        journal.next_capture_plan()
    with pytest.raises(VaultIntegrityError):
        journal.verify_all()
    if damage == "missing_row":
        with pytest.raises(VaultIntegrityError):
            CaptureJournal(archive)


def test_source_deletion_is_detected_before_empty_planner_status(tmp_path):
    journal, _archive, _receipt = window_fixture(tmp_path)
    with journal._connect() as db:
        db.execute("DELETE FROM capture_enrichment_sources")
    with pytest.raises(VaultIntegrityError):
        journal.next_capture_plan()
    with pytest.raises(VaultIntegrityError):
        journal.enrichment_status()


def test_balanced_hint_swap_is_rejected_when_inconsistent_source_is_selected(tmp_path):
    journal, archive, receipt = window_fixture(tmp_path)
    journal.queue_capture_windows(receipt, limit=4)
    source = tmp_path / "newer.jsonl"
    source.write_text(json.dumps({"type": "event_msg", "payload": {
        "type": "user_message", "message": "New unplanned observation."}}) + "\n")
    newer = archive.archive_file(source, "codex", include_snapshot_receipt=True)["snapshot_receipt"]
    journal.enqueue_enrichment_receipt(newer)
    with journal._connect() as db:
        db.execute("UPDATE capture_enrichment_sources SET planning_complete=1-planning_complete")
    with pytest.raises(VaultIntegrityError):
        journal.next_capture_plan()
    with pytest.raises(VaultIntegrityError):
        journal.verify_all()


def test_other_ordinal_cannot_be_bound_even_with_same_snapshot(tmp_path):
    journal, archive, receipt = window_fixture(tmp_path)
    journal.queue_capture_windows(receipt, limit=2)
    job = journal.claim_analysis(include_capture=True)
    from muninn.history.cited_windows import CitedWindowPlanStore
    plans = CitedWindowPlanStore(archive)
    entry = plans.source.ledger._entries[(receipt["blob"], receipt["version"])]
    other = plans.window_at(entry, receipt["version"], job.target["plan_attempt"], 1)
    with pytest.raises(VaultIntegrityError):
        journal.bind_analysis_window(job.job_id, job.lease_token, other)
    assert journal.get_analysis_job(job.job_id)["state"] == "running"
    assert journal.capture_window_status(receipt)["acknowledged"] == 0


@pytest.mark.asyncio
async def test_local_worker_acknowledges_all_windows_with_no_search_or_remote_policy(tmp_path, monkeypatch):
    journal, archive, receipt = window_fixture(tmp_path)
    planned = journal.queue_capture_windows(receipt, limit=4)
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(archive.root))
    from muninn.history.service import HistoryService
    from muninn.history import secure_analysis
    service = HistoryService(None, tmp_path / "unused", home=tmp_path,
                             archive_passphrase="test-only portable passphrase")
    seen = []

    def forbidden(*args, **kwargs):
        pytest.fail("Capture windows must not consult search terms or remote policy")

    monkeypatch.setattr("muninn.history.auto_routing.remote_policy_snapshot", forbidden)
    monkeypatch.setattr("muninn.history.blind_index.SecureHistoryBlindIndex._analysis_capability", forbidden)

    async def analyze(history, source, descriptor, *, allow_remote, should_cancel,
                      before_remote=None, remote_not_sent=None, expected_remote_generation,
                      prefer_remote=False, remote_gate=None):
        assert allow_remote is False and expected_remote_generation == -1
        assert prefer_remote is False
        assert await before_remote() is False
        seen.append(source.reopen(descriptor)["text"])
        result = {"status": "ok", "provider": "ollama", "model": "isolated-fixture-model",
                  "analysis": {"summary": "No supported claim extracted.", "decisions": [],
                               "open_items": [], "uncertainty": "Fixture validates plumbing only."}}
        return {**result, "extraction": {"format": 1, "window": descriptor, "proposals": [],
                                         "model_identity": "a" * 64, "result": result}}

    monkeypatch.setattr(secure_analysis, "analyze_cited_window", analyze)
    assert await service._process_secure_analysis_once() is False
    for _ in range(planned["windows"]):
        assert await service._process_secure_analysis_once(include_capture=True)
    status = journal.capture_window_status(receipt)
    assert status["state"] == "completed"
    assert status["acknowledged"] == status["windows"] == len(seen)
    assert "".join(seen) == "A local capture observation. " * 180
    assert journal.verify_all() == 0
    restored = SecureHistoryArchive.restore_from_backup(
        archive.root, tmp_path / "restored-completed", "test-only portable passphrase")
    assert CaptureJournal(restored).capture_window_status(receipt)["state"] == "completed"
    assert journal.enrichment_status()["pending_sources"] == 0
    assert journal.pending_enrichment() == []


@pytest.mark.asyncio
async def test_opted_in_capture_worker_stages_remote_only_with_bound_generation(tmp_path, monkeypatch):
    journal, archive, receipt = window_fixture(tmp_path)
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(archive.root))
    monkeypatch.setenv("MUNINN_CAPTURE_ENRICHMENT", "1")
    monkeypatch.setenv("MUNINN_CAPTURE_AUTO_ANALYSIS", "1")
    monkeypatch.setenv("MUNINN_CAPTURE_AUTO_REMOTE", "1")
    from muninn.history.service import HistoryService
    from muninn.history.remote_policy import write_policy
    from muninn.history import secure_analysis
    service = HistoryService(None, tmp_path / "unused", home=tmp_path,
                             archive_passphrase="test-only portable passphrase")
    service.data_dir.mkdir(parents=True, exist_ok=True)
    write_policy(service.data_dir, enabled=True, daily_usd=5, monthly_usd=50,
                 override_ceiling=False, fallback=lambda: (False, 1, 30, False))
    journal.queue_capture_windows(receipt, limit=1, remote_policy_generation=1)

    async def analyze(history, source, descriptor, *, allow_remote, should_cancel,
                      before_remote, remote_not_sent, expected_remote_generation,
                      prefer_remote, remote_gate):
        assert allow_remote and prefer_remote and expected_remote_generation == 1
        assert remote_gate()
        assert await before_remote()
        from muninn.history.remote_accounting import reserve
        admission = reserve(service.data_dir, 1, {"admission_ready": True,
            "usage_daily_usd": 0, "usage_monthly_usd": 0})
        admission.mark_unknown()
        assert admission.settle_response({"usage": {"cost": 0.001}})
        result = {"status": "ok", "provider": "openrouter", "model": "fixture-zdr",
                  "analysis": {"summary": "No supported claim.", "decisions": [],
                               "open_items": [], "uncertainty": ""}}
        return {**result, "extraction": {"format": 1, "window": descriptor,
                                         "proposals": [], "model_identity": "a" * 64,
                                         "result": result,
                                         "admission_id": admission.identifier}}

    monkeypatch.setattr(secure_analysis, "analyze_cited_window", analyze)
    assert await service._process_secure_analysis_once(include_capture=True)
    with journal._connect() as db:
        row = db.execute("SELECT state,provider,remote_dispatched FROM history_analysis_jobs").fetchone()
    assert tuple(row) == ("succeeded", "openrouter", 1)
    assert journal.capture_window_status(receipt)["acknowledged"] == 1
    assert journal.verify_all() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("remote_marked", [False, True])
@pytest.mark.parametrize("drain", [False, True])
async def test_remote_first_capture_falls_back_only_when_proven_unsent(
        tmp_path, monkeypatch, remote_marked, drain):
    journal, archive, receipt = window_fixture(tmp_path)
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(archive.root))
    monkeypatch.setenv("MUNINN_CAPTURE_ENRICHMENT", "1")
    monkeypatch.setenv("MUNINN_CAPTURE_AUTO_ANALYSIS", "1")
    monkeypatch.setenv("MUNINN_CAPTURE_AUTO_REMOTE", "1")
    from muninn.history.service import HistoryService
    from muninn.history.remote_policy import write_policy
    from muninn.history import secure_analysis
    service = HistoryService(None, tmp_path / "unused", home=tmp_path,
                             archive_passphrase="test-only portable passphrase")
    service.data_dir.mkdir(parents=True, exist_ok=True)
    write_policy(service.data_dir, enabled=True, daily_usd=5, monthly_usd=50,
                 override_ceiling=False, fallback=lambda: (False, 1, 30, False))
    journal.queue_capture_windows(receipt, limit=1, remote_policy_generation=1)
    attempts = []

    async def analyze(history, source, descriptor, *, allow_remote, prefer_remote=False,
                      should_cancel, before_remote=None, remote_not_sent=None,
                      expected_remote_generation, remote_gate=None):
        attempts.append("remote" if prefer_remote else "local")
        if prefer_remote:
            assert allow_remote and remote_gate()
            if remote_marked:
                assert await before_remote()
            return {"status": "deferred", "provider": None, "model": None,
                    "reason": "remote_cost_unresolved" if remote_marked else "remote_admission_threshold_reached"}
        assert not allow_remote and expected_remote_generation == -1
        result = {"status": "ok", "provider": "ollama", "model": "fixture-local",
                  "analysis": {"summary": "No supported claim.", "decisions": [],
                               "open_items": [], "uncertainty": ""}}
        return {**result, "extraction": {"format": 1, "window": descriptor,
                                         "proposals": [], "model_identity": "a" * 64,
                                         "result": result}}

    monkeypatch.setattr(secure_analysis, "analyze_cited_window", analyze)
    assert await service._process_secure_analysis_once(include_capture=True,
                                                      capture_remote_only=drain)
    with journal._connect() as db:
        row = db.execute("SELECT state,provider,remote_dispatched FROM history_analysis_jobs").fetchone()
    if remote_marked:
        assert attempts == ["remote"]
        assert tuple(row) == ("outcome_unknown", None, 1)
        assert journal.capture_window_status(receipt)["acknowledged"] == 0
    elif drain:
        assert attempts == ["remote"]
        assert tuple(row) == ("retry", None, 0)
        assert service._capture_drain.halted_reason == "remote_admission_threshold_reached"
        assert journal.capture_window_status(receipt)["acknowledged"] == 0
    else:
        assert attempts == ["remote", "local"]
        assert tuple(row) == ("succeeded", "ollama", 0)
        assert journal.capture_window_status(receipt)["acknowledged"] == 1
    assert journal.verify_all() == 0


@pytest.mark.parametrize("failure", ["cancelled", "failed", "deferred"])
def test_non_success_is_not_completed_coverage(tmp_path, failure):
    journal, _archive, receipt = window_fixture(tmp_path)
    journal.queue_capture_windows(receipt, limit=4)
    job = journal.claim_analysis(include_capture=True)
    if failure == "cancelled":
        assert journal.request_analysis_cancel(job.job_id)
        assert journal.fail_analysis(job.job_id, job.lease_token, "cancelled")
    elif failure == "deferred":
        assert journal.defer_analysis(job.job_id, job.lease_token, "gpu_busy")
    else:
        assert journal.fail_analysis(job.job_id, job.lease_token, "insufficient_context")
    assert journal.capture_window_status(receipt)["state"] != "completed"
    assert journal.capture_window_status(receipt)["acknowledged"] == 0
