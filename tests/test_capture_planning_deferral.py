"""Durable planner fairness, never analysis completion or parser resume proof."""
import pytest

from muninn.history.capture_journal import CaptureJournal
from muninn.history.credential_crypto import VaultIntegrityError
from tests.test_capture_enrichment import capture
from tests.test_capture_window_jobs import window_fixture


@pytest.mark.parametrize("code", ["io", "cancelled", "unsupported_source", "source_integrity"])
def test_deferred_source_yields_to_next_without_advancing_or_completing(tmp_path, monkeypatch, code):
    journal, archive, receipt = window_fixture(tmp_path)
    other = capture(archive, tmp_path / "other.jsonl", "Other source proceeds.")["snapshot_receipt"]
    journal.enqueue_enrichment_receipt(other)
    monkeypatch.setattr("muninn.history.capture_window_jobs.time.time", lambda: 100.0)
    ticket = journal.capture_planning_ticket(receipt)
    assert journal.defer_capture_plan(receipt, ticket, code)
    assert journal.next_capture_plan() == other
    state = journal.capture_window_status(receipt)
    assert state["state"] == "pending"
    assert state["planning"]["state"] == ("blocked" if code in {"unsupported_source", "source_integrity"} else "deferred")
    assert state["planning"]["reason"] == code
    assert journal.enrichment_status()["pending_sources"] == 2
    with journal._connect() as db:
        row = journal._capture_outbox_row(db, receipt)
        assert row["sealed_plan"] is None
        assert row["planning_complete"] == row["resolved"] == 0


def test_retry_boundary_and_success_clear_retry_without_stale_overwrite(tmp_path, monkeypatch):
    journal, archive, receipt = window_fixture(tmp_path)
    monkeypatch.setattr("muninn.history.capture_window_jobs.time.time", lambda: 100.0)
    stale = journal.capture_planning_ticket(receipt)
    assert journal.defer_capture_plan(receipt, stale, "io")
    assert journal.next_capture_plan() is None
    monkeypatch.setattr("muninn.history.capture_window_jobs.time.time", lambda: 104.999)
    assert journal.next_capture_plan() is None
    monkeypatch.setattr("muninn.history.capture_window_jobs.time.time", lambda: 105.0)
    assert journal.next_capture_plan() == receipt
    assert journal.queue_capture_windows(receipt)["queued"] > 0
    assert not journal.defer_capture_plan(receipt, stale, "source_integrity")
    assert "planning" not in journal.capture_window_status(receipt)


@pytest.mark.parametrize("damage", ["missing", "tampered"])
def test_established_planning_state_cannot_be_erased_or_repaired(tmp_path, damage):
    journal, archive, receipt = window_fixture(tmp_path)
    ticket = journal.capture_planning_ticket(receipt)
    assert journal.defer_capture_plan(receipt, ticket, "unsupported_source")
    with journal._connect() as db:
        db.execute("UPDATE capture_enrichment_sources SET sealed_planning_state="
                   + ("NULL" if damage == "missing" else "zeroblob(length(sealed_planning_state))"))
    for action in (journal.next_capture_plan, journal.verify_all,
                   lambda: journal.defer_capture_plan(receipt, ticket, "io")):
        with pytest.raises(VaultIntegrityError):
            action()
    with pytest.raises(VaultIntegrityError):
        CaptureJournal(archive).verify_all()


def test_transient_deferral_survives_portable_restore_and_stays_unresolved(tmp_path):
    from muninn.history.secure_archive import SecureHistoryArchive
    journal, archive, receipt = window_fixture(tmp_path)
    ticket = journal.capture_planning_ticket(receipt)
    assert journal.defer_capture_plan(receipt, ticket, "io")
    recovered = SecureHistoryArchive.restore_from_backup(
        archive.root, tmp_path / "restored", "test-only portable passphrase")
    restored = CaptureJournal(recovered)
    assert restored.capture_window_status(receipt) == journal.capture_window_status(receipt)
    assert restored.enrichment_status()["pending_sources"] == 1


def test_error_details_cannot_be_persisted_instead_of_sanitized_category(tmp_path):
    journal, archive, receipt = window_fixture(tmp_path)
    with pytest.raises(ValueError):
        journal.defer_capture_plan(receipt, journal.capture_planning_ticket(receipt),
                                   "private exception text TOKEN=CANARY-SECRET-129")


def legacy_state(journal):
    with journal._connect() as db:
        totals = journal._capture_schedule(db)
        db.execute("UPDATE capture_enrichment_sources SET sealed_planning_state=NULL")
        journal._write_capture_schedule(db, {**totals, "format": 1})


def test_authenticated_legacy_migration_initializes_all_states_atomically(tmp_path, monkeypatch):
    journal, archive, receipt = window_fixture(tmp_path)
    other = capture(archive, tmp_path / "other.jsonl", "Other source.")["snapshot_receipt"]
    journal.enqueue_enrichment_receipt(other)
    legacy_state(journal)
    original = CaptureJournal._new_capture_planning_state
    calls = 0
    def interrupted(self, ident):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("isolated interrupted migration")
        return original(self, ident)
    with monkeypatch.context() as patch:
        patch.setattr(CaptureJournal, "_new_capture_planning_state", interrupted)
        with pytest.raises(RuntimeError, match="interrupted migration"):
            CaptureJournal(archive, recover=False)
    with journal._connect() as db:
        assert journal._capture_schedule(db, allow_legacy=True)["format"] == 1
        assert db.execute("SELECT COUNT(*) FROM capture_enrichment_sources "
                          "WHERE sealed_planning_state IS NOT NULL").fetchone()[0] == 0
    migrated = CaptureJournal(archive, recover=False)
    assert migrated.next_capture_plan() == receipt
    assert migrated.verify_all() == 0
    with migrated._connect() as db:
        assert migrated._capture_schedule(db)["format"] == 2


def test_missing_established_schema_is_not_reinitialized_as_legacy(tmp_path):
    journal, archive, receipt = window_fixture(tmp_path)
    with journal._connect() as db:
        db.execute("ALTER TABLE capture_enrichment_sources DROP COLUMN sealed_planning_state")
    with pytest.raises(VaultIntegrityError, match="schema is missing"):
        CaptureJournal(archive, recover=False)


@pytest.mark.parametrize("change", [{"attempts": True}, {"code": []}, {"due_at": float("nan")},
                                    {"due_at": 0}, {"blocked": True}])
def test_invalid_authenticated_retry_state_fails_closed(tmp_path, change):
    journal, archive, receipt = window_fixture(tmp_path)
    ticket = journal.capture_planning_ticket(receipt)
    assert journal.defer_capture_plan(receipt, ticket, "io")
    with journal._connect() as db:
        row = journal._capture_outbox_row(db, receipt)
        state = {**journal._capture_planning_state(row), **change}
        db.execute("UPDATE capture_enrichment_sources SET sealed_planning_state=? WHERE work_id=?", (
            journal._seal_search(state, row["work_id"], "capture-planning-state-v1"), row["work_id"]))
    with pytest.raises(VaultIntegrityError):
        journal.next_capture_plan()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,code", [(OSError("private IO detail"), "io"),
                                         (RuntimeError("private unknown detail"), "preparation_error")])
async def test_worker_persists_failure_and_other_source_proceeds(tmp_path, monkeypatch, failure, code):
    from muninn.history.service import HistoryService
    journal, archive, receipt = window_fixture(tmp_path)
    other = capture(archive, tmp_path / "other.jsonl", "Other source proceeds.")["snapshot_receipt"]
    journal.enqueue_enrichment_receipt(other)
    service = HistoryService(None, tmp_path / "service", home=tmp_path)
    service._capture_journal = journal
    original = journal.queue_capture_windows
    def fail_first(source, **kwargs):
        if source == receipt:
            raise failure
        return original(source, **kwargs)
    monkeypatch.setattr(journal, "queue_capture_windows", fail_first)
    assert await service._process_capture_plan_once()
    assert journal.capture_window_status(receipt)["planning"]["reason"] == code
    assert await service._process_capture_plan_once()
    assert journal.capture_window_status(other)["queued"] > 0
    assert journal.capture_window_status(receipt)["state"] == "pending"


@pytest.mark.asyncio
async def test_worker_cancellation_drains_preparation_before_durable_deferral(tmp_path, monkeypatch):
    import asyncio
    import threading
    from muninn.history.service import HistoryService
    from muninn.history.structured_projector import ProjectionCancelled
    journal, archive, receipt = window_fixture(tmp_path)
    service = HistoryService(None, tmp_path / "service", home=tmp_path)
    service._capture_journal = journal
    started, finished = threading.Event(), threading.Event()
    def preparing(receipt, *, should_cancel, **kwargs):
        started.set()
        while not should_cancel():
            finished.wait(0.01)
        finished.set()
        raise ProjectionCancelled("isolated cancelled preparation")
    monkeypatch.setattr(journal, "queue_capture_windows", preparing)
    pending = asyncio.create_task(service._process_capture_plan_once())
    assert await asyncio.to_thread(started.wait, 2)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, 2)
    assert finished.is_set()
    assert journal.capture_window_status(receipt)["planning"]["reason"] == "cancelled"
    assert journal.enrichment_status()["pending_sources"] == 1
