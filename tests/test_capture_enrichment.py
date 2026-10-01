"""Durable post-enable capture outbox; isolated archives and no inference."""
import json

import pytest

from muninn.history.capture_journal import CaptureJournal
from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.secure_archive import SecureHistoryArchive
from tests.test_secure_capture_journal import _journal


def capture(archive, path, text):
    path.write_text(json.dumps({"type": "event_msg", "payload": {
        "type": "user_message", "message": text}}) + "\n", encoding="utf-8")
    return archive.archive_file(path, "codex", include_snapshot_receipt=True)


def test_crash_then_source_change_recovers_each_commit_without_historical_backfill(tmp_path):
    journal, archive = _journal(tmp_path)
    path = tmp_path / "private-project-session.jsonl"
    capture(archive, path, "Old baseline history.")
    baseline = archive._load_manifest()["generation"]
    assert journal.configure_enrichment(baseline) == baseline
    first = capture(archive, path, "Saved before interrupted outbox insertion.")
    # No enqueue: simulate exit after archive commit, then change the source.
    second = capture(archive, path, "A newer saved revision.")
    reopened = CaptureJournal(archive)
    assert reopened.reconcile_enrichment() == 2
    receipts = reopened.pending_enrichment()
    assert {r["version"] for r in receipts} == {1, 2}
    assert {r["blob"] for r in receipts} == {
        first["snapshot_receipt"]["blob"], second["snapshot_receipt"]["blob"]}
    assert reopened.reconcile_enrichment() == 0
    assert reopened.configure_enrichment(archive._load_manifest()["generation"]) == baseline
    assert reopened.verify_all() == 0
    assert b"private-project" not in reopened.path.read_bytes()
    assert b"Saved before" not in reopened.path.read_bytes()


def test_commit_receipt_keeps_exact_older_identity_after_new_capture(tmp_path):
    journal, archive = _journal(tmp_path)
    journal.configure_enrichment(0)
    path = tmp_path / "session.jsonl"
    first = capture(archive, path, "First committed version.")
    second = capture(archive, path, "Second committed version.")
    assert first["snapshot_receipt"]["version"] == 0
    assert second["snapshot_receipt"]["version"] == 1
    assert journal.enqueue_enrichment_receipt(first["snapshot_receipt"]) == "queued"
    assert journal.enqueue_enrichment_receipt(first["snapshot_receipt"]) == "existing"
    assert journal.pending_enrichment() == [first["snapshot_receipt"]]
    assert journal.reconcile_enrichment() == 1


def test_reconciliation_is_bounded_and_resumable(tmp_path):
    journal, archive = _journal(tmp_path)
    journal.configure_enrichment(0)
    for number in range(5):
        capture(archive, tmp_path / f"session-{number}.jsonl", f"Distinct observation {number}.")
    assert journal.reconcile_enrichment(limit=2) == 2
    assert journal.reconcile_enrichment(limit=2) == 2
    assert journal.reconcile_enrichment(limit=2) == 1
    assert journal.reconcile_enrichment(limit=2) == 0
    assert len(journal.pending_enrichment(limit=2)) == 2
    assert journal.enrichment_status() == {"configured": True, "pending_sources": 5}


def test_checkpoint_resumes_without_revisiting_completed_receipts_and_finds_new_earlier_path(tmp_path, monkeypatch):
    journal, archive = _journal(tmp_path)
    journal.configure_enrichment(0)
    for number in range(7):
        capture(archive, tmp_path / f"z-session-{number}.jsonl", f"Observation {number}.")
    visited = []
    original = archive._snapshot_receipt

    def counted(entry, version):
        visited.append(entry["blob"])
        return original(entry, version)

    monkeypatch.setattr(archive, "_snapshot_receipt", counted)
    assert journal.reconcile_enrichment(limit=2) == 2
    earlier = capture(archive, tmp_path / "a-session.jsonl", "Arrives during pinned traversal.")
    visited.clear()  # Exclude the capture's own receipt creation.
    restarted = CaptureJournal(archive)
    for expected in (2, 2, 1):
        assert restarted.reconcile_enrichment(limit=2) == expected
    assert len(visited) == len(set(visited)) == 5
    assert len(restarted.pending_enrichment()) == 7
    # A newer snapshot has a new path sorting before the previous checkpoint.
    assert restarted.reconcile_enrichment(limit=2) == 1
    assert earlier["snapshot_receipt"] in restarted.pending_enrichment()
    assert restarted.verify_all() == 0
    assert b"a-session" not in restarted.path.read_bytes()


def test_entry_bound_includes_legacy_and_excluded_records(tmp_path):
    journal, archive = _journal(tmp_path)
    for number in range(4):
        capture(archive, tmp_path / f"a-old-{number}.jsonl", "Pre-enable observation.")
    journal.configure_enrichment(archive._load_manifest()["generation"])
    new = capture(archive, tmp_path / "z-new.jsonl", "New observation.")["snapshot_receipt"]
    assert journal.reconcile_enrichment(limit=2) == 0
    assert journal.reconcile_enrichment(limit=2) == 0
    assert journal.reconcile_enrichment(limit=2) == 1
    assert journal.pending_enrichment() == [new]


def test_receipt_and_checkpoint_rollback_together(tmp_path, monkeypatch):
    journal, archive = _journal(tmp_path)
    journal.configure_enrichment(0)
    for number in range(3):
        capture(archive, tmp_path / f"session-{number}.jsonl", "Observation.")
    original = journal._store_enrichment_receipt
    calls = 0

    def interrupted(db, receipt, baseline):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Simulated interruption")
        return original(db, receipt, baseline)

    monkeypatch.setattr(journal, "_store_enrichment_receipt", interrupted)
    with pytest.raises(RuntimeError):
        journal.reconcile_enrichment(limit=2)
    assert journal.pending_enrichment() == []
    restarted = CaptureJournal(archive)
    assert restarted.reconcile_enrichment(limit=2) == 2
    assert restarted.reconcile_enrichment(limit=2) == 1


def test_concurrent_reconciler_cannot_regress_committed_checkpoint(tmp_path, monkeypatch):
    journal, archive = _journal(tmp_path)
    journal.configure_enrichment(0)
    for number in range(4):
        capture(archive, tmp_path / f"session-{number}.jsonl", "Observation.")
    original = journal._validate_cursor_position
    raced = False

    def race(cursor, manifest):
        nonlocal raced
        keys = original(cursor, manifest)
        if not raced:
            raced = True
            assert journal.reconcile_enrichment(limit=3) == 3
        return keys

    monkeypatch.setattr(journal, "_validate_cursor_position", race)
    assert journal.reconcile_enrichment(limit=2) == 0
    assert journal.reconcile_enrichment(limit=2) == 1
    assert len(journal.pending_enrichment()) == 4


@pytest.mark.parametrize("damage", ["cursor", "manifest"])
def test_pinned_checkpoint_integrity_failure_blocks_reconciliation_and_restore(tmp_path, damage):
    journal, archive = _journal(tmp_path)
    journal.configure_enrichment(0)
    for number in range(3):
        capture(archive, tmp_path / f"session-{number}.jsonl", "Observation.")
    assert journal.reconcile_enrichment(limit=1) == 1
    pinned = archive._load_manifest()["generation"]
    capture(archive, tmp_path / "newer.jsonl", "Later snapshot remains valid.")
    if damage == "cursor":
        with journal._connect() as db:
            db.execute("UPDATE capture_enrichment_progress SET sealed_cursor=zeroblob(length(sealed_cursor))")
    else:
        target = archive.root / f"manifest-{pinned:012d}.enc"
        raw = target.read_bytes()
        target.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(VaultIntegrityError):
        journal.reconcile_enrichment(limit=1)
    with pytest.raises(VaultIntegrityError):
        journal.verify_all()
    with pytest.raises(VaultIntegrityError):
        SecureHistoryArchive.restore_from_backup(
            archive.root, tmp_path / "restored", "test-only portable passphrase")


def test_disabled_outbox_and_default_archive_result_do_not_change_capture_contract(tmp_path):
    journal, archive = _journal(tmp_path)
    path = tmp_path / "session.jsonl"
    path.write_text("ordinary text", encoding="utf-8")
    result = archive.archive_file(path, "codex")
    assert set(result) == {"status", "versions", "size"}
    assert archive.archive_file(path, "codex") == {"status": "unchanged", "versions": 1}
    assert journal.reconcile_enrichment() == 0
    assert journal.enrichment_status() == {"configured": False, "pending_sources": 0}


def test_legacy_unchanged_receipt_is_not_reclassified_as_new_work(tmp_path):
    journal, archive = _journal(tmp_path)
    path = tmp_path / "session.jsonl"
    capture(archive, path, "Existing legacy snapshot.")
    with archive._write_lock():
        manifest = archive._load_manifest()
        del manifest["files"][str(path.resolve())][0]["commit_generation"]
        manifest["generation"] += 1
        archive._save_manifest(manifest)
    journal.configure_enrichment(archive._load_manifest()["generation"])
    result = archive.archive_file(path, "codex", include_snapshot_receipt=True)
    assert result["snapshot_receipt"]["commit_generation"] is None
    assert journal.enqueue_enrichment_receipt(result["snapshot_receipt"]) == "before_watermark"
    assert journal.reconcile_enrichment() == 0


def test_batched_archive_commits_share_exact_committed_generation(tmp_path):
    journal, archive = _journal(tmp_path)
    initial_generation = archive._load_manifest()["generation"]
    journal.configure_enrichment(0)
    paths = []
    for number in range(3):
        path = tmp_path / f"batch-{number}.jsonl"
        path.write_text("source text", encoding="utf-8")
        paths.append((path, "codex", "transcript"))
    assert archive.archive_many(paths, commit_every=2)["commits"] == 2
    receipts = list(archive.iter_committed_receipts(after_generation=0))
    assert sorted(r["commit_generation"] for r in receipts) == [
        initial_generation + 1, initial_generation + 1, initial_generation + 2]
    assert journal.reconcile_enrichment() == 3


def test_receipt_tamper_fail_closed(tmp_path):
    journal, archive = _journal(tmp_path)
    journal.configure_enrichment(0)
    receipt = capture(archive, tmp_path / "session.jsonl", "Portable observation.")["snapshot_receipt"]
    journal.enqueue_enrichment_receipt(receipt)
    with journal._connect() as db:
        db.execute("UPDATE capture_enrichment_sources SET sealed_receipt=zeroblob(length(sealed_receipt))")
    with pytest.raises(VaultIntegrityError):
        journal.pending_enrichment()
    with pytest.raises(VaultIntegrityError):
        journal.verify_all()


def test_control_tamper_cannot_silently_reset_watermark(tmp_path):
    journal, _archive = _journal(tmp_path)
    journal.configure_enrichment(0)
    with journal._connect() as db:
        db.execute("UPDATE capture_enrichment_control SET sealed_config=zeroblob(length(sealed_config))")
    for action in (lambda: journal.configure_enrichment(5), journal.enrichment_status,
                   journal.reconcile_enrichment, journal.verify_all):
        with pytest.raises(VaultIntegrityError):
            action()


def test_unchanged_new_receipt_keeps_original_commit_generation(tmp_path):
    journal, archive = _journal(tmp_path)
    journal.configure_enrichment(archive._load_manifest()["generation"])
    path = tmp_path / "session.jsonl"
    first = capture(archive, path, "Unchanged new observation.")["snapshot_receipt"]
    capture(archive, tmp_path / "other.jsonl", "Unrelated commit advances manifest.")
    repeated = archive.archive_file(path, "codex", include_snapshot_receipt=True)
    assert repeated["status"] == "unchanged"
    assert repeated["snapshot_receipt"] == first
    assert journal.enqueue_enrichment_receipt(first) == "queued"
    assert journal.enqueue_enrichment_receipt(repeated["snapshot_receipt"]) == "existing"


def test_pending_outbox_survives_portable_restore(tmp_path):
    journal, archive = _journal(tmp_path)
    journal.configure_enrichment(0)
    receipt = capture(archive, tmp_path / "session.jsonl", "Portable recovery observation.")["snapshot_receipt"]
    journal.enqueue_enrichment_receipt(receipt)
    restored = SecureHistoryArchive.restore_from_backup(
        archive.root, tmp_path / "restored", "test-only portable passphrase")
    reopened = CaptureJournal(restored)
    assert reopened.pending_enrichment() == [receipt]
    assert reopened.reconcile_enrichment() == 0


def test_partial_checkpoint_survives_portable_restore(tmp_path):
    journal, archive = _journal(tmp_path)
    journal.configure_enrichment(0)
    for number in range(5):
        capture(archive, tmp_path / f"session-{number}.jsonl", f"Portable observation {number}.")
    assert journal.reconcile_enrichment(limit=2) == 2
    restored = SecureHistoryArchive.restore_from_backup(
        archive.root, tmp_path / "restored", "test-only portable passphrase")
    reopened = CaptureJournal(restored)
    assert reopened.reconcile_enrichment(limit=2) == 2
    assert reopened.reconcile_enrichment(limit=2) == 1
    assert len(reopened.pending_enrichment()) == 5


@pytest.mark.parametrize("limit", [0, -1, True, 129])
def test_invalid_reconcile_batch_bound_is_rejected(tmp_path, limit):
    journal, _archive = _journal(tmp_path)
    with pytest.raises(ValueError):
        journal.reconcile_enrichment(limit=limit)
