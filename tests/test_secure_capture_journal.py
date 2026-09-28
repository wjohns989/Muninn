"""Durable, private hook capture jobs independent of model and live service."""

import os
import sqlite3
from pathlib import Path

import pytest

from muninn.history.capture_journal import CaptureJournal
from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.secure_archive import SecureHistoryArchive


def _journal(tmp_path: Path) -> tuple[CaptureJournal, SecureHistoryArchive]:
    archive = SecureHistoryArchive.create(tmp_path / "encrypted", "test-only portable passphrase")
    return CaptureJournal(archive), archive


def test_enqueue_survives_restart_without_plaintext_locator(tmp_path):
    journal, archive = _journal(tmp_path)
    source = tmp_path / "private-project" / "session.jsonl"
    source.parent.mkdir()
    source.write_text("private transcript marker", encoding="utf-8")

    journal.enqueue(source, "codex", force=True)
    reopened = CaptureJournal(archive)
    job = reopened.claim_due()

    assert job is not None
    assert job.path == source.resolve()
    assert job.provider == "codex"
    raw = (archive.root / "capture-jobs.db").read_bytes()
    assert b"private-project" not in raw
    assert b"private transcript marker" not in raw


def test_claimed_job_requeues_on_restart_and_newer_revision_wins(tmp_path):
    journal, archive = _journal(tmp_path)
    source = tmp_path / "session.jsonl"
    source.write_text("first", encoding="utf-8")
    journal.enqueue(source, "codex", force=True)
    first = journal.claim_due()
    assert first is not None

    restarted = CaptureJournal(archive)
    replay = restarted.claim_due()
    assert replay is not None and replay.revision == first.revision
    source.write_text("first then second", encoding="utf-8")
    restarted.enqueue(source, "codex", force=True)
    assert restarted.finish(replay, archived=True) is False
    newest = restarted.claim_due()
    assert newest is not None and newest.revision > replay.revision


def test_archive_commit_before_journal_finish_replays_without_duplicate_snapshot(tmp_path):
    journal, archive = _journal(tmp_path)
    source = tmp_path / "session.jsonl"
    source.write_text("committed snapshot", encoding="utf-8")
    journal.enqueue(source, "codex", force=True)
    claimed = journal.claim_due()
    assert claimed is not None
    assert archive.archive_file(source, "codex")["status"] == "captured"

    restarted = CaptureJournal(archive)
    replay = restarted.claim_due()
    assert replay is not None and replay.revision == claimed.revision
    assert archive.archive_file(source, "codex")["status"] == "unchanged"
    assert restarted.finish(replay, archived=True) is True
    assert archive.status()["snapshots"] == 1


def test_deleted_source_remains_retryable_without_false_completion(tmp_path):
    journal, archive = _journal(tmp_path)
    source = tmp_path / "session.jsonl"
    source.write_text("disappearing source", encoding="utf-8")
    journal.enqueue(source, "codex", force=True)
    job = journal.claim_due()
    assert job is not None
    source.unlink()
    journal.fail(job, "missing")

    assert journal.status()["retry"] == 1
    assert archive.status()["snapshots"] == 0


def test_interrupted_discovery_resumes_generation_without_repeating_checkpoint(tmp_path):
    journal, archive = _journal(tmp_path)
    generation = journal.begin_scan()
    key = journal.source_key(tmp_path / "session.jsonl", "codex")
    journal.record_scan_batch(generation, [(key, "unchanged")])

    restarted = CaptureJournal(archive)
    assert restarted.begin_scan() == generation
    assert restarted.scan_seen(key, generation) is True
    report = restarted.finish_scan(generation)
    assert report["seen"] == 1 and report["unchanged"] == 1
    assert restarted.begin_scan() == generation + 1
    assert restarted.scan_seen(key, generation + 1) is False


def test_scan_replay_after_enqueue_before_checkpoint_does_not_advance_revision(tmp_path):
    journal, _archive = _journal(tmp_path)
    source = tmp_path / "session.jsonl"
    source.write_text("unchanged since first scan attempt", encoding="utf-8")
    generation = journal.begin_scan()
    assert journal.enqueue(source, "codex", immediate=True) == "queued"
    # Simulate a crash before the scan_seen checkpoint transaction.
    assert journal.begin_scan() == generation
    assert journal.enqueue(source, "codex", immediate=True) == "coalesced"
    job = journal.claim_due()
    assert job is not None and job.revision == 1


@pytest.mark.skipif(os.name != "nt", reason="Unattended ciphertext backup uses Windows user protection")
def test_portable_archive_backup_preserves_pending_capture_job(tmp_path):
    journal, archive = _journal(tmp_path)
    source = tmp_path / "session.jsonl"
    source.write_text("recoverable queue", encoding="utf-8")
    journal.enqueue(source, "codex", force=True)

    backup = tmp_path / "backup"
    archive.backup_to(backup)
    restored = SecureHistoryArchive.restore_from_backup(
        backup, tmp_path / "restored", "test-only portable passphrase"
    )
    job = CaptureJournal(restored).claim_due()

    assert job is not None and job.path == source.resolve()
    assert b"session.jsonl" not in (backup / "capture-jobs.db").read_bytes()


def test_modified_sealed_locator_fails_authentication_without_path_output(tmp_path):
    journal, _archive = _journal(tmp_path)
    source = tmp_path / "private-session.jsonl"
    source.write_text("private", encoding="utf-8")
    journal.enqueue(source, "codex", force=True)
    with sqlite3.connect(journal.path) as db:
        sealed = db.execute("SELECT sealed_locator FROM jobs").fetchone()[0]
        changed = bytes([sealed[0] ^ 1]) + sealed[1:]
        db.execute("UPDATE jobs SET sealed_locator=?", (changed,))

    with pytest.raises(VaultIntegrityError, match="authentication failed") as caught:
        journal.verify_all()
    assert "private-session" not in str(caught.value)
