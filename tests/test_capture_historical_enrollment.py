"""Explicit latest-only enrollment; temporary encrypted archives, no inference."""
import pytest

from muninn.history.capture_journal import CaptureJournal
from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.secure_archive import SecureHistoryArchive
from tests.test_capture_enrichment import capture
from tests.test_secure_capture_journal import _journal
from scripts.enroll_history_backlog import ReadOnlyJournal
from scripts import enroll_history_backlog
import json
import sqlite3
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def legacy_fixture(tmp_path, count=3):
    journal, archive = _journal(tmp_path)
    for index in range(count):
        path = tmp_path / f"old-session-{index}.jsonl"
        capture(archive, path, "First old observation.")
        capture(archive, path, f"Latest old observation {index}.")
    with archive._write_lock():
        manifest = archive._load_manifest()
        for entries in manifest["files"].values():
            for entry in entries:
                entry.pop("commit_generation", None)
        manifest["generation"] += 1
        archive._save_manifest(manifest)
    journal.configure_enrichment(archive._load_manifest()["generation"])
    return journal, archive


def test_explicit_latest_only_enrollment_preserves_live_tracking(tmp_path):
    journal, archive = legacy_fixture(tmp_path)
    with journal._connect() as db:
        baseline = journal._enrichment_baseline(db)
        live_cursor = journal._enrichment_progress(db, baseline)[0]
    latest = list(archive._load_manifest()["files"].values())[0][-1]
    assert journal.enqueue_enrichment_receipt(archive._snapshot_receipt(latest, 1)) == "before_watermark"
    first = journal.enroll_historical_latest(limit=2)
    assert first["queued"] == 2 and not first["complete"]
    second = CaptureJournal(archive).enroll_historical_latest(limit=2)
    assert second["queued"] == 3 and second["complete"]
    assert journal.enroll_historical_latest(limit=2) == second
    receipts = journal.pending_enrichment()
    assert len(receipts) == 3 and {r["version"] for r in receipts} == {1}
    assert journal.queue_capture_windows(receipts[0], limit=1)["queued"] == 1
    assert journal.enrichment_status()["historical_enrollment"]["complete"]
    with journal._connect() as db:
        assert journal._enrichment_baseline(db) == baseline
        assert journal._enrichment_progress(db, baseline)[0] == live_cursor
    assert journal.verify_all() == 0


def test_pinned_selection_and_new_capture_have_independent_cursors(tmp_path):
    journal, archive = legacy_fixture(tmp_path)
    first = journal.enroll_historical_latest(limit=1)
    new = capture(archive, tmp_path / "new-session.jsonl", "Current observation.")
    capture(archive, tmp_path / "old-session-2.jsonl", "Changed after historical pin.")
    final = journal.enroll_historical_latest(limit=128)
    assert final["generation"] == first["generation"]
    assert final["queued"] == final["total_sources"] == 3
    assert journal.reconcile_enrichment() == 2
    assert new["snapshot_receipt"] in journal.pending_enrichment()
    assert len(journal.pending_enrichment()) == 5
    assert journal.verify_all() == 0


def test_insert_and_cursor_roll_back_together(tmp_path, monkeypatch):
    journal, archive = legacy_fixture(tmp_path)
    original = journal._store_enrichment_receipt
    calls = 0
    def interrupted(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 2:
            raise RuntimeError("isolated interrupted batch")
        return result
    monkeypatch.setattr(journal, "_store_enrichment_receipt", interrupted)
    with pytest.raises(RuntimeError):
        journal.enroll_historical_latest(limit=3)
    assert journal.pending_enrichment() == []
    monkeypatch.setattr(journal, "_store_enrichment_receipt", original)
    assert CaptureJournal(archive).enroll_historical_latest(limit=3)["queued"] == 3
    assert journal.verify_all() == 0


def test_stale_writer_cannot_regress_cursor(tmp_path, monkeypatch):
    journal, _archive = legacy_fixture(tmp_path)
    journal.enroll_historical_latest(limit=1)
    original = journal._historical_manifest
    raced = False
    def race(cursor):
        nonlocal raced
        manifest = original(cursor)
        if not raced:
            raced = True
            journal.enroll_historical_latest(limit=1)
        return manifest
    monkeypatch.setattr(journal, "_historical_manifest", race)
    assert journal.enroll_historical_latest(limit=1)["source_index"] == 2
    assert journal.enroll_historical_latest(limit=1)["queued"] == 3
    assert len(journal.pending_enrichment()) == 3
    assert journal.verify_all() == 0


@pytest.mark.parametrize("damage", ["cursor", "grant", "missing_manifest", "orphan"])
def test_historical_evidence_damage_blocks_verification_and_restore(tmp_path, damage):
    journal, archive = legacy_fixture(tmp_path)
    state = journal.enroll_historical_latest(limit=1)
    capture(archive, tmp_path / "later.jsonl", "Later current snapshot.")
    with journal._connect() as db:
        if damage == "cursor":
            db.execute("UPDATE capture_historical_enrollment SET sealed_cursor=zeroblob(length(sealed_cursor))")
        elif damage == "grant":
            db.execute("UPDATE capture_historical_receipts SET sealed_grant=zeroblob(length(sealed_grant))")
        elif damage == "orphan":
            db.execute("DELETE FROM capture_enrichment_sources")
    if damage == "missing_manifest":
        (archive.root / f"manifest-{state['generation']:012d}.enc").unlink()
    with pytest.raises(VaultIntegrityError):
        journal.verify_all()
    with pytest.raises(VaultIntegrityError):
        SecureHistoryArchive.restore_from_backup(
            archive.root, tmp_path / "restored", "test-only portable passphrase")


def test_historical_enrollment_survives_portable_restore(tmp_path):
    journal, archive = legacy_fixture(tmp_path)
    state = journal.enroll_historical_latest(limit=1)
    restored = SecureHistoryArchive.restore_from_backup(
        archive.root, tmp_path / "restored", "test-only portable passphrase")
    other = CaptureJournal(restored)
    assert other.historical_enrollment_status() == state
    assert other.enroll_historical_latest(limit=128)["queued"] == 3
    assert other.verify_all() == 0


def test_preview_does_not_migrate_or_write_journal(tmp_path):
    journal, archive = legacy_fixture(tmp_path)
    with journal._connect() as db:
        db.execute("DROP TABLE capture_historical_enrollment")
        db.execute("DROP TABLE capture_historical_receipts")
    before = journal.path.read_bytes()
    preview = ReadOnlyJournal(archive).preview_historical_latest(limit=2)
    assert preview["would_queue"] == 2 and preview["batch_sources"] == 2
    assert before == journal.path.read_bytes()


def test_modern_prebaseline_receipt_uses_explicit_grant(tmp_path):
    journal, archive = _journal(tmp_path)
    receipt = capture(archive, tmp_path / "old.jsonl", "Old modern-format snapshot.")["snapshot_receipt"]
    journal.configure_enrichment(archive._load_manifest()["generation"])
    assert journal.enqueue_enrichment_receipt(receipt) == "before_watermark"
    assert journal.enroll_historical_latest()["queued"] == 1
    assert journal.pending_enrichment() == [receipt]
    assert journal.enqueue_enrichment_receipt(receipt) == "before_watermark"
    assert journal.queue_capture_windows(receipt, limit=1)["queued"] == 1
    assert journal.verify_all() == 0


def test_completed_normal_receipt_is_counted_existing_without_duplicate(tmp_path):
    journal, archive = _journal(tmp_path)
    journal.configure_enrichment(0)
    receipt = capture(archive, tmp_path / "current.jsonl", "Current version.")["snapshot_receipt"]
    assert journal.enqueue_enrichment_receipt(receipt) == "queued"
    state = journal.enroll_historical_latest()
    assert state["existing"] == 1 and state["queued"] == 0 and state["complete"]
    assert journal.pending_enrichment() == [receipt]
    assert journal.verify_all() == 0


def test_read_only_adapter_refuses_writes(tmp_path):
    _journal_, archive = legacy_fixture(tmp_path)
    with ReadOnlyJournal(archive)._connect() as db:
        with pytest.raises(sqlite3.OperationalError):
            db.execute("CREATE TABLE forbidden_preview_write (id INTEGER)")


def test_cli_preview_and_bounded_apply_emit_only_metadata(tmp_path, monkeypatch, capsys):
    journal, archive = legacy_fixture(tmp_path)
    monkeypatch.setattr(enroll_history_backlog, "SecureHistoryArchive", lambda *_args: archive)
    args = ["--archive-root", str(archive.root), "--limit", "1"]
    before = journal.path.read_bytes()
    assert enroll_history_backlog.main(args) == 0
    output = capsys.readouterr().out
    assert json.loads(output)["would_queue"] == 1
    assert str(tmp_path) not in output and "Latest old observation" not in output
    assert journal.path.read_bytes() == before
    assert enroll_history_backlog.main([*args, "--apply", "--max-batches", "2"]) == 0
    output = capsys.readouterr().out
    result = json.loads(output.splitlines()[-1])
    assert result == {"stage": "enrollment_verified", "complete": False, "processed_by_models": False}
    assert len(journal.pending_enrichment()) == 2
    assert str(tmp_path) not in output and "Latest old observation" not in output


@pytest.mark.parametrize("damage", ["missing", "changed", "reordered"])
def test_ordinary_reader_rechecks_pinned_bytes_after_cache_warm(tmp_path, damage):
    journal, archive = legacy_fixture(tmp_path)
    state = journal.enroll_historical_latest(limit=1)
    assert len(journal.pending_enrichment()) == 1  # Warm proof-identity cache.
    capture(archive, tmp_path / "later.jsonl", "Current capture.")
    pin = archive.root / f"manifest-{state['generation']:012d}.enc"
    if damage == "missing":
        pin.unlink()
    elif damage == "changed":
        raw = pin.read_bytes()
        pin.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    else:
        manifest = archive._load_manifest(generation=state["generation"])
        manifest["files"] = dict(reversed(list(manifest["files"].items())))
        # Authenticated same-generation replacement is still not the enrolled pin.
        nonce = os.urandom(12)
        raw = json.dumps(manifest, separators=(",", ":")).encode()
        pin.write_bytes(nonce + AESGCM(archive._key).encrypt(
            nonce, raw, archive._manifest_aad(state["generation"])))
    with pytest.raises(VaultIntegrityError):
        journal.pending_enrichment()


def test_pin_proof_reuse_still_hashes_without_reparsing_catalog(tmp_path, monkeypatch):
    journal, archive = legacy_fixture(tmp_path)
    journal.enroll_historical_latest(limit=1)
    calls = []
    original = archive._load_manifest
    def observed(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(archive, "_load_manifest", observed)
    assert len(journal.pending_enrichment()) == 1
    assert len(journal.pending_enrichment()) == 1
    assert calls == []  # Immutable cryptographic proof reused, not a stat/time hint.
