"""Capture-to-outbox wiring with isolated archives, no models or live service."""
import sqlite3
from unittest.mock import Mock

import pytest

from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.service import HistoryService


def setup_service(monkeypatch, tmp_path, *, enabled=True):
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    monkeypatch.setenv("MUNINN_CAPTURE_ENRICHMENT", "1" if enabled else "0")
    for name in ("MUNINN_HISTORY_HOMES", "CODEX_HOME", "CLAUDE_CONFIG_DIR"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "encrypted"
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(root))
    archive = SecureHistoryArchive.create(root, "test-only portable passphrase")
    source = (tmp_path / ".codex" / "sessions" / "2026" /
              "rollout-2026-09-28T01-02-03-11111111-1111-4111-8111-111111111111.jsonl")
    source.parent.mkdir(parents=True)
    service = HistoryService(Mock(), tmp_path / "unused", home=tmp_path,
                             archive_passphrase="test-only portable passphrase")
    return service, archive, source


@pytest.mark.asyncio
async def test_enabled_capture_queues_exact_receipt_without_exposing_it(monkeypatch, tmp_path):
    service, archive, source = setup_service(monkeypatch, tmp_path)
    source.write_text("New chat content.", encoding="utf-8")
    outcome = await service.capture(str(source), "codex")
    assert outcome["captured"] and "snapshot_receipt" not in outcome["archive"]
    journal = service._require_capture_journal()
    assert len(journal.pending_enrichment()) == 1
    assert journal.pending_enrichment()[0]["commit_generation"] == archive._load_manifest()["generation"]
    await service.capture(str(source), "codex")
    assert len(journal.pending_enrichment()) == 1
    assert service.status()["capture_enrichment"]["capture_enabled"] is True
    service.memory.add.assert_not_called()


@pytest.mark.asyncio
async def test_queue_lock_failure_and_source_growth_are_recovered_by_scan(monkeypatch, tmp_path, caplog):
    service, archive, source = setup_service(monkeypatch, tmp_path)
    journal = service._require_capture_journal()
    source.write_text("Baseline content.", encoding="utf-8")
    archive.archive_file(source, "codex")
    await service.scan_capture_sources()  # Enable before the next commit, exclude baseline.

    def interrupted(_receipt):
        raise sqlite3.OperationalError("Private path must not enter logs")

    monkeypatch.setattr(journal, "enqueue_enrichment_receipt", interrupted)
    source.write_text("Saved while outbox insertion is blocked.", encoding="utf-8")
    assert (await service.capture(str(source), "codex"))["captured"]
    assert journal.pending_enrichment() == []
    source.write_text("Saved newer version before reconciliation.", encoding="utf-8")
    assert (await service.capture(str(source), "codex"))["captured"]
    assert archive.status()["snapshots"] == 3
    assert "Private path" not in caplog.text

    restarted = HistoryService(Mock(), tmp_path / "unused", home=tmp_path,
                               archive_passphrase="test-only portable passphrase")
    await restarted.scan_capture_sources()
    assert {r["version"] for r in restarted._capture_journal.pending_enrichment()} == {1, 2}
    assert restarted.status()["capture_enrichment"]["pending_sources"] == 2


@pytest.mark.asyncio
async def test_disabled_capture_preserves_existing_contract_and_leaves_outbox_unconfigured(monkeypatch, tmp_path):
    service, archive, source = setup_service(monkeypatch, tmp_path, enabled=False)
    source.write_text("CPU-only saved chat.", encoding="utf-8")
    assert (await service.capture(str(source), "codex"))["captured"]
    await service.scan_capture_sources()
    assert service._capture_journal.enrichment_status() == {
        "configured": False, "pending_sources": 0, "parked_private_windows": 0,
        "window_jobs": {"basis": "all_capture_lane_jobs", "total": 0, "states": {}}}
    assert service.status()["capture_enrichment"]["capture_enabled"] is False
    assert archive.status()["snapshots"] == 1


def test_enabling_baseline_holds_archive_writer_lock(monkeypatch, tmp_path):
    service, archive, _source = setup_service(monkeypatch, tmp_path)
    journal = service._require_capture_journal()
    original = journal.configure_enrichment
    observed = []
    from muninn.history.secure_archive import _thread_lock

    def under_lock(generation):
        assert _thread_lock._is_owned()
        observed.append(generation)
        return original(generation)

    monkeypatch.setattr(journal, "configure_enrichment", under_lock)
    assert service._configure_capture_enrichment() is True
    assert observed == [archive._load_manifest()["generation"]]
    assert service._configure_capture_enrichment() is True
    assert len(observed) == 1


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows archive lock regression")
def test_restart_reuses_authenticated_watermark_while_backup_holds_archive_lock(monkeypatch, tmp_path):
    import subprocess
    import sys

    service, archive, _source = setup_service(monkeypatch, tmp_path)
    assert service._configure_capture_enrichment()
    journal = service._require_capture_journal()
    with journal._connect() as db:
        before = (db.execute("SELECT sealed_config FROM capture_enrichment_control").fetchone()[0],
                  db.execute("SELECT sealed_cursor FROM capture_enrichment_progress").fetchone()[0])
    restarted = HistoryService(Mock(), tmp_path / "unused", home=tmp_path,
                               archive_passphrase="test-only portable passphrase")
    # Use a separate process: same-process thread-lock reentrancy must not hide
    # the actual Windows file-lock contention observed on the installed host.
    child = subprocess.Popen([sys.executable, "-c",
        "import msvcrt,sys; f=open(sys.argv[1],'r+b'); "
        "msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1); "
        "print('locked',flush=True); sys.stdin.readline(); "
        "f.seek(0); msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)",
        str(archive._lock_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "locked"
        assert restarted._configure_capture_enrichment()
        with journal._connect() as db:
            after = (db.execute("SELECT sealed_config FROM capture_enrichment_control").fetchone()[0],
                     db.execute("SELECT sealed_cursor FROM capture_enrichment_progress").fetchone()[0])
        assert before == after
    finally:
        child.communicate(input="release\n", timeout=10)
    assert child.returncode == 0


def test_existing_watermark_integrity_failure_is_not_treated_as_lock_contention(monkeypatch, tmp_path):
    from muninn.history.credential_crypto import VaultIntegrityError

    service, archive, _source = setup_service(monkeypatch, tmp_path)
    assert service._configure_capture_enrichment()
    journal = service._require_capture_journal()
    with journal._connect() as db:
        db.execute("UPDATE capture_enrichment_control SET sealed_config=?", (b"not an authenticated baseline",))
    restarted = HistoryService(Mock(), tmp_path / "unused", home=tmp_path,
                               archive_passphrase="test-only portable passphrase")
    with pytest.raises(VaultIntegrityError):
        restarted._configure_capture_enrichment()
    assert not restarted._capture_enrichment_configured


@pytest.mark.parametrize("fault", ["progress_seal", "progress_missing", "rollback", "cursor",
                                  "baseline_missing"])
def test_restart_configuration_fails_closed_on_invalid_progress(monkeypatch, tmp_path, fault):
    from muninn.history.credential_crypto import VaultIntegrityError

    service, archive, source = setup_service(monkeypatch, tmp_path)
    assert service._configure_capture_enrichment()
    journal = service._require_capture_journal()
    source.write_text("Synthetic local fixture", encoding="utf-8")
    archive.archive_file(source, "codex")
    with journal._connect() as db:
        baseline = journal._enrichment_baseline(db)
        generation = archive._load_manifest()["generation"]
        if fault == "progress_seal":
            db.execute("UPDATE capture_enrichment_progress SET sealed_cursor=?", (b"invalid",))
        elif fault == "progress_missing":
            db.execute("DELETE FROM capture_enrichment_progress")
        elif fault == "baseline_missing":
            db.execute("DELETE FROM capture_enrichment_control")
        else:
            cursor = {"format": 1, "after_generation": baseline,
                      "through_generation": generation + 1 if fault == "rollback" else generation,
                      "source_index": 999 if fault == "cursor" else 0, "version_index": 0}
            db.execute("UPDATE capture_enrichment_progress SET sealed_cursor=?", (
                journal._seal_search(cursor, "0" * 32, "capture-enrichment-progress-v1"),))
    restarted = HistoryService(Mock(), tmp_path / "unused", home=tmp_path,
                               archive_passphrase="test-only portable passphrase")
    with pytest.raises(VaultIntegrityError):
        restarted._configure_capture_enrichment()
    assert not restarted._capture_enrichment_configured
