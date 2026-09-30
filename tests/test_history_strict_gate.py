"""Strict history mode must not touch the legacy plaintext import pipeline."""

import asyncio
import os
import sqlite3
import threading
from unittest.mock import AsyncMock, Mock

import pytest

from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.service import HistoryService


def test_history_defaults_to_strict_and_legacy_needs_two_opt_ins(monkeypatch, tmp_path):
    from muninn.history.vault import HistoryVault

    monkeypatch.delenv("MUNINN_HISTORY_SECURITY", raising=False)
    service = HistoryService(Mock(), tmp_path / "legacy-vault", home=tmp_path)
    assert service.vault is None
    assert not (tmp_path / "legacy-vault").exists()
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "legacy")
    with pytest.raises(RuntimeError, match="caller opt-in"):
        HistoryVault(tmp_path / "direct-vault")


def test_archive_init_is_detected_without_restart(monkeypatch, tmp_path):
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    root = tmp_path / "encrypted"
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(root))
    service = HistoryService(Mock(), tmp_path / "legacy-vault", home=tmp_path,
                             archive_passphrase="recovery passphrase kept off chat")
    assert service.status()["vault"]["ready"] is False
    SecureHistoryArchive.create(root, "recovery passphrase kept off chat")
    assert service.status()["vault"]["ready"] is True


def test_portable_archive_unlock_is_in_memory_and_fails_closed(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    root = tmp_path / "encrypted"
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(root))
    passphrase = "test-only portable passphrase"
    SecureHistoryArchive.create(root, passphrase)

    unattended = HistoryService(Mock(), tmp_path / "unattended-vault", home=tmp_path)
    assert unattended.status()["vault"]["ready"] is (os.name == "nt")
    wrong = HistoryService(Mock(), tmp_path / "wrong-vault", home=tmp_path,
                           archive_passphrase="different test passphrase")
    assert wrong.status()["vault"]["ready"] is False
    unlocked = HistoryService(Mock(), tmp_path / "unlocked-vault", home=tmp_path,
                              archive_passphrase=passphrase)
    status = unlocked.status()
    assert status["vault"]["ready"] is True
    assert passphrase not in str(status)
    assert passphrase not in caplog.text


def test_strict_api_blocks_legacy_catalog_before_store_access(monkeypatch):
    from fastapi import HTTPException

    from server import _require_legacy_history

    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    with pytest.raises(HTTPException) as blocked:
        _require_legacy_history()
    assert blocked.value.status_code == 409


@pytest.mark.asyncio
async def test_strict_core_blocks_direct_file_ingestion_and_discovery(monkeypatch):
    from muninn.core.memory import MuninnMemory

    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    uninitialized = MuninnMemory.__new__(MuninnMemory)
    for action in (
        lambda: uninitialized.ingest_sources(sources=["secret.txt"]),
        lambda: uninitialized.discover_legacy_sources(),
        lambda: uninitialized.ingest_legacy_sources(selected_paths=["secret.txt"]),
    ):
        with pytest.raises(RuntimeError, match="strict history"):
            await action()


@pytest.mark.asyncio
async def test_strict_mode_blocks_legacy_service_entry_points(monkeypatch, tmp_path):
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    memory = Mock()
    service = HistoryService(memory, tmp_path / "legacy-vault", home=tmp_path)
    assert service.vault is None
    for operation in (
        lambda: service.sync(),
        lambda: service.run_import(apply=False),
        lambda: service.run_import(apply=True),
        lambda: service.run_analysis(apply=False),
        lambda: service.run_analysis(apply=True),
        lambda: service.capture(str(tmp_path / "chat.jsonl"), "claude_code"),
    ):
        with pytest.raises(RuntimeError, match="strict history"):
            await operation()
    assert not (tmp_path / "legacy-vault" / "manifest.db").exists()


@pytest.mark.asyncio
async def test_strict_mode_does_not_schedule_background_capture_or_analysis(monkeypatch, tmp_path):
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    service = HistoryService.__new__(HistoryService)
    service._background = set()
    service._auto_task = None
    service._task = None
    service.secure_archive = None
    service.secure_archive_root = tmp_path / "not-initialized"
    service.capture = AsyncMock()
    with pytest.raises(RuntimeError, match="strict history mode"):
        service.capture_later("chat.jsonl", "codex")
    service._launch_auto_analysis()
    with pytest.raises(RuntimeError, match="strict history mode"):
        await service.start()
    assert service._task is None
    assert not service._background
    assert service._auto_task is None
    service.capture.assert_not_awaited()


@pytest.mark.asyncio
async def test_strict_hook_ack_is_durable_and_restart_replays_it(monkeypatch, tmp_path):
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    root = tmp_path / "encrypted"
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(root))
    SecureHistoryArchive.create(root, "recovery passphrase kept off chat")
    source = (tmp_path / ".codex" / "sessions" / "2026" /
              "rollout-2026-09-28T01-02-03-11111111-1111-4111-8111-111111111111.jsonl")
    source.parent.mkdir(parents=True)
    source.write_text("PRIVATE-DURABLE-CAPTURE", encoding="utf-8")
    memory = Mock()

    before_crash = HistoryService(memory, tmp_path / "unused", home=tmp_path,
                                  archive_passphrase="recovery passphrase kept off chat")
    assert before_crash.capture_later(str(source), "codex", force=True) == "queued"
    assert before_crash._capture_journal.status()["pending"] == 1
    assert before_crash.secure_archive.status()["snapshots"] == 0

    after_restart = HistoryService(memory, tmp_path / "unused", home=tmp_path,
                                   archive_passphrase="recovery passphrase kept off chat")
    assert await after_restart._process_capture_job_once() is True
    assert after_restart.secure_archive.read_file(source) == b"PRIVATE-DURABLE-CAPTURE"
    assert after_restart._capture_journal.status()["archived"] == 1


@pytest.mark.asyncio
async def test_strict_scan_queues_only_new_or_changed_sources(monkeypatch, tmp_path):
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    monkeypatch.delenv("MUNINN_HISTORY_HOMES", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    root = tmp_path / "encrypted"
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(root))
    archive = SecureHistoryArchive.create(root, "recovery passphrase kept off chat")
    existing = (tmp_path / ".codex" / "sessions" / "2026" /
                "rollout-2026-09-28T01-02-03-11111111-1111-4111-8111-111111111111.jsonl")
    existing.parent.mkdir(parents=True)
    existing.write_text("already archived", encoding="utf-8")
    archive.archive_file(existing, "codex")
    service = HistoryService(Mock(), tmp_path / "unused", home=tmp_path,
                             archive_passphrase="recovery passphrase kept off chat")

    first = await service.scan_capture_sources()
    # The rotating integrity bucket may requeue an unchanged file for a
    # content-hash check. Which bucket matches depends on the archive key.
    assert first["queued"] + first["unchanged"] >= 1
    assert first["queued"] <= 1
    if first["queued"]:
        assert await service._process_capture_job_once() is True
        assert archive.status()["snapshots"] == 1

    new_source = existing.with_name(
        "rollout-2026-09-28T01-02-04-22222222-2222-4222-8222-222222222222.jsonl"
    )
    new_source.write_text("new missed hook", encoding="utf-8")
    second = await service.scan_capture_sources()
    assert second["queued"] == 1
    assert await service._process_capture_job_once() is True
    assert archive.read_file(new_source) == b"new missed hook"


@pytest.mark.asyncio
async def test_missing_allowed_hook_locator_is_a_retry_not_a_false_archive(monkeypatch, tmp_path):
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    root = tmp_path / "encrypted"
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(root))
    SecureHistoryArchive.create(root, "recovery passphrase kept off chat")
    service = HistoryService(Mock(), tmp_path / "unused", home=tmp_path,
                             archive_passphrase="recovery passphrase kept off chat")
    vanished = (tmp_path / ".codex" / "sessions" / "2026" /
                "rollout-2026-09-28T01-02-03-11111111-1111-4111-8111-111111111111.jsonl")
    vanished.parent.mkdir(parents=True)

    assert service.capture_later(str(vanished), "codex", force=True) == "queued"
    assert await service._process_capture_job_once() is True
    assert service._capture_journal.status()["retry"] == 1
    assert service.secure_archive.status()["snapshots"] == 0

    journal = service._capture_journal
    for _ in range(6):
        with journal._connect() as db:
            db.execute("UPDATE jobs SET due_at=0 WHERE state='retry'")
        job = journal.claim_due()
        assert job is not None
        assert journal.fail(job, "missing") == "retry"
    with journal._connect() as db:
        db.execute("UPDATE jobs SET due_at=0 WHERE state='retry'")
    assert await service._process_capture_job_once() is True
    assert service.last_secure_capture["state"] == "unavailable"
    assert journal.status() == {"unavailable": 1}


def test_missing_capture_eventually_reports_unavailable_and_recovers_when_source_appears(monkeypatch, tmp_path):
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    root = tmp_path / "encrypted"
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(root))
    SecureHistoryArchive.create(root, "recovery passphrase kept off chat")
    service = HistoryService(Mock(), tmp_path / "unused", home=tmp_path,
                             archive_passphrase="recovery passphrase kept off chat")
    source = (tmp_path / ".claude" / "projects" / "-repo" /
              "11111111-1111-4111-8111-111111111111.jsonl")
    source.parent.mkdir(parents=True)
    journal = service._require_capture_journal()
    assert journal.enqueue(source, "claude_code", immediate=True) == "queued"

    for _ in range(7):
        with journal._connect() as db:
            db.execute("UPDATE jobs SET due_at=0 WHERE state='retry'")
        job = journal.claim_due()
        assert job is not None
        assert journal.fail(job, "missing") == "retry"
    assert journal.status() == {"retry": 1}
    assert journal.enqueue(source, "claude_code") == "coalesced"

    with journal._connect() as db:
        attempts = db.execute("SELECT attempts FROM jobs").fetchone()[0]
        db.execute("UPDATE jobs SET due_at=0 WHERE state='retry'")
    assert attempts == 7
    job = journal.claim_due()
    assert job is not None
    assert journal.fail(job, "missing") == "unavailable"

    assert journal.status() == {"unavailable": 1}
    assert journal.claim_due() is None
    assert service.secure_archive.status()["snapshots"] == 0
    backup = root / "capture-jobs-backup.db"
    journal.backup_to(backup)
    with sqlite3.connect(backup) as db:
        assert db.execute("SELECT state, attempts, last_error_code FROM jobs").fetchone() == (
            "unavailable", 8, "missing")
    if os.name == "nt":
        backup_root = tmp_path / "encrypted-backup"
        service.secure_archive.backup_to(backup_root)
        restored = SecureHistoryArchive.restore_from_backup(
            backup_root, tmp_path / "restored", "recovery passphrase kept off chat")
        with sqlite3.connect(restored.root / "capture-jobs.db") as db:
            assert db.execute("SELECT state, attempts, last_error_code FROM jobs").fetchone() == (
                "unavailable", 8, "missing")

    source.write_text("later available", encoding="utf-8")
    assert journal.enqueue(source, "claude_code", immediate=True) == "queued"
    assert journal.status() == {"pending": 1}
    with journal._connect() as db:
        assert db.execute("SELECT attempts FROM jobs").fetchone()[0] == 0


def test_missing_capture_stale_failure_cannot_terminalize_reenqueued_source(monkeypatch, tmp_path):
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    root = tmp_path / "encrypted"
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(root))
    SecureHistoryArchive.create(root, "recovery passphrase kept off chat")
    service = HistoryService(Mock(), tmp_path / "unused", home=tmp_path,
                             archive_passphrase="recovery passphrase kept off chat")
    source = (tmp_path / ".claude" / "projects" / "-repo" /
              "22222222-2222-4222-8222-222222222222.jsonl")
    source.parent.mkdir(parents=True)
    journal = service._require_capture_journal()
    assert journal.enqueue(source, "claude_code", immediate=True) == "queued"
    stale = journal.claim_due()
    assert stale is not None
    source.write_text("now present", encoding="utf-8")
    assert journal.enqueue(source, "claude_code", immediate=True) == "queued"
    assert journal.fail(stale, "missing") is None
    assert journal.status() == {"pending": 1}


@pytest.mark.asyncio
async def test_source_growth_after_archive_commit_requeues_new_version(monkeypatch, tmp_path):
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    root = tmp_path / "encrypted"
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(root))
    SecureHistoryArchive.create(root, "recovery passphrase kept off chat")
    source = (tmp_path / ".codex" / "sessions" / "2026" /
              "rollout-2026-09-28T01-02-03-11111111-1111-4111-8111-111111111111.jsonl")
    source.parent.mkdir(parents=True)
    source.write_text("before", encoding="utf-8")
    service = HistoryService(Mock(), tmp_path / "unused", home=tmp_path,
                             archive_passphrase="recovery passphrase kept off chat")
    service.capture_later(str(source), "codex", force=True)
    archive = service.secure_archive
    original = archive.archive_file
    appended = False

    def append_after_commit(path, provider, **kwargs):
        nonlocal appended
        outcome = original(path, provider, **kwargs)
        if not appended:
            source.write_text("before after", encoding="utf-8")
            appended = True
        return outcome

    monkeypatch.setattr(archive, "archive_file", append_after_commit)
    assert await service._process_capture_job_once() is True
    assert service.last_secure_capture["state"] == "superseded"
    assert service._capture_journal.status()["pending"] == 1
    assert await service._process_capture_job_once() is True
    assert service.last_secure_capture["state"] == "archived"
    assert service._capture_journal.status()["archived"] == 1
    assert archive.read_file(source) == b"before after"
    assert archive.status()["snapshots"] == 2


@pytest.mark.asyncio
async def test_stop_waits_for_inflight_archive_write_before_returning(monkeypatch, tmp_path):
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    monkeypatch.setenv("MUNINN_HISTORY_INDEX_AUTO", "0")
    root = tmp_path / "encrypted"
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(root))
    SecureHistoryArchive.create(root, "recovery passphrase kept off chat")
    source = (tmp_path / ".codex" / "sessions" / "2026" /
              "rollout-2026-09-28T01-02-03-11111111-1111-4111-8111-111111111111.jsonl")
    source.parent.mkdir(parents=True)
    source.write_text("wait for the committed snapshot", encoding="utf-8")
    service = HistoryService(Mock(), tmp_path / "unused", home=tmp_path,
                             archive_passphrase="recovery passphrase kept off chat")
    service.capture_later(str(source), "codex", force=True)
    archive = service.secure_archive
    original = archive.archive_file
    entered = threading.Event()
    release = threading.Event()

    def blocked_copy(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(archive, "archive_file", blocked_copy)
    await service.start()
    assert await asyncio.to_thread(entered.wait, 5)
    stopping = asyncio.create_task(service.stop())
    await asyncio.sleep(0.05)
    assert not stopping.done()
    release.set()
    await asyncio.wait_for(stopping, 5)
    assert archive.read_file(source) == b"wait for the committed snapshot"


@pytest.mark.asyncio
async def test_rolling_scan_finds_same_size_same_mtime_rewrite(monkeypatch, tmp_path):
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    monkeypatch.setattr("muninn.history.service.STRICT_VERIFY_BUCKETS", 1)
    root = tmp_path / "encrypted"
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(root))
    archive = SecureHistoryArchive.create(root, "recovery passphrase kept off chat")
    source = (tmp_path / ".codex" / "sessions" / "2026" /
              "rollout-2026-09-28T01-02-03-11111111-1111-4111-8111-111111111111.jsonl")
    source.parent.mkdir(parents=True)
    source.write_bytes(b"first version")
    archive.archive_file(source, "codex")
    old = source.stat()
    source.write_bytes(b"other version")
    __import__("os").utime(source, ns=(old.st_atime_ns, old.st_mtime_ns))
    assert source.stat().st_size == old.st_size
    service = HistoryService(Mock(), tmp_path / "unused", home=tmp_path,
                             archive_passphrase="recovery passphrase kept off chat")

    assert (await service.scan_capture_sources())["queued"] == 1
    assert await service._process_capture_job_once() is True
    assert archive.read_file(source) == b"other version"
    assert archive.status()["snapshots"] == 2


@pytest.mark.asyncio
async def test_link_swap_after_validation_cannot_archive_outside_root(monkeypatch, tmp_path):
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    root = tmp_path / "encrypted"
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(root))
    SecureHistoryArchive.create(root, "recovery passphrase kept off chat")
    source = (tmp_path / ".codex" / "sessions" / "2026" /
              "rollout-2026-09-28T01-02-03-11111111-1111-4111-8111-111111111111.jsonl")
    source.parent.mkdir(parents=True)
    source.write_bytes(b"allowed")
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(b"OUTSIDE-PRIVATE-CANARY")
    service = HistoryService(Mock(), tmp_path / "unused", home=tmp_path,
                             archive_passphrase="recovery passphrase kept off chat")
    service.capture_later(str(source), "codex", force=True)
    archive = service.secure_archive
    original = archive.archive_file

    def swap_before_open(path, provider, **kwargs):
        source.rename(source.with_suffix(".parked"))
        try:
            source.symlink_to(outside)
        except OSError:
            pytest.skip("Symlink creation is not available on this Windows account")
        return original(path, provider, **kwargs)

    monkeypatch.setattr(archive, "archive_file", swap_before_open)
    assert await service._process_capture_job_once() is True
    assert service._capture_journal.status()["retry"] == 1
    assert archive.status()["snapshots"] == 0


@pytest.mark.asyncio
async def test_strict_hook_capture_uses_encrypted_archive_without_normal_memory(monkeypatch, tmp_path):
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    root = tmp_path / "encrypted"
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(root))
    SecureHistoryArchive.create(root, "recovery passphrase kept off chat")
    memory = Mock()
    service = HistoryService(memory, tmp_path / "legacy-vault", home=tmp_path,
                             archive_passphrase="recovery passphrase kept off chat")
    source = (tmp_path / ".codex" / "sessions" / "2026" /
              "rollout-2026-09-28T01-02-03-11111111-1111-4111-8111-111111111111.jsonl")
    source.parent.mkdir(parents=True)
    source.write_bytes(b"PRIVATE-CANARY-TRANSCRIPT")
    result = await service.capture(str(source), "codex")
    assert result["captured"] is True
    assert (await service.capture(str(source), "codex"))["archive"]["status"] == "unchanged"
    assert service.secure_archive.read_file(source) == b"PRIVATE-CANARY-TRANSCRIPT"
    assert not (tmp_path / "legacy-vault").exists()
    memory.add.assert_not_called()
    memory._metadata.set_meta.assert_not_called()
    catalog = service.secure_catalog(provider="codex")
    assert len(catalog) == 1
    assert set(catalog[0]) == {"ref", "provider", "kind", "captured_day_utc", "size_bucket_kib", "versions"}
    assert "PRIVATE-CANARY" not in str(catalog)
    assert str(source) not in str(catalog)
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError, match="outside configured"):
        await service.capture(str(outside), "codex")
    forbidden = tmp_path / ".codex" / "auth.json"
    forbidden.write_bytes(b"credential")
    with pytest.raises(ValueError, match="outside configured"):
        await service.capture(str(forbidden), "codex")
    archived = (tmp_path / ".codex" / "archived_sessions" /
                "rollout-2026-09-28T01-02-03-22222222-2222-4222-8222-222222222222.jsonl")
    archived.parent.mkdir(parents=True)
    archived.write_bytes(b"archived transcript")
    assert service._validate_capture_source(str(archived), "codex") == archived.resolve()
    note = tmp_path / ".claude" / "projects" / "-repo" / "notes" / "project-export.jsonl"
    note.parent.mkdir(parents=True)
    note.write_bytes(b"not a session")
    with pytest.raises(ValueError, match="outside configured"):
        await service.capture(str(note), "claude_code")
    gemini = tmp_path / ".gemini" / "tmp" / "project-hash" / "chats" / "session.json"
    gemini.parent.mkdir(parents=True)
    gemini.write_bytes(b'{"sessionId":"local","messages":[]}')
    assert (await service.capture(str(gemini), "gemini_cli"))["captured"] is True
    assert service._validate_capture_source(str(gemini), "gemini_cli") == gemini.resolve()
    wrong_gemini = tmp_path / ".gemini" / "settings.json"
    wrong_gemini.write_text("{}")
    with pytest.raises(ValueError, match="outside configured"):
        await service.capture(str(wrong_gemini), "gemini_cli")
    newest = sorted(root.glob("manifest-*.enc"))[-1]
    raw = bytearray(newest.read_bytes())
    raw[-1] ^= 1
    newest.write_bytes(raw)
    health = service.status()["vault"]
    assert health["ready"] is False
    assert health["error"] == "Encrypted archive integrity unavailable"


@pytest.mark.asyncio
async def test_secure_sync_is_copy_only_and_reports_unhandled_sqlite(monkeypatch, tmp_path):
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    monkeypatch.delenv("MUNINN_HISTORY_HOMES", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    root = tmp_path / "encrypted"
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(root))
    SecureHistoryArchive.create(root, "recovery passphrase kept off chat")
    transcript = (tmp_path / ".codex" / "sessions" / "2026" /
                  "rollout-2026-09-28T01-02-03-11111111-1111-4111-8111-111111111111.jsonl")
    transcript.parent.mkdir(parents=True)
    transcript.write_bytes(b"PRIVATE-COPY-ONLY-CANARY")
    (tmp_path / ".codex" / "state_1.sqlite").write_bytes(b"live-db-placeholder")
    memory = Mock()
    service = HistoryService(memory, tmp_path / "legacy-vault", home=tmp_path,
                             archive_passphrase="recovery passphrase kept off chat")
    planned = await service.secure_sync(dry_run=True)
    assert planned["apply"] is False
    assert planned["discovered"] == 1
    assert service.secure_archive.status()["snapshots"] == 0
    report = await service.secure_sync()
    assert report["captured"] == 1
    assert report["discovered"] == 1
    assert report["skipped_live_state_db"] == 1
    assert service.secure_archive.read_file(transcript) == b"PRIVATE-COPY-ONLY-CANARY"
    assert "PRIVATE-COPY-ONLY-CANARY" not in str(service.secure_catalog())
    assert not (tmp_path / "legacy-vault").exists()
    memory.add.assert_not_called()
