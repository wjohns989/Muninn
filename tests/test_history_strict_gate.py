"""Strict history mode must not touch the legacy plaintext import pipeline."""

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
    service = HistoryService(Mock(), tmp_path / "legacy-vault", home=tmp_path)
    assert service.status()["vault"]["ready"] is False
    SecureHistoryArchive.create(root, "recovery passphrase kept off chat")
    assert service.status()["vault"]["ready"] is True


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
    service.capture_later("chat.jsonl", "codex")
    service._launch_auto_analysis()
    await service.start()
    assert service._task is None
    assert not service._background
    assert service._auto_task is None
    service.capture.assert_not_awaited()


@pytest.mark.asyncio
async def test_strict_hook_capture_uses_encrypted_archive_without_normal_memory(monkeypatch, tmp_path):
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    root = tmp_path / "encrypted"
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(root))
    SecureHistoryArchive.create(root, "recovery passphrase kept off chat")
    memory = Mock()
    service = HistoryService(memory, tmp_path / "legacy-vault", home=tmp_path)
    source = (tmp_path / ".codex" / "sessions" / "2026" /
              "rollout-2026-09-28T01-02-03-11111111-1111-4111-8111-111111111111.jsonl")
    source.parent.mkdir(parents=True)
    source.write_bytes(b"PRIVATE-CANARY-TRANSCRIPT")
    result = await service.capture(str(source), "codex")
    assert result["captured"] is True
    assert await service.capture(str(source), "codex") == {"skipped": "debounced"}
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
    service = HistoryService(memory, tmp_path / "legacy-vault", home=tmp_path)
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
