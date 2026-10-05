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
        "configured": False, "pending_sources": 0, "parked_private_windows": 0}
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
