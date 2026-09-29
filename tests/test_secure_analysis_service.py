"""A pertinent secure search queues interpretation without blocking capture/search."""

from __future__ import annotations

import asyncio

import pytest

from muninn.history.blind_index import SecureHistoryBlindIndex
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.service import HistoryService


@pytest.mark.asyncio
async def test_search_automatically_queues_and_completes_one_analysis(tmp_path, monkeypatch):
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    monkeypatch.setenv("MUNINN_HISTORY_INDEX_AUTO", "0")
    monkeypatch.setenv("MUNINN_SECURE_AUTO_ANALYSIS", "1")
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(tmp_path / "archive"))
    source = tmp_path / "chat.jsonl"
    source.write_text("orbital-widget parser decision", encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", "test-only passphrase")
    archive.archive_file(source, "codex")
    SecureHistoryBlindIndex(archive).build()
    seen: dict[str, object] = {}

    async def fake_analyze(history, capability, *, allow_remote, should_cancel, before_remote):
        seen["raw_window"] = history._secure_model_window(capability)
        seen["allow_remote"] = allow_remote
        assert not should_cancel()
        assert before_remote is not None
        return {"status": "ok", "provider": "ollama", "model": "fixture-model",
                "analysis": {"summary": "A parser decision was mentioned.", "decisions": [],
                             "open_items": [], "uncertainty": "Not an execution receipt."}}

    monkeypatch.setattr("muninn.history.secure_analysis.analyze_secure_hit", fake_analyze)
    service = HistoryService(None, tmp_path / "history_vault", home=tmp_path)
    await service.start()
    try:
        search_id = service.queue_secure_search("orbital-widget", limit=1)

        async def completed():
            while True:
                search = service.secure_search_job_status(search_id)
                if search and search["state"] == "succeeded" and search.get("analysis_job_id"):
                    analysis = service.secure_analysis_job_status(search["analysis_job_id"])
                    if analysis and analysis["state"] == "succeeded":
                        return search, analysis
                await asyncio.sleep(0.02)

        search, analysis = await asyncio.wait_for(completed(), timeout=8)
        assert search["result"]["matches"]
        assert analysis["result"]["analysis"]["summary"] == "A parser decision was mentioned."
        assert "orbital-widget" in seen["raw_window"]
        assert seen["allow_remote"] is True
    finally:
        await service.stop()
