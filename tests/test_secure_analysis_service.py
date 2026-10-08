"""A pertinent secure search queues interpretation without blocking capture/search."""

from __future__ import annotations

import asyncio
import json

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
    source.write_text(json.dumps({"type": "event_msg", "payload": {
        "type": "user_message", "message": "orbital-widget parser decision"}}) + "\n", encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", "test-only passphrase")
    archive.archive_file(source, "codex")
    SecureHistoryBlindIndex(archive).build()
    seen: dict[str, object] = {}

    async def fake_analyze(history, cited_source, descriptor, *, allow_remote, should_cancel, before_remote,
                           remote_not_sent, expected_remote_generation,
                           prefer_remote=False, remote_gate=None):
        seen["raw_window"] = cited_source.reopen(descriptor)["text"]
        seen["allow_remote"] = allow_remote
        assert not should_cancel()
        assert before_remote is not None
        assert remote_not_sent is not None
        assert isinstance(expected_remote_generation, int)
        result = {"status": "ok", "provider": "ollama", "model": "fixture-model",
                "analysis": {"summary": "A parser decision was mentioned.", "decisions": [],
                             "open_items": [], "uncertainty": "Not an execution receipt."}}
        return {**result, "extraction": {"format": 1, "window": descriptor,
            "model_identity": "a" * 64, "result": result, "proposals": [{
                "type": "decision", "text": "A parser decision was mentioned.",
                "quote": seen["raw_window"], "start": 0}]}}

    monkeypatch.setattr("muninn.history.secure_analysis.analyze_cited_window", fake_analyze, raising=False)
    async def forbidden_legacy(*args, **kwargs):
        pytest.fail("queued durable analysis must not use the legacy inference route")
    monkeypatch.setattr("muninn.history.secure_analysis.analyze_secure_hit", forbidden_legacy)
    service = HistoryService(None, tmp_path / "history_vault", home=tmp_path,
                             archive_passphrase="test-only passphrase")
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
        assert seen["allow_remote"] is False
        assert len(analysis["memory_refs"]) == 1
        from muninn.history.memory_ledger import MemoryLedger
        candidate = MemoryLedger(archive).get(analysis["memory_refs"][0])
        assert candidate["proposal_origin"] == "model"
    finally:
        await service.stop()
