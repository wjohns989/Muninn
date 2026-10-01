"""Durable search jobs return authenticated evidence without blocking the hook loop."""

from __future__ import annotations

import asyncio
import sqlite3
import time

import pytest

from muninn.history.blind_index import SecureHistoryBlindIndex
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.service import HistoryService


@pytest.mark.asyncio
async def test_secure_search_job_reaches_result_and_preserves_capability(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    monkeypatch.setenv("MUNINN_HISTORY_INDEX_AUTO", "0")
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(tmp_path / "archive"))
    source = tmp_path / "chat.jsonl"
    source.write_text("orbital-widget parser decision", encoding="utf-8")
    archive = SecureHistoryArchive.create(
        tmp_path / "archive", "synthetic archive recovery passphrase",
    )
    archive.archive_file(source, "codex")
    SecureHistoryBlindIndex(archive).build()
    service = HistoryService(None, tmp_path / "history_vault", home=tmp_path,
                             archive_passphrase="synthetic archive recovery passphrase")
    await service.start()
    try:
        start = time.perf_counter()
        job_id = service.queue_secure_search("orbital-widget", limit=1)
        assert time.perf_counter() - start < 0.8
        assert isinstance(job_id, str) and len(job_id) >= 32
        async def completed():
            while True:
                try:
                    state = service.secure_search_job_status(job_id)
                except sqlite3.OperationalError as exc:
                    # The HTTP poll boundary exposes SQLITE_BUSY as retryable
                    # 503. This direct service test follows the same contract,
                    # within the original five-second completion deadline.
                    code = getattr(exc, "sqlite_errorcode", None)
                    if code is None or code & 0xff != sqlite3.SQLITE_BUSY:
                        raise
                    await asyncio.sleep(0.02)
                    continue
                if state["state"] == "succeeded":
                    return state
                await asyncio.sleep(0.02)

        state = await asyncio.wait_for(completed(), timeout=5)
        result = state["result"]
        assert result["total"] == result["ready"] == 1
        assert len(result["matches"]) == 1
        capability = result["matches"][0]["fetch_capability"]
        span = service.secure_fetch_span(capability)
        assert "orbital-widget" in span["redacted_text"]
    finally:
        await service.stop()
