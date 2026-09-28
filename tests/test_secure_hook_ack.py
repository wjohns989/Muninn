"""The HTTP hook response is a durable queue receipt, not a capture claim."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from time import perf_counter
from unittest.mock import Mock

from fastapi.testclient import TestClient

import server
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.service import HistoryService


def _service(monkeypatch, tmp_path):
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    root = tmp_path / "encrypted"
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(root))
    SecureHistoryArchive.create(root, "test-only portable passphrase")
    service = HistoryService(Mock(), tmp_path / "unused", home=tmp_path)
    monkeypatch.setattr(server, "memory", Mock())
    monkeypatch.setattr(server, "_history", service)
    monkeypatch.setattr(server, "is_security_enabled", lambda: False)
    source = (tmp_path / ".codex" / "sessions" / "2026" /
              "rollout-2026-09-28T01-02-03-11111111-1111-4111-8111-111111111111.jsonl")
    source.parent.mkdir(parents=True)
    source.write_text("PRIVATE-QUEUED-CANARY", encoding="utf-8")
    return service, source


def test_hook_http_200_follows_durable_row_not_background_capture(monkeypatch, tmp_path):
    service, source = _service(monkeypatch, tmp_path)
    response = TestClient(server.app).post("/hooks/codex", json={
        "hook_event_name": "Stop", "transcript_path": str(source),
    })

    assert response.status_code == 200
    assert response.json() == {}
    assert service._capture_journal.status()["pending"] == 1
    assert service.secure_archive.status()["snapshots"] == 0


def test_hook_failed_commit_is_nonacknowledged_without_path_in_output(monkeypatch, tmp_path, caplog):
    service, source = _service(monkeypatch, tmp_path)
    journal = service._require_capture_journal()

    def failed_commit(*_args, **_kwargs):
        raise OSError("PRIVATE-QUEUED-CANARY")

    monkeypatch.setattr(journal, "enqueue", failed_commit)
    response = TestClient(server.app).post("/hooks/codex", json={
        "hook_event_name": "Stop", "transcript_path": str(source),
    })

    assert response.status_code == 503
    assert "PRIVATE-QUEUED-CANARY" not in response.text
    assert "PRIVATE-QUEUED-CANARY" not in caplog.text
    assert journal.status() == {}


def test_hook_cannot_respond_before_journal_commit(monkeypatch, tmp_path):
    service, source = _service(monkeypatch, tmp_path)
    journal = service._require_capture_journal()
    original = journal.enqueue
    entered = Event()
    release = Event()

    def paused_enqueue(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(journal, "enqueue", paused_enqueue)
    payload = {"hook_event_name": "Stop", "transcript_path": str(source)}
    with ThreadPoolExecutor(max_workers=1) as pool:
        response = pool.submit(lambda: TestClient(server.app).post("/hooks/codex", json=payload))
        assert entered.wait(5)
        assert not response.done()
        assert journal.status() == {}
        release.set()
        assert response.result(timeout=5).status_code == 200
    assert journal.status()["pending"] == 1


def test_repeated_local_hook_fast_path_stays_within_client_deadline(monkeypatch, tmp_path):
    _service_instance, source = _service(monkeypatch, tmp_path)
    client = TestClient(server.app)
    payload = {"hook_event_name": "Stop", "transcript_path": str(source)}
    elapsed = []
    for _ in range(10):
        started = perf_counter()
        assert client.post("/hooks/codex", json=payload).status_code == 200
        elapsed.append(perf_counter() - started)
    assert max(elapsed) < 0.8


def test_journal_lock_contention_returns_bounded_nonacknowledgement(monkeypatch, tmp_path):
    service, source = _service(monkeypatch, tmp_path)
    journal = service._require_capture_journal()
    blocker = sqlite3.connect(journal.path)
    try:
        blocker.execute("BEGIN IMMEDIATE")
        started = perf_counter()
        response = TestClient(server.app).post("/hooks/codex", json={
            "hook_event_name": "Stop", "transcript_path": str(source),
        })
        elapsed = perf_counter() - started
    finally:
        blocker.rollback()
        blocker.close()

    assert response.status_code == 503
    assert elapsed < 0.8
    assert journal.status() == {}
