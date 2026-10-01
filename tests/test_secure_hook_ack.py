"""The HTTP hook response is a durable queue receipt, not a capture claim."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from time import perf_counter
from unittest.mock import Mock

from fastapi.testclient import TestClient

import server
from muninn.history import hooks
from muninn.history.capture_journal import CaptureJournal
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.service import HistoryService


def _service(monkeypatch, tmp_path):
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    root = tmp_path / "encrypted"
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(root))
    SecureHistoryArchive.create(root, "test-only portable passphrase")
    service = HistoryService(Mock(), tmp_path / "unused", home=tmp_path,
                             archive_passphrase="test-only portable passphrase")
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
    receipts = service._capture_journal.hook_receipts()
    assert len(receipts) == 1
    assert receipts == [{
        "provider": "codex", "event": "Stop", "accepted_invocations": 1,
        "last_outcome": "capture_intent", "last_accepted_at": receipts[0]["last_accepted_at"],
    }]


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
    assert journal.hook_receipts() == []


def test_receipt_failure_cannot_revoke_durable_capture_ack(monkeypatch, tmp_path, caplog):
    service, source = _service(monkeypatch, tmp_path)
    journal = service._require_capture_journal()

    def failed_receipt(*_args, **_kwargs):
        raise OSError("PRIVATE-QUEUED-CANARY")

    monkeypatch.setattr(journal, "record_hook_receipt", failed_receipt)
    response = TestClient(server.app).post("/hooks/codex", json={
        "hook_event_name": "Stop", "transcript_path": str(source),
    })
    assert response.status_code == 200
    assert journal.status()["pending"] == 1
    assert "PRIVATE-QUEUED-CANARY" not in response.text
    assert "PRIVATE-QUEUED-CANARY" not in caplog.text


def test_receipts_are_aggregate_private_and_survive_reopen(monkeypatch, tmp_path):
    service, source = _service(monkeypatch, tmp_path)
    client = TestClient(server.app)
    payload = {"hook_event_name": "Stop", "transcript_path": str(source),
               "session_id": "PRIVATE-SESSION-CANARY"}
    assert client.post("/hooks/codex", json=payload).status_code == 200
    assert client.post("/hooks/codex", json=payload).status_code == 200
    receipts = CaptureJournal(service.secure_archive, recover=False).hook_receipts()
    assert len(receipts) == 1
    assert receipts[0]["provider"] == "codex"
    assert receipts[0]["event"] == "Stop"
    assert receipts[0]["accepted_invocations"] == 2
    assert receipts[0]["last_outcome"] == "capture_intent"
    assert receipts[0]["last_accepted_at"] > 0
    status = client.get("/history/status")
    assert status.status_code == 200
    assert status.json()["data"]["hook_receipts"] == receipts
    raw = service._capture_journal.path.read_bytes()
    assert b"PRIVATE-SESSION-CANARY" not in raw
    assert b"PRIVATE-QUEUED-CANARY" not in raw
    assert str(source).encode() not in raw


def test_session_start_receipt_follows_successful_briefing(monkeypatch, tmp_path, caplog):
    service, _source = _service(monkeypatch, tmp_path)

    async def briefing(*_args, **_kwargs):
        return {}

    monkeypatch.setattr(hooks.handoffs, "project_context", briefing)
    monkeypatch.setattr(hooks.handoffs, "render_briefing", lambda _context: "ready")
    client = TestClient(server.app)
    payload = {"hook_event_name": "SessionStart", "cwd": str(tmp_path)}
    response = client.post("/hooks/claude-code", json=payload)
    assert response.status_code == 200
    assert response.json()["hookSpecificOutput"]["additionalContext"] == "ready"
    assert service._capture_journal.hook_receipts()[0]["last_outcome"] == "briefing"

    async def failed_briefing(*_args, **_kwargs):
        raise RuntimeError("PRIVATE-BRIEFING-CANARY")

    monkeypatch.setattr(hooks.handoffs, "project_context", failed_briefing)
    response = client.post("/hooks/codex", json=payload)
    assert response.status_code == 503
    assert len(service._capture_journal.hook_receipts()) == 1
    assert "PRIVATE-BRIEFING-CANARY" not in response.text
    assert "PRIVATE-BRIEFING-CANARY" not in caplog.text

    def failed_render(_context):
        raise RuntimeError("PRIVATE-RENDER-CANARY")

    monkeypatch.setattr(hooks.handoffs, "project_context", briefing)
    monkeypatch.setattr(hooks.handoffs, "render_briefing", failed_render)
    response = client.post("/hooks/codex", json=payload)
    assert response.status_code == 503
    assert len(service._capture_journal.hook_receipts()) == 1
    assert "PRIVATE-RENDER-CANARY" not in response.text
    assert "PRIVATE-RENDER-CANARY" not in caplog.text


def test_hook_receipts_status_keeps_existing_bearer_boundary(monkeypatch, tmp_path):
    service, _source = _service(monkeypatch, tmp_path)
    service.record_hook_receipt("gemini_cli", "SessionStart", "briefing")
    monkeypatch.setattr(server, "is_security_enabled", lambda: True)
    monkeypatch.setattr(server, "core_verify_token", lambda token: token == "test-only-bearer")
    client = TestClient(server.app)
    assert client.get("/history/status").status_code == 401
    response = client.get("/history/status", headers={"Authorization": "Bearer test-only-bearer"})
    assert response.status_code == 200
    assert response.json()["data"]["hook_receipts"][0]["provider"] == "gemini_cli"


def test_receipt_read_failure_does_not_hide_history_status(monkeypatch, tmp_path):
    service, _source = _service(monkeypatch, tmp_path)
    journal = service._require_capture_journal()

    def failed_read():
        raise OSError("PRIVATE-RECEIPT-READ-CANARY")

    monkeypatch.setattr(journal, "hook_receipts", failed_read)
    response = TestClient(server.app).get("/history/status")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["vault"]["ready"] is True
    assert data["hook_receipts"] is None
    assert data["hook_receipts_error"] == "unavailable"
    assert "PRIVATE-RECEIPT-READ-CANARY" not in response.text


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
