"""Real archived-snapshot search to continuation proof, with synthetic canaries."""

import base64
import json
import time

import pytest

from muninn.history.blind_index import SecureHistoryBlindIndex
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.secure_projection_access import ProjectionAccess
from muninn.history.secure_projection_store import ProjectionIntegrityError


def _ready(access, capability):
    result = access.start(capability)
    for _ in range(100):
        if result["state"] != "pending":
            return result
        time.sleep(0.01)
        result = access.poll(capability)
    pytest.fail("bounded transcript projection did not finish")


def _archive(tmp_path, rows, provider="codex"):
    source = tmp_path / "conversation.jsonl"
    source.write_bytes(b"\n".join(json.dumps(row).encode() for row in rows) + b"\n")
    archive = SecureHistoryArchive.create(tmp_path / "archive", "test-only portable recovery phrase")
    archive.archive_file(source, provider)
    SecureHistoryBlindIndex(archive).build()
    entry = archive._load_manifest()["files"][str(source.resolve())][0]
    return archive, entry


def _capability(archive, term):
    found = SecureHistoryBlindIndex(archive).search(term)
    assert found["matches"]
    return found["matches"][0]["fetch_capability"]


def test_real_search_projection_pages_replay_and_restart(tmp_path):
    long_message = "lunar-widget " * 1000 + " api_key=secretvalue " + "continuation " * 800
    archive, entry = _archive(tmp_path, [
        {"type": "event_msg", "payload": {"type": "user_message", "message": long_message}},
        {"type": "response_item", "payload": {"type": "function_call_output",
                                               "output": "TOOL_OUTPUT_CANARY"}},
        {"type": "event_msg", "payload": {"type": "agent_message", "message": "Final answer."}},
    ])
    capability = _capability(archive, "lunar-widget")
    access = ProjectionAccess(archive)
    try:
        ready = _ready(access, capability)
        assert ready["state"] == "ready" and ready["pages"] > 2
        assert ready["coverage"] == {"source_units": 3, "conversational_units": 2,
                                     "omitted_units": 1}
        cursor = ready["cursor"]
        pages = []
        while cursor:
            page = access.page(cursor)
            pages.append(page["redacted_text"])
            if len(pages) == 1:
                with pytest.raises(ProjectionIntegrityError):
                    access.page(cursor)
                raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
                tampered = base64.urlsafe_b64encode(raw[:-1] + bytes([raw[-1] ^ 1])).decode().rstrip("=")
                with pytest.raises(ProjectionIntegrityError):
                    access.page(tampered)
            cursor = page["next_cursor"]
        combined = "".join(pages)
        assert combined.count("lunar-widget") == 1000
        assert "Final answer." in combined
        assert "TOOL_OUTPUT_CANARY" not in combined and "secretvalue" not in combined
        assert b"secretvalue" not in access.store.db_path.read_bytes()
    finally:
        access.close()
    reopened = ProjectionAccess(archive)
    try:
        assert reopened.start(capability)["state"] == "ready"
        with pytest.raises(ProjectionIntegrityError):
            reopened.page(ready["cursor"])  # per-process cursors expire on restart
    finally:
        reopened.close()


def test_tool_only_hit_no_match(tmp_path):
    archive, _entry = _archive(tmp_path, [
        {"type": "response_item", "payload": {"type": "function_call_output",
                                               "output": "toolonly-canary"}},
    ])
    capability = _capability(archive, "toolonly-canary")
    access = ProjectionAccess(archive)
    try:
        empty = _ready(access, capability)
        assert empty["state"] == "no_conversational_match"
        assert empty["coverage"]["omitted_units"] == 1
        with pytest.raises(ProjectionIntegrityError):
            access.page("not-a-cursor")
    finally:
        access.close()


def test_expired_cursor_and_no_page_count_cutoff(tmp_path, monkeypatch):
    archive, entry = _archive(tmp_path, [
        {"type": "event_msg", "payload": {"type": "user_message", "message": "lunar-widget"}},
    ])
    access = ProjectionAccess(archive)
    try:
        ready = _ready(access, _capability(archive, "lunar-widget"))
        assert ready["state"] == "ready"
        original_time = time.time
        with monkeypatch.context() as clock:
            clock.setattr(time, "time", lambda: original_time() + 601)
            with pytest.raises(ProjectionIntegrityError):
                access.page(ready["cursor"])
        # Simulate a very long completed projection at its 4,097th page;
        # cursor replay state depends on sessions, not pages already read.
        session = "c" * 32
        access._sessions[session] = (4096, int(time.time()) + 600)
        monkeypatch.setattr(access, "_find_complete", lambda _entry, _version: ("a" * 32, 5000))
        monkeypatch.setattr(access.store, "get_page", lambda *_args: "safe page")
        cursor = access._cursor(entry, 0, "lunar-widget", "a" * 32, session, 4096)
        page = access.page(cursor)
        assert page["redacted_text"] == "safe page" and page["next_cursor"]
    finally:
        access.close()
