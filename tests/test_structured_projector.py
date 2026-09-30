import json

import pytest

from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.secure_projection_store import SecureProjectionStore
from muninn.history.streaming_jsonl import StreamingJSONError
from muninn.history.structured_projector import (
    ProjectionCancelled,
    UnsupportedTranscript,
    build_transcript_projection,
    project_transcript,
)


def _project(tmp_path, provider, rows):
    source = tmp_path / "conversation.jsonl"
    source.write_bytes(b"\n".join(json.dumps(row).encode() for row in rows) + b"\n")
    archive = SecureHistoryArchive.create(tmp_path / "archive", "test-only portable recovery phrase")
    archive.archive_file(source, provider)
    entry = archive._load_manifest()["files"][str(source.resolve())][0]
    store = SecureProjectionStore(archive)
    attempt = build_transcript_projection(store, entry, 0)
    with store._connect() as db:
        count = db.execute("SELECT count FROM attempts WHERE attempt=?", (attempt,)).fetchone()[0]
    return "".join(store.get_page(entry, 0, attempt, i) for i in range(count)), store


def test_codex_oversized_message_and_tool_output_omitted(tmp_path):
    huge = "ordinary prose " * 20000  # One physical record well beyond 256 KiB.
    output, store = _project(tmp_path, "codex", [
        {"type": "event_msg", "payload": {"type": "user_message",
                                           "message": huge + " api_key=secretvalue"}},
        {"type": "response_item", "payload": {"type": "function_call_output",
                                               "output": "TOOL_OUTPUT_CANARY"}},
        {"type": "event_msg", "payload": {"type": "agent_message",
                                           "message": "Final answer remains."}},
    ])
    assert output.count("ordinary prose") == 20000
    assert "Final answer remains" in output
    assert "TOOL_OUTPUT_CANARY" not in output
    assert "secretvalue" not in output
    assert b"secretvalue" not in store.db_path.read_bytes()


def test_codex_response_blocks_require_text_type(tmp_path):
    output, _ = _project(tmp_path, "codex", [
        {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                               "content": [
                                                   {"type": "output_text", "text": "Visible answer"},
                                                   {"type": "tool_result", "text": "TOOL_CANARY"},
                                               ]}},
    ])
    assert "Visible answer" in output and "TOOL_CANARY" not in output


def test_codex_non_user_visible_analysis_channel_is_omitted(tmp_path):
    output, _ = _project(tmp_path, "codex", [
        {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                               "channel": "analysis", "content": "HIDDEN_CANARY"}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                               "channel": "final", "content": "Visible answer"}},
    ])
    assert "Visible answer" in output and "HIDDEN_CANARY" not in output


def test_claude_sidechain_system_and_tool_result_omitted(tmp_path):
    output, _ = _project(tmp_path, "claude_code", [
        {"type": "user", "isSidechain": True, "message": {"content": "SIDE_CANARY"}},
        {"type": "user", "promptSource": "system", "message": {"content": "SYSTEM_CANARY"}},
        {"type": "user", "message": {"content": [
            {"type": "text", "text": "Actual request"},
            {"type": "tool_result", "content": "TOOL_CANARY"},
        ]}},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Actual reply"},
            {"type": "tool_use", "input": "CALL_CANARY"},
        ]}},
    ])
    assert "Actual request" in output and "Actual reply" in output
    assert all(canary not in output for canary in
               ("SIDE_CANARY", "SYSTEM_CANARY", "TOOL_CANARY", "CALL_CANARY"))


def test_gemini_jsonl_text_not_tool_calls(tmp_path):
    output, _ = _project(tmp_path, "gemini_cli", [
        {"type": "user", "content": "Gemini question"},
        {"type": "gemini", "content": [{"type": "text", "text": "Gemini answer"}],
         "toolCalls": [{"result": "TOOL_CANARY"}]},
    ])
    assert "Gemini question" in output and "Gemini answer" in output
    assert "TOOL_CANARY" not in output


def test_gemini_object_text_before_tool_type_is_omitted(tmp_path):
    output, _ = _project(tmp_path, "gemini_cli", [
        {"type": "gemini", "content": [
            {"text": "TOOL_CANARY", "type": "functionCall"},
            {"type": "text", "text": "Visible answer"},
        ]},
    ])
    assert "Visible answer" in output and "TOOL_CANARY" not in output


def test_gemini_mixed_content_object_is_omitted(tmp_path):
    output, _ = _project(tmp_path, "gemini_cli", [
        {"type": "gemini", "content": {"text": "TOOL_CANARY",
                                       "functionCall": {"name": "shell"}}},
        {"type": "gemini", "content": "Visible answer"},
    ])
    assert "Visible answer" in output and "TOOL_CANARY" not in output


def test_gemini_container_json_streams_oversized_message(tmp_path):
    source = tmp_path / "whole.json"
    large = "lunar-widget " * 25000
    source.write_text(json.dumps({"sessionId": "local-fixture", "messages": [
        {"content": large, "type": "user"},  # role appears after the long text
        {"type": "tool", "content": "TOOL_CANARY"},
        {"type": "gemini", "content": "Final answer."},
    ]}))
    archive = SecureHistoryArchive.create(tmp_path / "archive", "test-only portable recovery phrase")
    archive.archive_file(source, "gemini_cli")
    entry = archive._load_manifest()["files"][str(source.resolve())][0]
    store = SecureProjectionStore(archive)
    attempt = build_transcript_projection(store, entry, 0)
    with store._connect() as db:
        count = db.execute("SELECT count FROM attempts WHERE attempt=?", (attempt,)).fetchone()[0]
    output = "".join(store.get_page(entry, 0, attempt, i) for i in range(count))
    assert output.count("lunar-widget") == 25000
    assert "Final answer." in output and "TOOL_CANARY" not in output


def test_gemini_invalid_container_message_fails_closed(tmp_path):
    source = tmp_path / "whole.json"
    source.write_text(json.dumps({"messages": ["not a message"]}))
    archive = SecureHistoryArchive.create(tmp_path / "archive", "test-only portable recovery phrase")
    archive.archive_file(source, "gemini_cli")
    entry = archive._load_manifest()["files"][str(source.resolve())][0]
    store = SecureProjectionStore(archive)
    with pytest.raises(UnsupportedTranscript):
        build_transcript_projection(store, entry, 0)
    with store._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0


def test_two_pass_record_count_disagreement_fails_closed():
    one = b'{"type":"event_msg","payload":{"type":"user_message","message":"one"}}\n'
    two = b'{"type":"event_msg","payload":{"type":"agent_message","message":"two"}}\n'

    class DifferentFirstPass:
        def _iter_verified_entry(self, _entry):
            yield one

    with pytest.raises(StreamingJSONError, match="different record counts"):
        list(project_transcript(DifferentFirstPass(), {"kind": "transcript", "provider": "codex"},
                                (one, two)))


def test_cancelled_projection_leaves_no_completed_or_staged_pages(tmp_path):
    source = tmp_path / "chat.jsonl"
    source.write_text(json.dumps({"type": "event_msg", "payload": {
        "type": "user_message", "message": "large conversation " * 20000}}))
    archive = SecureHistoryArchive.create(tmp_path / "archive", "test-only portable recovery phrase")
    archive.archive_file(source, "codex")
    entry = archive._load_manifest()["files"][str(source.resolve())][0]
    store = SecureProjectionStore(archive)
    with pytest.raises(ProjectionCancelled):
        build_transcript_projection(store, entry, 0, should_cancel=lambda: True)
    with store._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == 0
