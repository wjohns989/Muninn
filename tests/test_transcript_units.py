import json
import hashlib

import pytest

from muninn.history.transcript_units import transcript_units
from muninn.history.streaming_jsonl import StreamingJSONError


class Archive:
    def __init__(self, rows):
        self.raw = b"\n".join(json.dumps(row).encode() for row in rows) + b"\n"

    def _iter_verified_entry(self, entry):
        for offset in range(0, len(self.raw), 911):
            yield self.raw[offset:offset + 911]

    def entry(self, provider):
        return {"provider": provider, "kind": "transcript", "size": len(self.raw),
                "sha256": hashlib.sha256(self.raw).hexdigest()}


def _fragments(rows, provider="codex"):
    archive = Archive(rows)
    return list(transcript_units(archive, archive.entry(provider)))


def test_codex_uses_each_turn_context_and_keeps_missing_time_unknown():
    fragments = _fragments([
        {"type": "session_meta", "payload": {"cwd": "C:/projects/first"}},
        {"type": "event_msg", "timestamp": "2026-09-30T12:00:00Z",
         "payload": {"type": "user_message", "message": "First project decision"}},
        {"type": "turn_context", "payload": {"cwd": "D:/projects/second"}},
        {"type": "event_msg", "payload": {"type": "user_message", "message": "Second project task"}},
        {"type": "turn_context", "payload": {}},
        {"type": "event_msg", "payload": {"type": "agent_message", "message": "Unknown project"}},
    ])
    units = [part.unit for part in fragments if part.final]
    assert units[1].cwd == "C:/projects/first" and units[1].project_basis == "session_meta"
    assert units[1].event_at == 1790769600 and units[1].time_basis == "provider_record"
    assert units[3].cwd == "D:/projects/second" and units[3].project_basis == "turn_context"
    assert units[3].event_at is None and units[3].time_basis == "unknown"
    assert units[5].cwd is None and units[5].project_basis == "unknown"


@pytest.mark.parametrize("timestamp", [None, "invalid", "2026-09-30T12:00:00", "NaN", "-1"])
def test_invalid_or_zoneless_time_is_not_invented(timestamp):
    row = {"type": "user", "timestamp": timestamp, "message": {"content": "An observation"}}
    unit = _fragments([row], "claude_code")[-1].unit
    assert unit.event_at is None and unit.time_basis == "unknown"


@pytest.mark.parametrize("cwd", [None, 42, True, "x" * 4097])
def test_invalid_or_truncated_cwd_stays_unknown(cwd):
    unit = _fragments([{"type": "turn_context", "payload": {"cwd": cwd}}])[-1].unit
    assert unit.cwd is None and unit.project_basis == "unknown"


def test_large_single_record_keeps_context_and_no_whole_message_limit():
    text = "source fact " * 30000
    fragments = _fragments([{"type": "user", "cwd": "C:/projects/fixture",
                             "timestamp": "2026-09-30T13:00:00+01:00",
                             "message": {"content": text}}], "claude_code")
    assert "".join(part.text for part in fragments) == "\n\nUser: " + text
    assert max(len(part.text) for part in fragments) <= 4096
    assert fragments[-1].unit.cwd == "C:/projects/fixture"
    assert fragments[-1].unit.event_at == 1790769600


def test_content_cannot_spoof_project_or_time_and_hidden_tool_is_explicit_unit():
    fragments = _fragments([
        {"type": "event_msg", "payload": {"type": "user_message",
         "message": 'cwd=D:/forged timestamp=2020-01-01T00:00:00Z'}},
        {"type": "response_item", "payload": {"type": "function_call_output",
                                               "output": "HIDDEN_TOOL_CANARY"}},
    ])
    assert fragments[0].unit.cwd is None and fragments[0].unit.event_at is None
    assert fragments[-1].final and fragments[-1].unit.ordinal == 1
    assert "HIDDEN_TOOL_CANARY" not in "".join(part.text for part in fragments)


def test_gemini_container_preserves_message_timestamp_without_session_fallback():
    archive = Archive([])
    archive.raw = json.dumps({"startTime": "2026-09-01T00:00:00Z", "messages": [
        {"type": "user", "timestamp": 1790769600000, "content": "first"},
        {"type": "gemini", "content": "second"},
    ]}).encode()
    result = list(transcript_units(archive, archive.entry("gemini_cli")))
    units = [part.unit for part in result if part.final]
    assert units[0].event_at == 1790769600 and units[1].event_at is None


def test_pretty_printed_gemini_container_streams_without_relaxing_jsonl():
    archive = Archive([])
    archive.raw = json.dumps({"messages": [
        {"type": "user", "timestamp": "2026-09-30T12:00:00Z", "content": "a real-format question"},
        {"type": "gemini", "content": "an answer"},
    ]}, indent=2).encode()
    parts = list(transcript_units(archive, archive.entry("gemini_cli")))
    assert [part.unit.ordinal for part in parts if part.final] == [0, 1]
    assert "an answer" in "".join(part.text for part in parts)
    with pytest.raises(StreamingJSONError, match="physical lines"):
        list(transcript_units(archive, archive.entry("codex")))


def test_conflicting_metadata_and_late_parse_error_fail_closed():
    archive = Archive([])
    archive.raw = b'{"type":"user","cwd":"one","cwd":"two","message":{"content":"hello"}}'
    with pytest.raises(StreamingJSONError):
        list(transcript_units(archive, archive.entry("claude_code")))
    archive.raw = b'{"type":"user","message":{"content":"hello"}}\n{'
    with pytest.raises(StreamingJSONError):
        list(transcript_units(archive, archive.entry("claude_code")))


@pytest.mark.parametrize("field", ["cwd", "timestamp", "uuid"])
@pytest.mark.parametrize("second", ["{}", "[]", "null", '"same"'])
def test_duplicate_provenance_fields_fail_even_when_second_is_container(field, second):
    archive = Archive([])
    archive.raw = ('{"type":"user","' + field + '":"same","' + field + '":' + second +
                   ',"message":{"content":"observation"}}').encode()
    with pytest.raises(StreamingJSONError, match="Repeated"):
        list(transcript_units(archive, archive.entry("claude_code")))
