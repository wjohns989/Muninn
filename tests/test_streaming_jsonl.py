"""The projection lexer must not have a line or string-size cutoff."""

from __future__ import annotations

import json

import pytest

from muninn.history.streaming_jsonl import StreamingJSONError, events, tokens


def _pieces(raw: bytes, width: int):
    return (raw[index:index + width] for index in range(0, len(raw), width))


def test_long_string_is_emitted_in_bounded_chunks() -> None:
    message = "a" * (2 * 1024 * 1024) + " café " + "z" * 8192
    raw = (json.dumps({"message": message}, ensure_ascii=False) + "\n").encode()
    values = list(tokens(_pieces(raw, 113), string_chunk_chars=2048))
    chunks = [value for kind, value in values if kind == "string_chunk"]
    assert max(map(len, chunks)) <= 2048
    assert "".join(chunks[1:]) == message
    assert values[-1] == ("newline", "")


@pytest.mark.parametrize("width", [1, 2, 3, 7])
def test_split_escapes_unicode_and_newlines(width: int) -> None:
    raw = b'{"text":"A\\n\\uD83D\\uDE00 caf\xc3\xa9"}\n'
    values = list(tokens(_pieces(raw, width), string_chunk_chars=2))
    assert "".join(value for kind, value in values if kind == "string_chunk") == "textA\n😀 café"
    assert values[-1] == ("newline", "")


@pytest.mark.parametrize("raw", [
    b'{"x":"\\uD83D"}\n', b'{"x":"\\uDE00"}\n',
    b'{"x":"\\uD83D\\n"}\n', b'{"x":"bad\x01"}\n',
    b'{"x":"\\q"}\n', b'{"x":"unfinished',
    b'{"x":"\xff"}\n', b'{"x":' + b'9' * 129 + b'}\n',
])
def test_invalid_or_truncated_tokens_fail_closed(raw: bytes) -> None:
    with pytest.raises(StreamingJSONError):
        list(tokens(_pieces(raw, 3)))


def test_events_track_only_bounded_string_fragments_with_paths() -> None:
    raw = b'{"type":"event_msg","payload":{"message":"' + b'a' * 100000 + b'"},"tools":["hidden"]}\n'
    seen = list(events(_pieces(raw, 19), string_chunk_chars=1024))
    message = [value for kind, path, value in seen
               if kind == "value_chunk" and path == ("payload", "message")]
    assert len(message) > 90 and max(map(len, message)) <= 1024
    assert sum(map(len, message)) == 100000
    assert ("value_chunk", ("tools", 0), "hidden") in seen
    assert seen[-1] == ("record_end", (), "")


@pytest.mark.parametrize("raw", [
    b'{"a":1,}\n', b'{"a":}\n', b'{"a" 1}\n', b'{"a":1 "b":2}\n',
    b'[1,]\n', b'{"a":1}{"b":2}\n', b'{"a":\n1}\n',
])
def test_events_reject_malformed_jsonl_grammar(raw: bytes) -> None:
    with pytest.raises(StreamingJSONError):
        list(events(_pieces(raw, 2)))
