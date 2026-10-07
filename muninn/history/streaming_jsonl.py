"""Bounded lexical stream for JSONL projections.

The lexer emits decoded strings in bounded pieces. It does not authorize any
field for release: a schema-aware parser must consume whole records, validate
roles, and stage candidate text encrypted before publishing it. In particular,
tool payloads must never be exposed merely because they contain a string.
"""

from __future__ import annotations

import codecs
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass


class StreamingJSONError(ValueError):
    """Malformed JSON token stream; callers must fail the projection closed."""


_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")
_HEX = frozenset("0123456789abcdefABCDEF")
_ESCAPES = {"\"": "\"", "\\": "\\", "/": "/", "b": "\b", "f": "\f",
            "n": "\n", "r": "\r", "t": "\t"}
_OVERSIZED_KEY = "\0oversized-json-key"


def tokens(chunks: Iterable[bytes], *, string_chunk_chars: int = 4096
           ) -> Iterator[tuple[str, str]]:
    """Lex JSONL without retaining a whole line or string in memory.

    Emits ``start_string``, zero or more ``string_chunk`` values,
    ``end_string``, ``punct``, ``scalar``, and ``newline``. JSON grammar and
    provider schemas are deliberately the responsibility of the next layer.
    The only scalar buffer is capped at 128 characters. String values have no
    size limit; even a single multi-gigabyte value is emitted in pieces.
    """
    if not 1 <= string_chunk_chars <= 65536:
        raise ValueError("Invalid JSON string chunk size")
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    state = "default"
    text_buffer: list[str] = []
    scalar: list[str] = []
    unicode_digits = ""
    high_surrogate: int | None = None

    def flush_text() -> Iterator[tuple[str, str]]:
        if text_buffer:
            yield "string_chunk", "".join(text_buffer)
            text_buffer.clear()

    def push_codepoint(value: int) -> None:
        nonlocal high_surrogate
        if high_surrogate is not None:
            if not 0xDC00 <= value <= 0xDFFF:
                raise StreamingJSONError("Invalid JSON Unicode surrogate pair")
            value = 0x10000 + ((high_surrogate - 0xD800) << 10) + (value - 0xDC00)
            high_surrogate = None
        elif 0xD800 <= value <= 0xDBFF:
            high_surrogate = value
            return
        elif 0xDC00 <= value <= 0xDFFF:
            raise StreamingJSONError("Lone JSON low surrogate")
        text_buffer.append(chr(value))

    def flush_scalar() -> Iterator[tuple[str, str]]:
        if scalar:
            value = "".join(scalar)
            scalar.clear()
            if value not in {"true", "false", "null"} and not _NUMBER.fullmatch(value):
                raise StreamingJSONError("Invalid JSON scalar")
            yield "scalar", value

    def accept(decoded: str) -> Iterator[tuple[str, str]]:
        nonlocal state, unicode_digits
        for char in decoded:
            if state == "unicode":
                if char not in _HEX:
                    raise StreamingJSONError("Invalid JSON Unicode escape")
                unicode_digits += char
                if len(unicode_digits) == 4:
                    push_codepoint(int(unicode_digits, 16))
                    unicode_digits = ""
                    state = "string"
            elif state == "escape":
                if high_surrogate is not None and char != "u":
                    raise StreamingJSONError("Invalid JSON Unicode surrogate pair")
                if char == "u":
                    unicode_digits = ""
                    state = "unicode"
                elif char in _ESCAPES:
                    text_buffer.append(_ESCAPES[char])
                    state = "string"
                else:
                    raise StreamingJSONError("Invalid JSON escape")
            elif state == "string":
                if high_surrogate is not None and char != "\\":
                    raise StreamingJSONError("Invalid JSON Unicode surrogate pair")
                if char == "\\":
                    state = "escape"
                elif char == '"':
                    yield from flush_text()
                    yield "end_string", ""
                    state = "default"
                else:
                    if ord(char) < 32 or 0xD800 <= ord(char) <= 0xDFFF:
                        raise StreamingJSONError("Invalid JSON string character")
                    text_buffer.append(char)
            else:
                if char == '"':
                    yield from flush_scalar()
                    yield "start_string", ""
                    state = "string"
                elif char in "{}[]:,":
                    yield from flush_scalar()
                    yield "punct", char
                elif char == "\n":
                    yield from flush_scalar()
                    yield "newline", ""
                elif char in " \t\r":
                    yield from flush_scalar()
                else:
                    scalar.append(char)
                    if len(scalar) > 128:
                        raise StreamingJSONError("Invalid JSON scalar length")
            # Escapes and completed Unicode pairs append decoded codepoints
            # too. Flush at their boundary, not only after the next literal.
            if len(text_buffer) >= string_chunk_chars:
                yield from flush_text()

    try:
        for chunk in chunks:
            if not isinstance(chunk, bytes):
                raise TypeError("JSONL chunks must be bytes")
            yield from accept(decoder.decode(chunk))
        yield from accept(decoder.decode(b"", final=True))
    except UnicodeDecodeError as exc:
        raise StreamingJSONError("Invalid JSONL UTF-8") from exc
    if state != "default" or high_surrogate is not None:
        raise StreamingJSONError("Truncated JSON string")
    yield from flush_scalar()


@dataclass
class _Frame:
    kind: str
    path: tuple[str | int, ...]
    state: str
    key: str | None = None
    next_index: int = 0


def events(chunks: Iterable[bytes], *, string_chunk_chars: int = 4096,
           include_record_position: bool = False,
           allow_multiline: bool = False,
           ) -> Iterator[tuple[str, tuple[str | int, ...], str]]:
    """Validate JSONL grammar and emit scalar/string fragments with paths.

    A consumer may stage selected fields encrypted, but must wait for
    ``record_end`` and validate the provider schema before publishing them.
    Unknown or tool fields are not inherently safe just because they have a
    path. The parser holds only bounded key/scalar state, not whole values.
    """
    stack: list[_Frame] = []
    root_state = "empty"
    string_mode: str | None = None
    key_parts: list[str] = []
    key_chars = 0
    key_overflow = False
    value_path: tuple[str | int, ...] = ()
    physical_line = 0

    def start_value() -> tuple[str | int, ...]:
        nonlocal root_state
        if stack:
            frame = stack[-1]
            if frame.kind == "object" and frame.state == "value" and frame.key is not None:
                frame.state = "in_value"
                return frame.path + (frame.key,)
            if frame.kind == "array" and frame.state in {"first_value", "next_value"}:
                frame.state = "in_value"
                return frame.path + (frame.next_index,)
            raise StreamingJSONError("Unexpected JSON value")
        if root_state != "empty":
            raise StreamingJSONError("Multiple JSONL roots on one line")
        root_state = "in_value"
        return ()

    def complete_value() -> None:
        nonlocal root_state
        if stack:
            frame = stack[-1]
            if frame.state != "in_value":
                raise StreamingJSONError("Invalid JSON value completion")
            frame.state = "comma_or_end"
            if frame.kind == "array":
                frame.next_index += 1
        elif root_state == "in_value":
            root_state = "complete"
        else:
            raise StreamingJSONError("Invalid JSON root completion")

    for kind, value in tokens(chunks, string_chunk_chars=string_chunk_chars):
        if kind == "start_string":
            if stack and stack[-1].kind == "object" and stack[-1].state in {"first_key", "next_key"}:
                string_mode = "key"
                key_parts = []
                key_chars = 0
                key_overflow = False
            else:
                value_path = start_value()
                string_mode = "value"
                yield "value_start", value_path, "string"
        elif kind == "string_chunk":
            if string_mode == "key":
                if not key_overflow:
                    key_chars += len(value)
                    if key_chars > 128:
                        key_overflow = True
                        key_parts.clear()
                    else:
                        key_parts.append(value)
            elif string_mode == "value":
                yield "value_chunk", value_path, value
            else:
                raise StreamingJSONError("Unexpected JSON string fragment")
        elif kind == "end_string":
            if string_mode == "key":
                frame = stack[-1]
                frame.key = _OVERSIZED_KEY if key_overflow else "".join(key_parts)
                frame.state = "colon"
            elif string_mode == "value":
                yield "value_end", value_path, ""
                complete_value()
            else:
                raise StreamingJSONError("Unexpected JSON string end")
            string_mode = None
        elif kind == "scalar":
            value_path = start_value()
            yield "value_start", value_path, "scalar"
            yield "value_chunk", value_path, value
            yield "value_end", value_path, ""
            complete_value()
        elif kind == "punct":
            if value in "{[":
                value_path = start_value()
                if len(stack) >= 64:
                    raise StreamingJSONError("JSON nesting exceeds schema bound")
                container_kind = "object" if value == "{" else "array"
                stack.append(_Frame(container_kind, value_path,
                                    "first_key" if value == "{" else "first_value"))
                yield "container_start", value_path, container_kind
            elif value in "}]":
                expected = "object" if value == "}" else "array"
                if (not stack or stack[-1].kind != expected
                        or stack[-1].state not in ({"first_key", "comma_or_end"} if expected == "object"
                                                else {"first_value", "comma_or_end"})):
                    raise StreamingJSONError("Unexpected JSON container end")
                finished = stack.pop()
                yield "container_end", finished.path, finished.kind
                complete_value()
            elif value == ":":
                if not stack or stack[-1].kind != "object" or stack[-1].state != "colon":
                    raise StreamingJSONError("Unexpected JSON colon")
                stack[-1].state = "value"
            elif value == ",":
                if not stack or stack[-1].state != "comma_or_end":
                    raise StreamingJSONError("Unexpected JSON comma")
                stack[-1].state = "next_key" if stack[-1].kind == "object" else "next_value"
        elif kind == "newline":
            if stack or root_state == "in_value":
                if not allow_multiline:
                    raise StreamingJSONError("JSONL record crosses physical lines")
                physical_line += 1
                continue
            if root_state == "complete":
                yield "record_end", (), str(physical_line) if include_record_position else ""
            root_state = "empty"
            physical_line += 1
    if stack or root_state == "in_value":
        raise StreamingJSONError("Truncated JSONL record")
    if root_state == "complete":
        yield "record_end", (), str(physical_line) if include_record_position else ""
