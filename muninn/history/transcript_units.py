"""Private streaming source units with provider event time and per-turn cwd.

Fragments are provisional until the iterator finishes authenticating both
archive passes. Persist them only encrypted and publish only after exhaustion.
Capture time and file mtime never stand in for an absent message timestamp.
"""

from __future__ import annotations

import math
import hashlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from muninn.history.streaming_jsonl import StreamingJSONError, events
from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.structured_projector import (
    ProjectionCancelled, UnsupportedTranscript, _metadata_units, _project_record, _role,
)

PARSER_VERSION = 1
_PATHS = {("timestamp",): 128, ("cwd",): 4096, ("id",): 256,
          ("uuid",): 256, ("sessionId",): 256, ("payload", "cwd"): 4096,
          ("payload", "id"): 256, ("payload", "session_id"): 256}


@dataclass(frozen=True)
class SourceUnit:
    ordinal: int
    provider: str
    kind: str | None
    role: str | None
    event_at: float | None
    time_basis: str
    cwd: str | None
    project_basis: str
    native_id: str | None
    physical_line: int | None = None


@dataclass(frozen=True)
class UnitFragment:
    unit: SourceUnit
    text: str
    final: bool = False


def _bounded(meta: dict, path: tuple[str, ...]) -> str | None:
    value = meta.get(path)
    limit = _PATHS.get(path, 128)
    if (path != ("timestamp",) and meta.get(("__value_kind__",) + path) != "string"
            or not isinstance(value, str) or not value or len(value) > limit
            or any(ord(char) < 32 for char in value)):
        return None
    return value


def _event_time(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        if value.replace(".", "", 1).isdigit():
            result = float(value)
            if result > 1e11:
                result /= 1000
        else:
            stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
            # A zone-less timestamp cannot safely acquire this machine's zone.
            if stamp.tzinfo is None:
                return None
            result = stamp.timestamp()
        return result if math.isfinite(result) and 0 <= result <= 253402300799 else None
    except (ValueError, OverflowError, OSError):
        return None


def transcript_units(archive: Any, entry: dict, source: Iterable[bytes] | None = None, *,
                     should_cancel: Callable[[], bool] = lambda: False,
                     _prefix_entry: dict | None = None,
                     _parent_parts: Callable[[], Iterator[UnitFragment]] | None = None,
                     ) -> Iterator[UnitFragment]:
    """Stream every source unit, including an explicit end marker for omissions.

    Uses the same supported conversational schema as redacted agent projection.
    Tool output is not silently interpreted as user intent. Each JSONL record,
    or Gemini container message, has a stable ordinal within this snapshot.
    Neither the whole record nor an unbounded message is retained in memory.
    """
    provider = entry.get("provider")
    if entry.get("kind") != "transcript" or provider not in {
            "codex", "claude_code", "gemini_cli"}:
        raise UnsupportedTranscript("Unsupported transcript snapshot")
    def checked(stream: Iterable[bytes]) -> Iterator[bytes]:
        digest, size = hashlib.sha256(), 0
        try:
            for chunk in stream:
                if should_cancel():
                    raise ProjectionCancelled("source-unit extraction cancelled")
                digest.update(chunk)
                size += len(chunk)
                yield chunk
            if size != entry.get("size") or digest.hexdigest() != entry.get("sha256"):
                raise VaultIntegrityError("Source-unit stream does not match authenticated snapshot")
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()

    first = checked(archive._iter_verified_entry(entry))
    second = checked(source if source is not None else archive._iter_verified_entry(entry))
    streams = [first, second]
    labels = body = None
    cwd, project_basis = None, "unknown"
    ordinal_base = line_base = 0
    saw_container = False
    try:
        if _prefix_entry is not None and _parent_parts is not None:
            if provider not in {"codex", "claude_code"}:
                raise UnsupportedTranscript("Incremental source units require JSONL")
            tail, line_base, ended = _prefix_tail(first, _prefix_entry, should_cancel)
            streams.append(tail)
            if not ended:
                # No record boundary: suffix might continue a JSONL record.
                # Discard the bounded probe and take the original full parser.
                first.close()
                first = checked(archive._iter_verified_entry(entry))
                streams.append(first)
                line_base = 0
            else:
                parent_parts = _parent_parts()
                try:
                    for part in parent_parts:
                        if should_cancel():
                            raise ProjectionCancelled("source-unit extraction cancelled")
                        yield part
                        if part.final:
                            ordinal_base = part.unit.ordinal + 1
                            if provider == "codex":
                                cwd, project_basis = part.unit.cwd, part.unit.project_basis
                finally:
                    parent_parts.close()
                first = tail
                second, second_lines, second_ended = _prefix_tail(second, _prefix_entry, should_cancel)
                streams.append(second)
                if second_lines != line_base or not second_ended:
                    raise VaultIntegrityError("Source-unit prefix passes disagree")
        labels = _metadata_units(first, provider, should_cancel, extra_paths=_PATHS)
        body = iter(events(second, allow_multiline=provider == "gemini_cli"))
        for ordinal, (prefix, meta) in enumerate(labels, start=ordinal_base):
            kind = meta.get(("type",))
            role = _role(provider, meta)
            if prefix:
                saw_container = True
            if provider == "codex" and kind in {"session_meta", "turn_context"}:
                cwd = _bounded(meta, ("payload", "cwd"))
                project_basis = kind if cwd is not None else "unknown"
            elif provider in {"claude_code", "gemini_cli"}:
                # Record-local provenance must not inherit a later cwd.
                cwd = _bounded(meta, ("cwd",))
                project_basis = "record_cwd" if cwd is not None else "unknown"
            event_at = _event_time(_bounded(meta, ("timestamp",)))
            unit = SourceUnit(
                ordinal, provider, kind, role, event_at,
                "provider_record" if event_at is not None else "unknown", cwd,
                project_basis,
                _bounded(meta, ("payload", "id")) if provider == "codex" else
                _bounded(meta, ("uuid",)) or _bounded(meta, ("id",)),
                int(meta[("__physical_line__",)]) + line_base if not prefix else None,
            )
            for text in _project_record(body, provider, kind, role, prefix, should_cancel):
                yield UnitFragment(unit, text)
            yield UnitFragment(unit, "", True)
        # Authenticate all trailing container tokens and both EOF checks.
        for event, path, part in body:
            if should_cancel():
                raise ProjectionCancelled("source-unit extraction cancelled")
            if event == "container_start" and path == ("messages",) and part == "array":
                saw_container = True
            if (event == "container_end" and len(path) == 2 and path[0] == "messages"
                    and isinstance(path[1], int)):
                raise StreamingJSONError("Transcript passes have different message counts")
            if event == "record_end" and not saw_container:
                raise StreamingJSONError("Transcript passes have different record counts")
    finally:
        for stream in (labels, body, *streams):
            close = getattr(stream, "close", None)
            if close is not None:
                close()


def _prefix_tail(stream: Iterator[bytes], parent: dict,
                 should_cancel: Callable[[], bool]) -> tuple[Iterator[bytes], int, bool]:
    """Skip tokenization, NOT authentication, of one certified byte prefix.

    Consume only through its boundary; retain at most the current archive chunk.
    The returned tail still drains the original whole-file checked iterator.
    """
    remaining = parent["size"]
    digest = hashlib.sha256()
    lines, last, remainder = 0, b"", b""
    while remaining:
        if should_cancel():
            raise ProjectionCancelled("source-unit extraction cancelled")
        try:
            chunk = next(stream)
        except StopIteration as exc:
            raise VaultIntegrityError("Source-unit prefix is truncated") from exc
        prefix = chunk[:remaining]
        digest.update(prefix)
        lines += prefix.count(b"\n")
        if prefix:
            last = prefix[-1:]
        remaining -= len(prefix)
        remainder = chunk[len(prefix):]
    if digest.hexdigest() != parent["sha256"]:
        raise VaultIntegrityError("Source-unit prefix does not match authenticated parent")

    def tail():
        try:
            if remainder:
                yield remainder
            yield from stream
        finally:
            stream.close()

    return tail(), lines, parent["size"] == 0 or last == b"\n"
