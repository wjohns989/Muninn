"""Bounded, fail-closed conversational projection of archived JSONL transcripts.

The first authenticated stream classifies one record at a time.  The second
stream visits that same record only after its role is known, emitting selected
text fields without ever retaining the record or a whole message in memory.
The projection store is responsible for redaction, encryption, and publication.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

from muninn.history.streaming_jsonl import StreamingJSONError, events


class UnsupportedTranscript(ValueError):
    """The provider or transcript shape cannot be safely projected."""


class ProjectionCancelled(RuntimeError):
    """A service shutdown cancelled an unpublished projection."""


_META_PATHS = {
    ("type",), ("role",), ("isSidechain",), ("isMeta",),
    ("isCompactSummary",), ("promptSource",), ("origin", "kind"),
    ("payload", "type"), ("payload", "role"), ("payload", "channel"),
}
_BLOCK_TYPES = {"text", "input_text", "output_text"}


def _metadata_units(source: Iterable[bytes], provider: str,
                    should_cancel: Callable[[], bool], *,
                    extra_paths: dict[tuple[str, ...], int] | None = None,
                    ) -> Iterator[tuple[tuple[str | int, ...], dict[tuple[str, ...], str]]]:
    limits = {path: 128 for path in _META_PATHS}
    limits.update(extra_paths or {})
    meta: dict[tuple[str, ...], str] = {}
    active: tuple[str, ...] | None = None
    value_kind = ""
    value = ""
    container_messages = False
    item_prefix: tuple[str | int, ...] | None = None
    seen: set[tuple[str | int, ...]] = set()
    for event, path, part in events(source, include_record_position=True,
                                    allow_multiline=provider == "gemini_cli"):
        if should_cancel():
            raise ProjectionCancelled("projection cancelled")
        if provider == "gemini_cli" and event == "container_start" and path == ("messages",):
            if part != "array":
                raise UnsupportedTranscript("Gemini messages container is not an array")
            container_messages = True
        if (container_messages and event == "container_start" and len(path) == 2
                and path[0] == "messages" and isinstance(path[1], int)):
            if part != "object":
                raise UnsupportedTranscript("Gemini message item is not an object")
            item_prefix = path
            meta = {}
            seen = set()
        if (container_messages and event == "value_start" and len(path) == 2
                and path[0] == "messages" and isinstance(path[1], int)):
            raise UnsupportedTranscript("Gemini message item is not an object")
        relative = path[len(item_prefix):] if item_prefix and path[:len(item_prefix)] == item_prefix else (
            path if not container_messages else None)
        if event in {"value_start", "container_start"} and relative in limits:
            if relative in seen:
                raise StreamingJSONError("Repeated transcript metadata key")
            seen.add(relative)
        if event == "value_start":
            active = relative if relative in limits else None
            value_kind = part
            value = ""
        elif event == "value_chunk" and active is not None:
            bound = limits[active]
            if len(value) <= bound:
                value += part[:bound + 1 - len(value)]
        elif event == "value_end" and active is not None:
            if active in meta and meta[active] != value:
                raise StreamingJSONError("Conflicting transcript metadata")
            meta[active] = value
            if extra_paths and active in extra_paths:
                kind_path = ("__value_kind__",) + active
                if kind_path in meta and meta[kind_path] != value_kind:
                    raise StreamingJSONError("Conflicting transcript metadata type")
                meta[kind_path] = value_kind
            active = None
        elif event == "record_end":
            if not container_messages:
                meta[("__physical_line__",)] = part
                yield (), meta
            meta = {}
            seen = set()
        elif event == "container_end" and item_prefix is not None and path == item_prefix:
            yield item_prefix, meta
            item_prefix = None
            meta = {}
            seen = set()


def _role(provider: str, meta: dict[tuple[str, ...], str]) -> str | None:
    kind = meta.get(("type",))
    if provider == "codex":
        payload_type = meta.get(("payload", "type"))
        if kind == "event_msg":
            return {"user_message": "User", "agent_message": "Assistant"}.get(payload_type)
        if kind == "response_item" and payload_type == "message":
            if meta.get(("payload", "channel")) not in (None, "final", "commentary"):
                return None
            return {"user": "User", "assistant": "Assistant"}.get(meta.get(("payload", "role")))
        return None
    if provider == "claude_code":
        if (meta.get(("isSidechain",)) == "true" or meta.get(("isMeta",)) == "true"
                or meta.get(("isCompactSummary",)) == "true"
                or meta.get(("promptSource",)) == "system"
                or meta.get(("origin", "kind")) not in (None, "human")):
            return None
        return {"user": "User", "assistant": "Assistant"}.get(kind)
    if provider == "gemini_cli":
        return {"user": "User", "assistant": "Assistant", "gemini": "Assistant",
                "model": "Assistant"}.get(kind or meta.get(("role",)))
    raise UnsupportedTranscript("Unsupported transcript provider")


def _selected(provider: str, kind: str | None, path: tuple[str | int, ...],
              block_type: str | None) -> bool:
    if provider == "codex":
        if kind == "event_msg":
            return path == ("payload", "message")
        if path == ("payload", "content"):
            return True
        return (len(path) == 4 and path[:2] == ("payload", "content")
                and isinstance(path[2], int) and path[3] == "text"
                and block_type in _BLOCK_TYPES)
    if provider == "claude_code":
        if path == ("message", "content"):
            return True
        return (len(path) == 4 and path[:2] == ("message", "content")
                and isinstance(path[2], int) and path[3] == "text" and block_type == "text")
    if provider == "gemini_cli":
        if path == ("content",):
            return True
        if len(path) == 2 and path[0] == "parts" and isinstance(path[1], int):
            return True
        return (len(path) == 3 and path[0] in {"content", "parts"}
                and isinstance(path[1], int) and path[2] == "text"
                and block_type == "text")
    return False


def _project_record(stream: Iterator[tuple[str, tuple[str | int, ...], str]],
                    provider: str, kind: str | None, role: str | None,
                    prefix: tuple[str | int, ...],
                    should_cancel: Callable[[], bool]) -> Iterator[str]:
    active: tuple[str | int, ...] | None = None
    active_block: tuple[str | int, ...] | None = None
    block_type: str | None = None
    type_value = ""
    type_path: tuple[str | int, ...] | None = None
    selected = False
    emitted = False
    for event, path, part in stream:
        if should_cancel():
            raise ProjectionCancelled("projection cancelled")
        if (event == "record_end" and not prefix
                or event == "container_end" and prefix and path == prefix):
            if active is not None:
                raise StreamingJSONError("Unclosed transcript value")
            return emitted
        relative = path[len(prefix):] if path[:len(prefix)] == prefix else None
        if event == "value_start":
            active = relative
            selected = False
            type_path = None
            if relative is not None and len(relative) >= 3 and isinstance(relative[-2], int):
                block = relative[:-1]
                if block != active_block:
                    active_block, block_type = block, None
                if relative[-1] == "type":
                    type_path = relative
                    type_value = ""
            if role and part == "string" and relative is not None and _selected(
                    provider, kind, relative, block_type):
                selected = True
                if not emitted:
                    yield f"\n\n{role}: "
                    emitted = True
                else:
                    yield "\n"
        elif event == "value_chunk":
            if type_path == relative:
                type_value += part[:33 - len(type_value)]
            if selected:
                yield part
        elif event == "value_end":
            if type_path == relative:
                if block_type is not None and block_type != type_value:
                    raise StreamingJSONError("Conflicting transcript block type")
                block_type = type_value
            active = None
            selected = False
            type_path = None
    raise StreamingJSONError("Truncated transcript record")


def project_transcript(archive: Any, entry: dict[str, Any],
                       source: Iterable[bytes], *,
                       stats: dict[str, int] | None = None,
                       should_cancel: Callable[[], bool] = lambda: False) -> Iterator[str]:
    """Yield only supported user/assistant text, including oversized JSONL rows.

    Gemini JSONL and container JSON are supported. Unknown schemas never fall
    back to returning raw source or tool output.
    """
    if entry.get("kind") != "transcript" or entry.get("provider") not in {
            "codex", "claude_code", "gemini_cli"}:
        raise UnsupportedTranscript("Unsupported transcript snapshot")
    provider = entry["provider"]
    first_source = archive._iter_verified_entry(entry)
    labels = _metadata_units(first_source, provider, should_cancel)
    body = iter(events(source, allow_multiline=provider == "gemini_cli"))
    saw_container = False
    try:
        while True:
            try:
                prefix, meta = next(labels)
            except StopIteration:
                # Container JSON still has a root closing token after the
                # final message. Drain it, but reject any extra body unit.
                for event, path, part in body:
                    if should_cancel():
                        raise ProjectionCancelled("projection cancelled")
                    if event == "container_start" and path == ("messages",) and part == "array":
                        saw_container = True
                    if (event == "container_end" and len(path) == 2
                            and path[0] == "messages" and isinstance(path[1], int)):
                        raise StreamingJSONError("Transcript passes have different message counts")
                    if event == "record_end" and not saw_container:
                        raise StreamingJSONError("Transcript passes have different record counts")
                return
            if prefix:
                saw_container = True
            role = _role(provider, meta)
            emitted = yield from _project_record(
                body, provider, meta.get(("type",)), role, prefix, should_cancel)
            if stats is not None:
                stats["source_units"] += 1
                stats["conversational_units" if emitted else "omitted_units"] += 1
    finally:
        labels.close()
        body.close()
        for stream in (first_source, source):
            close = getattr(stream, "close", None)
            if close is not None:
                close()


def build_transcript_projection(store: Any, entry: dict[str, Any], version: int, *,
                                should_cancel: Callable[[], bool] = lambda: False) -> str:
    """Build an encrypted, redacted, fully authenticated transcript projection."""
    stats = {"source_units": 0, "conversational_units": 0, "omitted_units": 0}
    return store.build(entry, version,
                       lambda source: project_transcript(
                           store.archive, entry, source, stats=stats, should_cancel=should_cancel),
                       stats=stats)
