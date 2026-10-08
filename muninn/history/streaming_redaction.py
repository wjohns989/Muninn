"""Bounded, conservative redaction before encrypted transcript pagination.

This is an additional privacy boundary, not a credential classifier or a
substitute for the authenticated local credential vault. No partial token is
released until its full lexical extent has been classified, even when input
chunks or output pages split inside that token.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator

# Bump on any change to whole-unit redaction semantics. Persisted screening
# attestations must never carry admission forward under a different policy.
UNIT_SCREEN_VERSION = 1

_OPAQUE_LIMIT = 28
_REDACTED = "[REDACTED_SENSITIVE_VALUE]"
_SENSITIVE = re.compile(
    r"(?i)(?:^|_)(?:api_key|access_key|secret_key|private_key|access_token|"
    r"refresh_token|auth_token|client_secret|token|secret|password|passwd|"
    r"credential|authorization|bearer|cookie|key)$"
)
_CONNECTORS = frozenset({"is", "was", "the", "a", "an", "value", "equals", "bearer"})
_BENIGN_STATUS = frozenset({"stored", "configured", "missing", "available", "not", "in", "at"})
_SECRET_PREFIXES = ("sk-", "ghp_", "gho_", "github_pat_", "hf_", "xoxb-", "xoxp-")
_WINDOWS_HOME = re.compile(r"(?i)^(?:[a-z]:)?\\users\\")


def _token_char(char: str) -> bool:
    return char.isalnum() or char in "_-/+.@\\"


def redacted_fragments(chunks: Iterable[str], *,
                       on_unchanged: Callable[[str, int, int], None] | None = None) -> Iterator[str]:
    """Redact complete lexical tokens with constant retained text memory.

    A long opaque token is replaced after 28 characters and the remainder is
    discarded until its delimiter. Assignment labels are retained while their
    next significant value is replaced. This errs toward suppressing context
    when an unlabeled value cannot be safely distinguished from prose.

    Optional callbacks describe unchanged runs at ORIGINAL Unicode character
    offsets, never substituted markers or suppressed tokens. They are NOT an
    egress permission or a credential classification. Coordinates remain
    speculative until the entire source iterator finishes successfully; callers
    must discard them after cancellation, authentication or any other failure.
    The callback must retain only bounded, selected ranges for large sources.
    """
    if on_unchanged is not None and not callable(on_unchanged):
        raise TypeError("Invalid original-coordinate observer")
    token: list[str] = []
    long_token = False
    pending_value = False
    pending_assignment = False
    pending_key_quote: str | None = None
    recent_quote: str | None = None
    token_open_quote: str | None = None
    quoted_value: str | None = None
    quoted_escape = False
    original_offset = 0
    token_start = 0
    token_length = 0
    finished_start = 0
    finished_length = 0

    def finish() -> str:
        nonlocal long_token, pending_value, pending_assignment, pending_key_quote, token_open_quote
        nonlocal token_length, finished_start, finished_length
        if not token and not long_token:
            return ""
        finished_start, finished_length = token_start, token_length
        token_length = 0
        value = "".join(token)
        token.clear()
        was_long = long_token
        long_token = False
        lowered = value.casefold()
        if was_long or lowered.startswith(_SECRET_PREFIXES) or _WINDOWS_HOME.match(value) \
                or ("@" in value and "." in value):
            pending_value = False
            pending_assignment = False
            pending_key_quote = None
            token_open_quote = None
            return _REDACTED
        if pending_value:
            if not pending_assignment and lowered in _BENIGN_STATUS:
                pending_value = False
                token_open_quote = None
                return value
            if lowered in _CONNECTORS:
                token_open_quote = None
                # After '=' or ':' even an innocuous-looking connector could
                # itself be a short credential value. Suppress it and keep the
                # state until a non-connector value is also suppressed.
                return _REDACTED if pending_assignment else value
            pending_value = False
            pending_assignment = False
            pending_key_quote = None
            token_open_quote = None
            return _REDACTED
        if _SENSITIVE.search(lowered.replace("-", "_")):
            pending_value = True
            pending_assignment = False
            pending_key_quote = token_open_quote
            token_open_quote = None
            return value
        token_open_quote = None
        return value

    for chunk in chunks:
        if not isinstance(chunk, str):
            raise TypeError("Transcript text chunks must be strings")
        for char in chunk:
            char_offset = original_offset
            original_offset += 1
            if quoted_value is not None:
                if quoted_escape:
                    quoted_escape = False
                elif char == "\\":
                    quoted_escape = True
                elif char == quoted_value:
                    quoted_value = None
                continue
            if _token_char(char):
                if not token and not long_token:
                    token_open_quote = recent_quote
                    token_start = char_offset
                token_length += 1
                recent_quote = None
                if not long_token:
                    token.append(char)
                    if len(token) >= _OPAQUE_LIMIT:
                        token.clear()
                        long_token = True
            else:
                value = finish()
                if value:
                    if on_unchanged is not None and value != _REDACTED:
                        on_unchanged(value, finished_start, finished_length)
                    yield value
                if pending_key_quote == char:
                    pending_key_quote = None
                    if on_unchanged is not None:
                        on_unchanged(char, char_offset, 1)
                    yield char
                elif pending_value and char in "\"'":
                    pending_value = False
                    pending_assignment = False
                    pending_key_quote = None
                    quoted_value = char
                    yield _REDACTED
                else:
                    if on_unchanged is not None:
                        on_unchanged(char, char_offset, 1)
                    yield char
                if pending_value and char in "=:":
                    pending_assignment = True
                recent_quote = char if char in "\"'" else None
                if char in "\r\n":
                    pending_value = False
                    pending_assignment = False
                    pending_key_quote = None
    value = finish()
    if value:
        if on_unchanged is not None and value != _REDACTED:
            on_unchanged(value, finished_start, finished_length)
        yield value


def redacted_pages(chunks: Iterable[str], *, page_chars: int = 4000) -> Iterator[str]:
    """Yield every redacted character in pages, without a total-size cutoff."""
    if not 1 <= page_chars <= 4000:
        raise ValueError("Invalid redacted transcript page size")
    page: list[str] = []
    length = 0
    for fragment in redacted_fragments(chunks):
        while fragment:
            room = page_chars - length
            head, fragment = fragment[:room], fragment[room:]
            page.append(head)
            length += len(head)
            if length == page_chars:
                yield "".join(page)
                page.clear()
                length = 0
    if page:
        yield "".join(page)
