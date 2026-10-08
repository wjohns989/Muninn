"""Chunk and page boundaries must not expose partial credential values."""

from __future__ import annotations

import pytest

from muninn.history.streaming_redaction import redacted_pages


def _text(chunks, page_chars=4000):
    pages = list(redacted_pages(chunks, page_chars=page_chars))
    assert all(0 < len(page) <= page_chars for page in pages)
    return "".join(pages), pages


def test_assignment_crossing_input_and_page_boundaries() -> None:
    raw = ["before " + "a" * 17 + " api_", "key", " = ", "short", "-secret", " after"]
    safe, pages = _text(raw, page_chars=13)
    assert "short" not in safe and "secret" not in safe
    assert "before" in safe and "after" in safe
    assert "api_key" in safe and "[REDACTED_SENSITIVE_VALUE]" in safe
    assert len(pages) > 2


def test_long_opaque_value_crossing_many_chunks_releases_no_prefix() -> None:
    opaque = "AbCd1234_-" * 10000
    safe, pages = _text(["start ", *[opaque[i:i + 3] for i in range(0, len(opaque), 3)], " end"],
                        page_chars=7)
    assert opaque[:27] not in safe and opaque[-27:] not in safe
    assert safe == "start [REDACTED_SENSITIVE_VALUE] end"
    assert len(pages) > 1


def test_benign_multi_megabyte_message_has_no_total_cutoff() -> None:
    chunks = ("ordinary note " for _ in range(160000))
    length = 0
    count = 0
    for page in redacted_pages(chunks):
        assert len(page) <= 4000
        length += len(page)
        count += 1
    assert length == 160000 * len("ordinary note ") and count > 400


@pytest.mark.parametrize("chunks", [
    ["Authorization: Bea", "rer abc123"],
    ["password:", "\"", "tiny123", "\""],
    ["service gh", "p_ABC123456"],
    ["C:\\Us", "ers\\wjohn\\secret"],
    ["api-key=", "tiny123"],
])
def test_other_sensitive_forms_do_not_escape(chunks) -> None:
    safe, _ = _text(chunks, page_chars=5)
    assert "abc123" not in safe
    assert "tiny123" not in safe
    assert "ghp_ABC123456" not in safe
    assert "C:\\Users\\wjohn" not in safe


def test_newline_does_not_redact_next_unrelated_message() -> None:
    safe, _ = _text(["api_key\nordinary text"])
    assert safe == "api_key\nordinary text"


def test_preserves_location_metadata_without_a_value() -> None:
    text = "The OpenRouter API key is stored in the user environment; use Ollama locally."
    safe, _ = _text([text[:23], text[23:]], page_chars=17)
    assert safe == text


def test_assignment_does_not_treat_benign_word_as_safe_value() -> None:
    safe, _ = _text(["api_key=stored and continue"])
    assert safe == "api_key=[REDACTED_SENSITIVE_VALUE] and continue"


@pytest.mark.parametrize("raw", [
    "api_key = the abc123 after",
    "api_key = a abc123 after",
    "api_key = an abc123 after",
    "api_key = value abc123 after",
    "api_key = equals abc123 after",
])
def test_assignment_connectors_cannot_expose_following_secret(raw: str) -> None:
    for width in (1, 2, 5, 11):
        safe, _ = _text((raw[index:index + width] for index in range(0, len(raw), width)),
                        page_chars=9)
        assert "abc123" not in safe
        assert "[REDACTED_SENSITIVE_VALUE]" in safe


@pytest.mark.parametrize("raw", [
    'api_key="two words" then continue',
    '{"api_key": "two words"} then continue',
    'Authorization: Bearer tiny123 then continue',
    'api-key=short-secret then continue',
    'The OpenRouter API key is stored in the user environment.',
])
def test_output_is_independent_of_input_chunk_boundaries(raw: str) -> None:
    expected, _ = _text([raw], page_chars=11)
    for width in range(1, min(len(raw), 19)):
        chunks = (raw[index:index + width] for index in range(0, len(raw), width))
        actual, _ = _text(chunks, page_chars=11)
        assert actual == expected


@pytest.mark.parametrize("chunks", [
    ["password=\"two", " words with a token\" then safe"],
    ["{\"api_", "key\": \"two words\"} then safe"],
    ["Authorization: Bearer '", "short token", "' then safe"],
])
def test_quoted_values_are_suppressed_through_the_closing_quote(chunks) -> None:
    safe, _ = _text(chunks, page_chars=9)
    assert "two" not in safe and "words" not in safe and "short token" not in safe
    assert "then safe" in safe
