"""Original Unicode coordinates, not an additional disclosure permission."""
import pytest

from muninn.history.streaming_redaction import redacted_fragments


def mapped(chunks):
    spans = []
    output = ''.join(redacted_fragments(
        chunks, on_unchanged=lambda text, start, length: spans.append((text, start, length))))
    return output, spans


@pytest.mark.parametrize('text', [
    'Use SQLite. API_KEY=short-value\nKeep the index local.',
    '🔒 Keep café records. password="space separated value"\nNext task.',
    'token: "escaped \\" quoted private value"\nPublic decision.',
    'API_KEY=is another-value; status stored.',
    'API_KEY="unterminated value stays private',
    'Visit C:\\Users\\user\\synthetic-project and user@example.invalid.',
    'literal [REDACTED_SENSITIVE_VALUE] text.',
    'A' * 10000 + '\nEnd of a long opaque value.',
])
@pytest.mark.parametrize('width', [1, 7, 4096])
def test_coordinates_are_original_and_output_is_unchanged(text, width):
    chunks = [text[i:i + width] for i in range(0, len(text), width)]
    output, spans = mapped(chunks)
    assert output == ''.join(redacted_fragments(chunks))
    previous_end = 0
    for value, start, length in spans:
        assert type(start) is int and type(length) is int
        assert start >= previous_end and length == len(value)
        assert text[start:start + length] == value
        previous_end = start + length


@pytest.mark.parametrize('text,secret', [
    ('Before API_KEY=short-value after.', 'short-value'),
    ('Before password="two word value" after.', 'two word value'),
    ('Before API_KEY=is extra-value after.', 'is'),
    ('Before API_KEY=is extra-value after.', 'extra-value'),
    ('Before sk-abcdefghij after.', 'sk-abcdefghij'),
    ('Before ' + 'X' * 10000 + ' after.', 'X' * 10000),
])
def test_observer_never_returns_any_characters_of_suppressed_values(text, secret):
    _, spans = mapped(list(text))
    start = text.index(secret)
    end = start + len(secret)
    assert all(offset + length <= start or offset >= end
               for _, offset, length in spans)


def test_offsets_do_not_reset_between_chunks_or_expand_unicode_scalars():
    text = '💡 café\nAPI_KEY="hidden value"\nKeep SQLite.'
    _, spans = mapped(list(text))
    sqlite = next(item for item in spans if item[0] == 'SQLite.')
    assert sqlite[1] == text.index('SQLite')


def test_callbacks_are_not_proof_of_successful_whole_source_authentication():
    spans = []
    def damaged_source():
        yield 'Early harmless evidence.'
        raise RuntimeError('late source failure')
    with pytest.raises(RuntimeError, match='late source failure'):
        list(redacted_fragments(damaged_source(), on_unchanged=lambda *span: spans.append(span)))
    # A consumer must discard these speculative coordinates after ANY error.
    assert spans


def test_callback_failure_aborts_the_stream():
    def fail(*args):
        raise RuntimeError('consumer cancelled')
    with pytest.raises(RuntimeError, match='consumer cancelled'):
        list(redacted_fragments(['Keep this.'], on_unchanged=fail))


def test_noncallable_observer_fails_before_consuming_any_source():
    consumed = []
    def source():
        consumed.append(True)
        yield 'Keep this.'
    with pytest.raises(TypeError, match='observer'):
        list(redacted_fragments(source(), on_unchanged=False))
    assert consumed == []
