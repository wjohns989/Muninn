"""Bounded original ranges from isolated encrypted, authenticated whole units."""
import pytest

from muninn.history.cited_analysis_source import CitedSourceError
from muninn.history.cited_redaction_ranges import unchanged_cited_ranges
from muninn.history.secure_projection_store import ProjectionIntegrityError
from tests.test_cited_analysis_source import fixture


def test_plain_window_is_one_original_contiguous_range(tmp_path):
    _, source, capability = fixture(tmp_path)
    descriptor = source.prepare(capability)
    window = source.reopen(descriptor)
    assert unchanged_cited_ranges(source, descriptor) == [
        {'start': 0, 'text': window['text']}]


def test_suppressed_value_is_a_gap_not_a_fabricated_quote(tmp_path):
    _, source, capability = fixture(
        tmp_path, 'Keep orbital-widget caching. API_KEY=short-value\nKeep SQLite.')
    descriptor = source.prepare(capability)
    original = source.reopen(descriptor)['text']
    ranges = unchanged_cited_ranges(source, descriptor)
    assert len(ranges) == 2
    assert 'short-value' not in ''.join(item['text'] for item in ranges)
    assert ranges[0]['start'] + len(ranges[0]['text']) < ranges[1]['start']
    assert all(original[item['start']:item['start'] + len(item['text'])] == item['text']
               for item in ranges)
    # This primitive does not authorize private-unit provider egress.
    assert source.remote_input(descriptor) is None


def test_long_opaque_value_crossing_selected_window_is_never_partly_retained(tmp_path):
    _, source, capability = fixture(tmp_path, 'orbital ' + 'X' * 12000 + ' End.')
    descriptor = source.prepare(capability)
    assert unchanged_cited_ranges(source, descriptor) == [{'start': 0, 'text': 'orbital '}]


def test_late_authenticated_source_failure_discards_early_ranges(tmp_path, monkeypatch):
    _, source, capability = fixture(tmp_path)
    descriptor = source.prepare(capability)
    original = source.ledger.units.unit_fragments
    def fail_late(*args, **kwargs):
        yield from original(*args, **kwargs)
        raise ProjectionIntegrityError('late isolated authentication failure')
    monkeypatch.setattr(source.ledger.units, 'unit_fragments', fail_late)
    with pytest.raises(CitedSourceError, match='authentication'):
        unchanged_cited_ranges(source, descriptor)


def test_cancellation_during_drain_returns_no_partial_mapping(tmp_path, monkeypatch):
    _, source, capability = fixture(tmp_path, 'orbital ' + 'X' * 12000 + ' End.')
    descriptor = source.prepare(capability)
    calls = []
    def cancel():
        calls.append(True)
        return len(calls) > 2
    with pytest.raises(CitedSourceError, match='cancelled'):
        unchanged_cited_ranges(source, descriptor, should_cancel=cancel)


def test_cancellation_at_authenticated_eof_returns_no_mapping(tmp_path, monkeypatch):
    _, source, capability = fixture(tmp_path, 'Keep orbital decisions. API_KEY=short-value')
    descriptor = source.prepare(capability)
    original = source.ledger.units.unit_fragments
    cancelled = {'value': False}
    def finish_and_cancel(*args, **kwargs):
        yield from original(*args, **kwargs)
        cancelled['value'] = True
    monkeypatch.setattr(source.ledger.units, 'unit_fragments', finish_and_cancel)
    with pytest.raises(CitedSourceError, match='cancelled'):
        unchanged_cited_ranges(source, descriptor, should_cancel=lambda: cancelled['value'])


def test_changed_descriptor_fails_before_range_selection(tmp_path):
    _, source, capability = fixture(tmp_path)
    descriptor = source.prepare(capability)
    with pytest.raises(CitedSourceError):
        unchanged_cited_ranges(source, {**descriptor, 'input_sha256': '0' * 64})


def test_original_page_citation_boundary_never_becomes_one_joined_quote(tmp_path):
    _, source, capability = fixture(tmp_path, 'word ' * 3000 + 'orbital decision.')
    descriptor = source.prepare(capability)
    entry, _, _ = source._window(descriptor)
    _, earlier = source.ledger._source(entry, descriptor['version'], descriptor['attempt'], descriptor['page'] - 1)
    assert len(earlier['text']) >= 500
    boundary = {**descriptor, 'offset': 0, 'length': 500, 'boundary_hit': True,
                'prefix': {'page': descriptor['page'] - 1,
                           'offset': len(earlier['text']) - 500, 'length': 500}}
    window = source._window(boundary)[2]
    boundary['input_sha256'] = source._digest(window)
    assert unchanged_cited_ranges(source, boundary) == [
        {'start': 0, 'text': window['text'][:500]},
        {'start': 500, 'text': window['text'][500:]}]


def test_retained_text_is_bounded_by_selected_window_not_whole_unit(tmp_path):
    _, source, capability = fixture(
        tmp_path, 'orbital decision. ' + 'short words ' * 4000 + 'API_KEY=short-value')
    descriptor = source.prepare(capability)
    original = source.reopen(descriptor)
    selected = unchanged_cited_ranges(source, descriptor)
    assert sum(len(item['text']) for item in selected) <= len(original['text']) <= 3000
    assert len(selected) <= 3000
    assert source.remote_input(descriptor) is None
