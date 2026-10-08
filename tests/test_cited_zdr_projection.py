"""Isolated exact-coordinate ZDR projection; no live inference or vault access."""
import json

import pytest

from muninn.history import secure_analysis as analysis
from muninn.history.cited_analysis_source import CitedSourceError
from muninn.history.cited_zdr_projection import CitedZDRProjection
from muninn.history.secure_projection_store import ProjectionIntegrityError
from tests.test_cited_analysis_source import fixture
from tests.test_cited_analysis_transport import transport


def projected(tmp_path):
    _, source, cap = fixture(tmp_path, 'Keep orbital caching.\nAPI_KEY=short-value\nKeep SQLite.')
    descriptor = source.prepare(cap)
    return source, descriptor, CitedZDRProjection(source, descriptor)


def test_projected_text_retains_original_coordinates_and_metadata(tmp_path):
    source, descriptor, projection = projected(tmp_path)
    raw = source.reopen(descriptor)
    window = projection.reopen(descriptor)
    assert len(window['text']) == len(raw['text']) <= 3000
    assert 'short-value' not in json.dumps(window)
    assert 'API_KEY' not in window['text']
    assert window['event_at'] == raw['event_at']
    assert window['project_ref'] == raw['project_ref']
    assert window['projection_policy'] == 'zdr-ranges-v1'
    for span in window['citation_ranges']:
        start, length = span['start'], span['length']
        assert window['text'][start:start + length] == raw['text'][start:start + length]
    assert projection.remote_input(descriptor) == window
    assert source.remote_input(descriptor) is None  # Batch/public policy unchanged.


def test_valid_quote_stays_bound_to_original_not_compacted_projection(tmp_path):
    source, descriptor, projection = projected(tmp_path)
    raw = source.reopen(descriptor)
    proposal = {'type': 'decision', 'text': 'Keep SQLite.', 'quote': 'Keep SQLite.',
                'start': raw['text'].index('Keep SQLite.')}
    assert projection.validated_proposals(descriptor, [proposal]) == source.validated_proposals(descriptor, [proposal])


def test_cross_gap_quote_cannot_acquire_evidence(tmp_path):
    _, descriptor, projection = projected(tmp_path)
    text = projection.reopen(descriptor)['text']
    with pytest.raises(CitedSourceError, match='admitted'):
        projection.validated_proposals(descriptor, [{'type': 'decision', 'text': 'Combined.',
            'quote': text, 'start': 0}])


def test_whitespace_is_not_evidence(tmp_path):
    _, descriptor, projection = projected(tmp_path)
    with pytest.raises(CitedSourceError):
        projection.validated_proposals(descriptor, [{'type': 'decision', 'text': 'Blank.',
            'quote': ' ', 'start': 4}])


def test_descriptor_and_returned_projection_cannot_be_mutated(tmp_path):
    _, descriptor, projection = projected(tmp_path)
    result = projection.reopen(descriptor)
    result['text'] = 'injected'
    result['citation_ranges'].clear()
    assert projection.reopen(descriptor)['citation_ranges']
    with pytest.raises(CitedSourceError):
        projection.reopen({**descriptor, 'input_sha256': '0' * 64})


def test_late_authenticated_failure_cannot_construct_projection(tmp_path, monkeypatch):
    _, source, cap = fixture(tmp_path)
    descriptor = source.prepare(cap)
    original = source.ledger.units.unit_fragments
    def fail(*args, **kwargs):
        yield from original(*args, **kwargs)
        raise ProjectionIntegrityError('isolated late failure')
    monkeypatch.setattr(source.ledger.units, 'unit_fragments', fail)
    with pytest.raises(CitedSourceError):
        CitedZDRProjection(source, descriptor)


def test_cancellation_after_projection_prevents_admission(tmp_path):
    _, source, cap = fixture(tmp_path)
    descriptor = source.prepare(cap)
    cancelled = False
    projection = CitedZDRProjection(source, descriptor, should_cancel=lambda: cancelled)
    cancelled = True
    with pytest.raises(CitedSourceError, match='cancelled'):
        projection.remote_input(descriptor)


def test_private_path_and_adjacent_values_not_retained(tmp_path):
    _, source, cap = fixture(tmp_path,
        'Keep orbital caching.\nC:\\Users\\user\\synthetic-private\\secret.env\nTOKEN=short-value\nKeep SQLite.')
    descriptor = source.prepare(cap)
    window = CitedZDRProjection(source, descriptor).remote_input(descriptor)
    assert window is not None
    assert 'synthetic-private' not in window['text'] and 'short-value' not in window['text']
    assert 'Keep SQLite.' in window['text']


@pytest.mark.asyncio
@pytest.mark.usefixtures('fake_strict_remote_admission')
async def test_explicit_zdr_route_never_probes_local_or_sends_removed_values(tmp_path, monkeypatch):
    source, descriptor, projection = projected(tmp_path)
    raw = source.reopen(descriptor)
    proposal = {'type': 'decision', 'text': 'Keep SQLite.', 'quote': 'Keep SQLite.',
                'start': raw['text'].index('Keep SQLite.')}
    result = {'summary': 'SQLite requested.', 'decisions': [], 'open_items': [], 'uncertainty': 'Unverified.'}
    stage = {'window': descriptor, 'result': {'analysis': result}, 'proposals': [proposal]}
    seen, _ = transport(monkeypatch, source, stage, local=False)
    monkeypatch.setattr(analysis, '_select_local', lambda *a: pytest.fail('local probe'))
    output = await analysis.analyze_cited_window(object(), source, descriptor,
        private_zdr=True, allow_remote=True, prefer_remote=True)
    assert output['status'] == 'ok' and len(seen) == 1
    body = seen[0][1]
    assert body['provider'] == {'zdr': True, 'data_collection': 'deny', 'require_parameters': True}
    assert 'short-value' not in json.dumps(body)
    assert json.loads(body['messages'][1]['content'])['text'] == projection.reopen(descriptor)['text']
    assert output['extraction']['proposals'] == [proposal]
    assert output['extraction']['window'] == descriptor
    raw_identity = analysis._cited_model_identity(raw, 'openrouter', 'fixture-model')
    assert output['extraction']['model_identity'] != raw_identity
    assert source.ledger.verify_all()['candidates'] == 0  # Dispatch isn't publication.


@pytest.mark.asyncio
@pytest.mark.parametrize('kwargs', [{}, {'allow_remote': True}, {'prefer_remote': True},
                                 {'allow_remote': True, 'prefer_remote': True, 'private_zdr': 1}])
async def test_private_route_requires_explicit_remote_only_authority(tmp_path, kwargs):
    source, descriptor, _ = projected(tmp_path)
    with pytest.raises(ValueError):
        await analysis.analyze_cited_window(object(), source, descriptor,
            **{'private_zdr': True, **kwargs})


@pytest.mark.asyncio
@pytest.mark.usefixtures('fake_strict_remote_admission')
async def test_default_remote_route_still_refuses_mixed_unit(tmp_path, monkeypatch):
    source, descriptor, _ = projected(tmp_path)
    stage = {'window': descriptor, 'result': {'analysis': {
        'summary': 'Context.', 'decisions': [], 'open_items': [], 'uncertainty': ''}}, 'proposals': []}
    seen, _ = transport(monkeypatch, source, stage, local=False)
    outcome = await analysis.analyze_cited_window(object(), source, descriptor,
        allow_remote=True, prefer_remote=True)
    assert outcome['reason'] == 'source_not_remote_safe' and not seen


def test_result_scrub_uses_local_original_values_and_discloses_partial_view(tmp_path):
    _, descriptor, projection = projected(tmp_path)
    content = json.dumps({'summary': 'Untrusted guessed short-value.',
        'decisions': [], 'open_items': [], 'uncertainty': '', 'proposals': []})
    result = analysis._cited_outcome(content, projection, descriptor, 'openrouter', 'fixture-model')
    assert 'short-value' not in json.dumps(result)
    assert 'Partial visible ranges only' in result['analysis']['uncertainty']
