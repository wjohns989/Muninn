"""Synthetic mixed transcripts only; no live secrets, providers or service writes."""
import copy
import json

import pytest

from muninn.history.cited_analysis_source import CitedSourceError
from muninn.history.cited_zdr_projection import CitedZDRProjection
from muninn.history.memory_ledger import MemoryLedger
from muninn.history.secure_projection_store import ProjectionIntegrityError
from tests.test_cited_analysis_source import MODEL, PHRASE, fixture


def mixed(tmp_path):
    archive, source, cap = fixture(tmp_path,
        'Keep orbital caching.\nAPI_KEY=short-value\nKeep SQLite.')
    descriptor = source.prepare(cap)
    projection = CitedZDRProjection(source, descriptor)
    proposal = {'type': 'decision', 'text': 'Keep SQLite.', 'quote': 'Keep SQLite.',
                'start': source.reopen(descriptor)['text'].index('Keep SQLite.')}
    window = projection.reopen(descriptor)
    view = {'policy': 'zdr-ranges-v1', 'window': descriptor,
            'ranges': window['citation_ranges']}
    return archive, source, descriptor, proposal, view


def test_safe_mixed_excerpt_is_public_provisional_and_legacy_stays_withheld(tmp_path):
    import hashlib
    import hmac

    from muninn.history.memory_ledger import POLICY, _json
    archive, source, desc, proposal, view = mixed(tmp_path)
    legacy = source.record_proposals(desc, [proposal], model_identity=MODEL)[0]
    expected = source.expected_refs(desc, [proposal], model_identity=MODEL)
    checked = source.validated_proposals(desc, [proposal])[0]
    entry = source._window(desc)[0]
    unit, page = source.ledger._source(entry, desc['version'], desc['attempt'], checked['page'])
    old_identity = {'blob': entry['blob'], 'sha': entry['sha256'], 'version': desc['version'],
                    'unit': unit.ordinal, 'fragment': page['fragment'], 'proposal': checked['proposal'],
                    'policy': POLICY, 'model': MODEL, 'proposal_origin': 'model'}
    assert legacy == hmac.new(source.ledger._key, b'candidate\0' + _json(old_identity),
                              hashlib.sha256).hexdigest()
    refs = source.record_proposals(desc, [proposal], model_identity=MODEL, source_view=view)
    assert refs == source.expected_refs(desc, [proposal], model_identity=MODEL, source_view=view)
    assert refs[0] != legacy
    assert source.expected_refs(desc, [proposal], model_identity=MODEL) == expected
    ledger = MemoryLedger(archive)
    assert 'text' not in ledger.get(legacy)
    result = ledger.get(refs[0])
    assert result['state'] == 'provisional' and result['proposal_origin'] == 'model'
    assert result['text'] == 'Keep SQLite.'
    assert [r['id'] for r in ledger.search('SQLite')['matches']] == refs
    context = ledger.source(refs[0])
    assert context['context_state'] == 'available'
    assert 'Keep orbital' in context['context'] and 'Keep SQLite.' in context['context']
    assert 'API_KEY' not in json.dumps(context) and 'short-value' not in json.dumps(context)
    assert 'source_view' not in json.dumps(context)
    assert [r['id'] for r in ledger.review_page()['matches']] == refs
    assert source.record_proposals(desc, [proposal], model_identity=MODEL, source_view=view) == refs


@pytest.mark.parametrize('change', ['policy', 'window', 'extra', 'reverse', 'overlap', 'numeric_alias'])
def test_noncanonical_view_cannot_publish(tmp_path, change):
    _, source, desc, proposal, view = mixed(tmp_path)
    bad = copy.deepcopy(view)
    if change == 'policy':
        bad['policy'] = 'unknown'
    elif change == 'window':
        bad['window']['offset'] += 1
    elif change == 'extra':
        bad['unexpected'] = True
    elif change == 'reverse':
        bad['ranges'].reverse()
    elif change == 'numeric_alias':
        bad['window']['format'] = 1.0
    else:
        bad['ranges'].append({'start': 0, 'length': desc['length']})
    with pytest.raises((ValueError, CitedSourceError)):
        source.record_proposals(desc, [proposal], model_identity=MODEL, source_view=bad)
    assert source.ledger.search('SQLite')['matches'] == []


@pytest.mark.parametrize('kind', ['possible_credential', 'unsafe_claim'])
def test_credential_risk_dominates_safe_quote(tmp_path, kind):
    archive, source, desc, proposal, view = mixed(tmp_path)
    if kind == 'possible_credential':
        proposal['type'] = kind
    else:
        proposal['text'] = 'TOKEN=short-value'
    ident = source.record_proposals(desc, [proposal], model_identity=MODEL, source_view=view)[0]
    result = MemoryLedger(archive).get(ident)
    assert result['type'] == 'possible_credential' and 'text' not in result
    assert MemoryLedger(archive).source(ident)['context_state'] == 'withheld'


def test_public_read_rechecks_eof_and_fails_closed(tmp_path, monkeypatch):
    archive, source, desc, proposal, view = mixed(tmp_path)
    ident = source.record_proposals(desc, [proposal], model_identity=MODEL, source_view=view)[0]
    ledger = MemoryLedger(archive)
    original = ledger.units.unit_fragments
    def fail(*args, **kwargs):
        yield from original(*args, **kwargs)
        raise ProjectionIntegrityError('late synthetic failure')
    monkeypatch.setattr(ledger.units, 'unit_fragments', fail)
    assert 'text' not in ledger.get(ident)
    assert ledger.source(ident)['context_state'] == 'withheld'
    assert ledger.search('SQLite')['matches'] == []
    assert ledger.review_page()['matches'] == []
    monkeypatch.setattr(archive, '_unlocked_with_passphrase', True)
    with pytest.raises(ValueError, match='safe noncredential'):
        ledger.resolve_review(ident, state='filed', expected_state='provisional', reason='source_supported')


def test_empty_stage_and_publication_still_require_canonical_proof(tmp_path):
    from muninn.history.capture_journal import CaptureJournal, SearchJobError
    from muninn.history.secure_analysis import _cited_outcome
    archive, source, desc, _, view = mixed(tmp_path)
    projection = CitedZDRProjection(source, desc)
    output = json.dumps({'summary': 'No proposals.', 'decisions': [], 'open_items': [],
                         'uncertainty': '', 'proposals': []})
    stage = _cited_outcome(output, projection, desc, 'openrouter', 'fixture')['extraction']
    journal = CaptureJournal(archive, recover=False)
    assert journal._validate_extraction(stage)['source_view'] == view
    stage['source_view']['ranges'].reverse()
    with pytest.raises(SearchJobError):
        journal._validate_extraction(stage)
    with pytest.raises(CitedSourceError):
        source.record_proposals(desc, [], model_identity=MODEL, source_view=stage['source_view'])


def test_review_passphrase_cas_and_portable_restore(tmp_path, monkeypatch):
    from muninn.history.secure_archive import SecureHistoryArchive
    archive, source, desc, proposal, view = mixed(tmp_path)
    ident = source.record_proposals(desc, [proposal], model_identity=MODEL, source_view=view)[0]
    ledger = source.ledger
    monkeypatch.setattr(archive, '_unlocked_with_passphrase', False)
    with pytest.raises(PermissionError):
        ledger.resolve_review(ident, state='filed', expected_state='provisional', reason='source_supported')
    unlocked = SecureHistoryArchive(archive.root, PHRASE)
    local = MemoryLedger(unlocked)
    local.resolve_review(ident, state='filed', expected_state='provisional', reason='source_supported')
    with pytest.raises(ValueError, match='state changed'):
        local.resolve_review(ident, state='rejected', expected_state='provisional', reason='user_rejected')
    restored = SecureHistoryArchive.restore_from_backup(archive.root, tmp_path / 'restored', PHRASE)
    reopened = MemoryLedger(restored)
    assert reopened.get(ident)['state'] == 'filed'
    assert reopened.get(ident)['truth_status'] == 'model_inferred'
    assert reopened.source(ident)['partial_visible_ranges']
    assert reopened.search('SQLite')['matches'][0]['id'] == ident


def test_projection_preserves_page_boundary_and_original_prefix_citations(tmp_path):
    archive, source, cap = fixture(tmp_path, 'word ' * 3000 + 'orbital decision.\nTOKEN=short-value')
    desc = source.prepare(cap)
    entry = source._window(desc)[0]
    _, earlier = source.ledger._source(entry, desc['version'], desc['attempt'], desc['page'] - 1)
    desc = {**desc, 'offset': 0, 'length': 500, 'boundary_hit': True,
            'prefix': {'page': desc['page'] - 1, 'offset': len(earlier['text']) - 500, 'length': 500}}
    desc['input_sha256'] = source._digest(source._window(desc)[2])
    projection = CitedZDRProjection(source, desc)
    window, view = projection.reopen(desc), projection.source_view()
    refs = []
    for start in (0, 500):
        quote = window['text'][start:start + 10]
        proposal = {'type': 'observation', 'text': quote, 'quote': quote, 'start': start}
        ident = source.record_proposals(desc, [proposal], model_identity=MODEL, source_view=view)[0]
        refs.append(ident)
        assert MemoryLedger(archive).get(ident)['quote'] == quote
        assert MemoryLedger(archive).source(ident)['context_coordinate'] == 'cited_window'
    assert refs[0] != refs[1]  # Same repeated text at two different original citations.
    crossing = {'type': 'observation', 'text': 'Joined.',
                'quote': window['text'][490:510], 'start': 490}
    with pytest.raises(CitedSourceError):
        source.record_proposals(desc, [crossing], model_identity=MODEL, source_view=view)


def test_review_page_never_persists_projection_screen_and_revalidates_ranges(tmp_path, monkeypatch):
    archive, source, desc, proposal, view = mixed(tmp_path)
    ident = source.record_proposals(desc, [proposal], model_identity=MODEL, source_view=view)[0]
    ledger = MemoryLedger(archive)
    monkeypatch.setattr(ledger.units, '_store_screen_info', lambda *a, **kw: pytest.fail('read wrote proof'))
    assert ledger.review_page()['matches'][0]['id'] == ident
    candidate, _ = ledger._read_candidate(ident)
    candidate['source_view']['ranges'].reverse()
    assert ledger._public_projection(candidate) is None


def test_expected_projected_refs_are_read_only_and_drain_unit_once(tmp_path, monkeypatch):
    from muninn.history.cited_analysis_source import CitedAnalysisSource
    archive, source, desc, proposal, view = mixed(tmp_path)
    readonly = CitedAnalysisSource(archive, read_only=True)
    fragments, calls = readonly.ledger.units.unit_fragments, []
    def counted(*args, **kwargs):
        calls.append(True)
        yield from fragments(*args, **kwargs)
    monkeypatch.setattr(readonly.ledger.units, 'unit_fragments', counted)
    monkeypatch.setattr(readonly.ledger.units, '_store_screen_info', lambda *a, **kw: pytest.fail('proof wrote'))
    refs = readonly.expected_refs(desc, [proposal, {**proposal, 'type': 'fact'}],
                                  model_identity=MODEL, source_view=view)
    assert len(refs) == 2 and calls == [True]
    assert source.ledger.verify_all()['events'] == 0
