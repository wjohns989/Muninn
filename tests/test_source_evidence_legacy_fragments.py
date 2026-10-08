"""Synthetic legacy envelopes: preserve authenticated data, bound emitted pieces."""
import hashlib
import json
from dataclasses import asdict, replace

import pytest

from muninn.history.secure_projection_store import ProjectionIntegrityError
from muninn.history.source_evidence import SourceEvidenceStore
from muninn.history.transcript_units import transcript_units
from tests.test_source_evidence import _fixture


def legacy_fixture(tmp_path, *, damage=None, text=None, legacy_break=None):
    archive, entry = _fixture(tmp_path, text or 'x' * 4095 + '\n' + '\U0001f642')
    store = SourceEvidenceStore(archive)
    stats = {'source_units': 2, 'conversational_units': 1, 'omitted_units': 1}

    def old_projector(source):
        # Mimic the old escaped-character flush bug, not altered original text.
        parts = list(transcript_units(archive, entry, source))
        unit = parts[-1].unit
        body = ''.join(part.text for part in parts[2:])
        rows = [(parts[0].unit, '', True), (unit, parts[1].text, False)]
        if legacy_break:
            rows.extend([(unit, body[:legacy_break], False), (unit, body[legacy_break:], False)])
        else:
            rows.append((unit, body, False))
        rows.append((unit, '', True))
        fragment = 0
        for ordinal, (metadata, text, final) in enumerate(rows):
            if damage == 'metadata' and ordinal == 3:
                metadata = replace(metadata, cwd='C:/different-synthetic')
            if damage == 'final_text' and ordinal == 3:
                text = 'invalid final'
            number = fragment + 1 if damage == 'fragment' and ordinal == 2 else fragment
            yield json.dumps({'unit': asdict(metadata), 'fragment': number,
                              'text': text, 'final': final}, ensure_ascii=False)
            fragment = 0 if final else fragment + 1

    attempt = store.build(entry, 0, old_projector, stats=stats)
    return archive, entry, store, attempt


def test_legacy_authenticated_physical_pages_are_preserved_without_rewriting(tmp_path):
    archive, entry, store, attempt = legacy_fixture(tmp_path)
    before = hashlib.sha256(store.db_path.read_bytes()).digest()
    full = list(store.fragments(entry, 0, attempt))
    selected = list(store.unit_fragments(entry, 0, attempt, 1))
    expected = list(transcript_units(archive, entry))
    assert max(len(part.text) for part in full + selected) == 4097
    assert full[2] == selected[1]  # Physical page IDs must not shift.
    assert ''.join(part.text for part in full) == ''.join(part.text for part in expected)
    assert ''.join(part.text for part in selected) == ''.join(part.text for part in expected if part.unit.ordinal == 1)
    assert [part.unit for part in full if part.final] == [part.unit for part in expected if part.final]
    assert sum(part.final for part in selected) == 1
    assert len(full) == store.count_pages(entry, 0, attempt)
    assert store.verify_all() == {'snapshots': 1, 'units': 2, 'fragments': 4}
    assert hashlib.sha256(store.db_path.read_bytes()).digest() == before


def test_legacy_screen_and_portable_restore_keep_original_references(tmp_path):
    archive, entry, store, attempt = legacy_fixture(tmp_path)
    unit = list(transcript_units(archive, entry))[-1].unit
    store._store_screen_info(entry, 0, attempt, unit, raw_sha='a' * 64,
                             screened_sha='a' * 64, body_length=4097, body_sha='b' * 64)
    ref = store._screen_ref(store._screen_binding(entry, 0, attempt, unit))
    original = hashlib.sha256(store.db_path.read_bytes()).digest()
    with store._connect() as db:
        sealed_rows = {table: db.execute(f'SELECT * FROM {table} ORDER BY rowid').fetchall()
                       for table in ('attempts', 'pages', 'unit_screens')}
    restored = archive.restore_from_backup(archive.root, tmp_path / 'restored',
                                           'synthetic portable recovery phrase')
    recovered = SourceEvidenceStore(restored)
    assert recovered.find_snapshot(entry, 0) == attempt
    assert recovered.verify_all()['fragments'] == 4
    assert recovered._screen_ref(recovered._screen_binding(entry, 0, attempt, unit)) == ref
    assert recovered.screen_info(entry, 0, attempt, unit) == (True, 4097, bytes.fromhex('b' * 64))
    assert hashlib.sha256(store.db_path.read_bytes()).digest() == original
    with recovered._connect() as db:
        assert {table: db.execute(f'SELECT * FROM {table} ORDER BY rowid').fetchall()
                for table in sealed_rows} == sealed_rows


@pytest.mark.parametrize('damage', ['fragment', 'metadata', 'final_text'])
def test_legacy_normalization_never_accepts_bad_sequence_or_metadata(tmp_path, damage):
    _archive, entry, store, attempt = legacy_fixture(tmp_path, damage=damage)
    for read in (lambda: store.fragments(entry, 0, attempt),
                 lambda: store.unit_fragments(entry, 0, attempt, 1)):
        with pytest.raises(ProjectionIntegrityError):
            list(read())
    with pytest.raises(ProjectionIntegrityError):
        store.verify_all()


@pytest.mark.parametrize('damage', ['oversized', 'ordinal', 'metadata', 'final_text', 'missing_final', 'type'])
def test_writer_rejects_invalid_unit_fragments_before_sealing(tmp_path, monkeypatch, damage):
    from muninn.history import source_evidence
    archive, entry = _fixture(tmp_path)
    store = SourceEvidenceStore(archive)
    original = source_evidence.transcript_units

    def broken(*args, **kwargs):
        for part in original(*args, **kwargs):
            if part.unit.ordinal == 1 and not part.final:
                if damage == 'oversized':
                    part = replace(part, text='x' * 4097)
                elif damage == 'ordinal':
                    part = replace(part, unit=replace(part.unit, ordinal=2))
                elif damage == 'metadata':
                    part = replace(part, unit=replace(part.unit, cwd='C:/changed-synthetic'))
                elif damage == 'type':
                    part = replace(part, final=1)
            if part.final and part.unit.ordinal == 1:
                if damage == 'final_text':
                    part = replace(part, text='invalid')
                elif damage == 'missing_final':
                    continue
            yield part

    monkeypatch.setattr(source_evidence, 'transcript_units', broken)
    with pytest.raises(ProjectionIntegrityError):
        store.build_snapshot(entry, 0)
    with store._connect() as db:
        assert db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0] == 0
        assert db.execute('SELECT COUNT(*) FROM pages').fetchone()[0] == 0


def test_legacy_envelope_limit_is_still_enforced_before_json_decode(tmp_path):
    _archive, _entry, store, _attempt = legacy_fixture(tmp_path)
    stats = {'source_units': 1, 'conversational_units': 1, 'omitted_units': 0}
    with pytest.raises(ProjectionIntegrityError):
        list(store._decoded_fragments([' ' * (store.max_page_chars + 1)], 1, stats))


def test_legacy_cited_windows_and_candidate_keep_physical_coordinates(tmp_path):
    from muninn.history.cited_windows import CitedWindowPlanStore
    from muninn.history.memory_ledger import MemoryLedger
    text = 'safe words ' * 460 + 'tail-marker'
    archive, entry, units, attempt = legacy_fixture(tmp_path, text=text)
    plans = CitedWindowPlanStore(archive)
    plan = plans.build_snapshot(entry, 0)
    count = plans.count_pages(entry, 0, plan)
    descriptors = [plans.window_at(entry, 0, plan, i) for i in range(count)]
    assert len(descriptors) == 2 and {d['page'] for d in descriptors} == {2}
    windows = [plans.source.reopen(d) for d in descriptors]
    assert ''.join(w['text'] for w in windows) == text
    assert max(len(w['text']) for w in windows) <= 3000
    assert plans.source.remote_input(descriptors[-1]) == windows[-1]
    assert MemoryLedger(archive).remote_input(entry, 0, attempt, 2) is None
    quote = 'tail-marker'
    refs = plans.source.record_proposals(descriptors[-1], [{
        'type': 'observation', 'text': quote, 'quote': quote,
        'start': windows[-1]['text'].index(quote)}], model_identity='a' * 64)
    assert plans.source.ledger.get(refs[0])['state'] == 'provisional'
    assert plans.source.ledger.verify_all()['candidates'] == 1
    assert plans.verify_all()['windows'] == 2
    # The end marker is the original following physical page, not a shifted ID.
    assert json.loads(units.get_page(entry, 0, attempt, 3))['final'] is True


def test_legacy_secret_elsewhere_in_same_unit_still_blocks_cited_egress(tmp_path):
    from muninn.history.cited_windows import CitedWindowPlanStore
    text = 'harmless words ' * 310 + ' SERVICE_API_KEY=synthetic$secret'
    archive, entry, _units, _attempt = legacy_fixture(tmp_path, text=text)
    plans = CitedWindowPlanStore(archive)
    plan = plans.build_snapshot(entry, 0)
    descriptor = plans.window_at(entry, 0, plan, 0)
    assert 'synthetic$secret' not in plans.source.reopen(descriptor)['text']
    assert plans.source.remote_input(descriptor) is None


def test_legacy_parent_appends_into_new_bounded_attempt_without_changing_parent(tmp_path):
    archive, parent, store, previous = legacy_fixture(tmp_path)
    parent_raw = [store.get_page(parent, 0, previous, page) for page in range(4)]
    path = tmp_path / 'session.jsonl'
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps({'type': 'event_msg', 'payload': {
            'type': 'user_message', 'message': 'new tail unit'}}) + '\n')
    archive.archive_file(path, 'codex')
    child = archive._load_manifest()['files'][str(path.resolve())][1]
    assert 'prefix_of' in child
    attempt = store.build_snapshot(child, 1)
    parts = list(store.fragments(child, 1, attempt))
    expected = list(transcript_units(archive, child))
    assert max(len(part.text) for part in parts) <= 4096
    assert ''.join(part.text for part in parts) == ''.join(part.text for part in expected)
    assert [part.unit for part in parts if part.final] == [part.unit for part in expected if part.final]
    assert [store.get_page(parent, 0, previous, page) for page in range(4)] == parent_raw
    assert store.find_snapshot(parent, 0) == previous


def test_legacy_long_page_followed_by_another_text_page_keeps_citations(tmp_path):
    from muninn.history.cited_windows import CitedWindowPlanStore
    text = 'ordinary words ' * 600 + 'following-page-marker'
    archive, entry, units, attempt = legacy_fixture(tmp_path, text=text, legacy_break=7000)
    plans = CitedWindowPlanStore(archive)
    plan = plans.build_snapshot(entry, 0)
    descriptors = [plans.window_at(entry, 0, plan, i) for i in range(plans.count_pages(entry, 0, plan))]
    assert [d['page'] for d in descriptors] == [2, 2, 2, 3]
    assert [d['offset'] for d in descriptors] == [0, 3000, 6000, 0]
    windows = [plans.source.reopen(d) for d in descriptors]
    assert ''.join(w['text'] for w in windows) == text
    for index, (descriptor, window) in enumerate(zip(descriptors, windows)):
        quote = window['text'][-20:]
        refs = plans.source.record_proposals(descriptor, [{
            'type': 'observation', 'text': quote, 'quote': quote,
            'start': len(window['text']) - len(quote)}], model_identity='a' * 64)
        assert plans.source.ledger.get(refs[0])['state'] == 'provisional'
    assert plans.source.ledger.verify_all()['candidates'] == 4
    assert json.loads(units.get_page(entry, 0, attempt, 3))['text'] == text[7000:]


@pytest.mark.parametrize('reader', ['full', 'unit', 'page'])
def test_ciphertext_over_envelope_bound_never_reaches_decryption(tmp_path, monkeypatch, reader):
    _archive, entry, store, attempt = legacy_fixture(tmp_path)
    with store._connect() as db:
        length = store.max_page_chars * 4 + 1
        db.execute('UPDATE pages SET length=?,ciphertext=zeroblob(?) WHERE attempt=? AND ordinal=2',
                   (length, length + 28, attempt))
    import muninn.history.secure_projection_store as base
    decrypt = base.SecureProjectionStore._decrypt_page
    calls = []

    def observed(self, ident, attempt, ordinal, page, cipher=None):
        calls.append(ordinal)
        return decrypt(self, ident, attempt, ordinal, page, cipher)

    monkeypatch.setattr(base.SecureProjectionStore, '_decrypt_page', observed)
    with pytest.raises(ProjectionIntegrityError):
        if reader == 'full':
            list(store.fragments(entry, 0, attempt))
        elif reader == 'unit':
            list(store.unit_fragments(entry, 0, attempt, 1))
        else:
            store.get_page(entry, 0, attempt, 2)
    assert 2 not in calls
