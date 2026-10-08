"""Frozen real-source geometry and encrypted recovery; no live/provider calls."""
import copy
import json

import pytest

from muninn.history.batch_packing import pack_items as _pack_items
from muninn.history.cited_windows import CitedWindowPlanStore
from muninn.history.historical_batch import (
    MODEL, BatchError, BatchOutbox, payload, prepare_items, terminal_results, validate_item,
)
from muninn.history.secure_analysis import ModelOutputInvalid
from tests.test_cited_analysis_source import PHRASE, fixture


def prepared(tmp_path, count=11):
    text = ''.join(f'Keep orbital plan {n} for synthetic widget caching.\n'
                   for n in range(max(1000, count * 80)))
    archive, source, cap = fixture(tmp_path, text)
    desc = source.prepare(cap)
    entry = source._window(desc)[0]
    plans = CitedWindowPlanStore(archive)
    plan = plans.build_snapshot(entry, 0)
    descriptors = list(plans._descriptors(entry, 0, desc['attempt']))[:count]
    assert len(descriptors) == count
    plain = prepare_items(source, [(f'{n + 1:032x}', d) for n, d in enumerate(descriptors)])
    source._packing_positions = {item['job_id']: (plan, n) for n, item in enumerate(plain)}
    return archive, source, plain


def pack_items(source, items):
    # Retain the original ten-window contract as compatibility coverage.
    return _pack_items(source, items, positions=source._packing_positions, max_windows=10)


@pytest.mark.parametrize('count,request_sizes', [(30, [30]), (50, [50]), (51, [50, 1])])
def test_new_default_packs_fifty_without_changing_window_identity(tmp_path, count, request_sizes):
    _, source, plain = prepared(tmp_path, count)
    legacy = copy.deepcopy(payload(plain))
    packed = _pack_items(source, plain, positions=source._packing_positions)
    requests = payload(packed)['requests']
    assert [r['body']['max_tokens'] for r in requests] == [2048 * n for n in request_sizes]
    assert all(r['body']['max_tokens'] <= 102400 for r in requests)
    assert [i['window'] for i in packed] == [i['window'] for i in plain]
    assert payload(plain) == legacy
    rows = terminal_results(packed, completed(packed))
    stages = [validate_item(source, item, rows[item['custom_id']])['extraction'] for item in packed]
    assert [s['window'] for s in stages] == [i['window'] for i in plain]
    assert len({s['model_identity'] for s in stages}) == count


def test_retained_ten_window_wire_is_identical_after_ceiling_increase(tmp_path, monkeypatch):
    from muninn.history import batch_packing
    from muninn.history.historical_batch import _wire_json
    archive, source, plain = prepared(tmp_path)
    with monkeypatch.context() as prior:
        prior.setattr(batch_packing, 'MAX_PACK', 10)
        old = _pack_items(source, plain, positions=source._packing_positions, max_windows=10)
        wire = _wire_json(payload(old))
    outbox = BatchOutbox(archive)
    ident = outbox.prepare(old, consent_generation=1)
    assert _wire_json(payload(outbox.read(ident)['items'])) == wire
    assert len(payload(old)['requests']) == 2
    assert len(payload(_pack_items(source, plain, positions=source._packing_positions))['requests']) == 1


@pytest.mark.parametrize('maximum', [51, True, 50.0, 1])
def test_new_selection_ceiling_cannot_be_bypassed(tmp_path, maximum):
    _, source, plain = prepared(tmp_path, 2)
    with pytest.raises(BatchError):
        _pack_items(source, plain, positions=source._packing_positions, max_windows=maximum)


def analysis(window):
    quote = window['text'][:80]
    return {'summary': 'A cited excerpt.', 'decisions': [], 'open_items': [], 'uncertainty': '',
            'proposals': [{'type': 'observation', 'text': quote, 'quote': quote, 'start': 0}]}


def completed(items):
    requests = payload(items)['requests']
    replies = []
    for request in requests:
        content = json.loads(request['body']['messages'][1]['content'])
        result = ({'windows': [{'slot': w['slot'], 'analysis': analysis(w['window'])}
                                for w in content['windows']]} if 'windows' in content else analysis(content))
        replies.append({'custom_id': request['custom_id'], 'error': None,
                        'response': {'status_code': 200, 'body': {'model': MODEL,
                            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(result)}}]}}})
    return {'id': 'batch_packed_fixture', 'endpoint': '/v1/chat/completions', 'model': MODEL,
            'completion_window': '24h', 'status': 'completed',
            'request_counts': {'total': len(requests), 'completed': len(requests), 'failed': 0},
            'usage': {'cost': 0.012, 'is_byok': False}, 'results': replies}


def test_eleven_windows_become_two_requests_without_changing_geometry_or_legacy(tmp_path):
    _, source, plain = prepared(tmp_path)
    legacy = payload(plain)
    packed = pack_items(source, plain)
    assert payload(plain) == legacy and len(legacy['requests']) == 11
    assert len(packed) == 11 and len(payload(packed)['requests']) == 2
    assert [i['window'] for i in packed] == [i['window'] for i in plain]
    assert all(i['window']['length'] <= 3000 for i in packed)
    assert payload(packed)['requests'][0]['body']['max_tokens'] == 20480
    assert payload(packed)['requests'][1]['body']['max_tokens'] == 2048
    wire = json.dumps(payload(packed))
    assert 'scope_ref' not in wire and 'project_ref' not in wire and 'job_id' not in wire


@pytest.mark.parametrize('change', ['missing', 'reordered', 'slot_bool', 'ordinal_bool', 'digest',
                                  'scope', 'source', 'plan', 'null_pack', 'root_collision'])
def test_invalid_cohort_cannot_form_a_provider_payload(tmp_path, change):
    _, source, plain = prepared(tmp_path, 3)
    packed = pack_items(source, plain)
    if change == 'missing':
        packed.pop(1)
    elif change == 'reordered':
        packed.reverse()
    elif change == 'slot_bool':
        packed[1]['pack']['slot'] = True
    elif change == 'ordinal_bool':
        packed[1]['pack']['ordinal'] = True
    elif change == 'plan':
        packed[1]['pack']['plan_attempt'] = 'f' * 32
    elif change == 'null_pack':
        packed[1]['pack'] = None
    elif change == 'root_collision':
        for item in packed:
            item['pack']['request_id'] = packed[0]['custom_id']
    elif change == 'digest':
        packed[0]['pack']['input_sha256'] = '0' * 64
    elif change == 'scope':
        packed[1]['pack']['scope_ref'] = '0' * 64
    else:
        packed[1]['window']['blob'] = 'f' * 32
    with pytest.raises(BatchError):
        payload(packed)


def test_provider_counts_and_per_window_identities_are_separate(tmp_path):
    _, source, plain = prepared(tmp_path)
    packed = pack_items(source, plain)
    result = completed(packed)
    rows = terminal_results(packed, result)
    assert len(rows) == 11 and result['request_counts']['total'] == 2
    stages = [validate_item(source, item, rows[item['custom_id']])['extraction'] for item in packed]
    assert [s['window'] for s in stages] == [i['window'] for i in plain]
    assert len({s['model_identity'] for s in stages}) == 11
    old_rows = terminal_results(plain, completed(plain))
    old = validate_item(source, plain[0], old_rows[plain[0]['custom_id']])['extraction']
    assert old['model_identity'] != stages[0]['model_identity']
    bad = copy.deepcopy(result)
    bad['request_counts']['total'] = 11
    with pytest.raises(BatchError):
        terminal_results(packed, bad)


@pytest.mark.parametrize('failure', ['quote', 'missing_slot', 'duplicate_slot'])
def test_bad_slot_does_not_erase_successful_siblings(tmp_path, failure):
    _, source, plain = prepared(tmp_path, 3)
    packed = pack_items(source, plain)
    result = completed(packed)
    message = result['results'][0]['response']['body']['choices'][0]['message']
    frame = json.loads(message['content'])
    if failure == 'quote':
        frame['windows'][1]['analysis']['proposals'][0]['quote'] = 'not in any fixture source'
    elif failure == 'missing_slot':
        frame['windows'].pop(1)
    else:
        frame['windows'][1]['slot'] = 'w1'
        frame['windows'][2]['slot'] = 'w1'
    message['content'] = json.dumps(frame)
    rows = terminal_results(packed, result)
    assert validate_item(source, packed[0], rows[packed[0]['custom_id']])
    with pytest.raises((ModelOutputInvalid, BatchError)):
        validate_item(source, packed[1], rows[packed[1]['custom_id']])
    if failure == 'duplicate_slot':
        with pytest.raises(BatchError):
            validate_item(source, packed[2], rows[packed[2]['custom_id']])
    else:
        assert validate_item(source, packed[2], rows[packed[2]['custom_id']])


def test_gaps_unknown_scope_and_new_sources_are_not_packed(tmp_path):
    _, source, plain = prepared(tmp_path, 4)
    assert len(payload(pack_items(source, [plain[0], plain[2]]))['requests']) == 2
    from unittest.mock import patch
    original = source.remote_input
    with patch.object(source, 'remote_input', side_effect=lambda d: {**original(d), 'project_ref': None}):
        assert all('pack' not in i for i in pack_items(source, plain))


def test_fresh_scope_and_plan_proof_required(tmp_path):
    from muninn.history.batch_packing import verify_scopes
    from unittest.mock import patch
    _, source, plain = prepared(tmp_path, 3)
    packed = pack_items(source, plain)
    original = source.remote_input
    with patch.object(source, 'remote_input', side_effect=lambda d: {**original(d), 'project_ref': 'f' * 64}):
        with pytest.raises(BatchError):
            verify_scopes(source, packed)
    for item in packed:
        item['pack']['ordinal'] += 1
    with pytest.raises(BatchError):
        verify_scopes(source, packed)


def test_duplicate_json_fields_cannot_hide_a_slot_or_analysis(tmp_path):
    _, source, plain = prepared(tmp_path, 3)
    packed = pack_items(source, plain)
    result = completed(packed)
    message = result['results'][0]['response']['body']['choices'][0]['message']
    message['content'] = message['content'].replace('"slot": "w0"', '"slot": "w2", "slot": "w0"')
    rows = terminal_results(packed, result)
    with pytest.raises(BatchError):
        validate_item(source, packed[0], rows[packed[0]['custom_id']])


@pytest.mark.asyncio
@pytest.mark.parametrize('count,failure', [(2, 'quote'), (50, 'quote'), (50, 'truncated')])
async def test_packed_worker_restart_publication_failed_only_repair_and_single_bills(tmp_path, count, failure):
    from muninn.history.batch_activation import bind_consent, configure_batch, read_batch_policy
    from muninn.history.capture_journal import CaptureJournal
    from muninn.history.cited_analysis_source import CitedAnalysisSource
    from muninn.history.historical_batch_worker import HistoricalBatchWorker
    from muninn.history.remote_accounting import status
    from tests.test_historical_batch_worker import ready
    from muninn.history.remote_policy import write_policy

    archive, source, _ = prepared(tmp_path, count)
    journal = CaptureJournal(archive)
    journal.configure_enrichment(0)
    entry = next(iter(source.ledger._entries.values()))
    receipt = archive._snapshot_receipt(entry, 0)
    journal.enqueue_enrichment_receipt(receipt)
    write_policy(journal.policy_root, enabled=True, daily_usd=5, monthly_usd=50,
                 override_ceiling=False, fallback=lambda: (False, 1, 30, False))
    configure_batch(journal.policy_root, enabled=True, max_batches=10)
    # Planning transactions stay bounded independently of provider request packing.
    queued = 0
    while queued < count:
        result = journal.queue_capture_windows(
            receipt, limit=min(32, count - queued), remote_policy_generation=1)
        assert result['queued'] > 0
        queued += result['queued']
    assert queued == count
    plans = CitedWindowPlanStore(archive)
    positions = {}
    bindings = []
    with journal._connect() as db:
        for row in db.execute('SELECT * FROM history_analysis_jobs'):
            target = journal._validated_analysis_target(row, db)
            positions[row['job_id']] = (target['plan_attempt'], target['ordinal'])
            bindings.append((row['job_id'], plans.window_at(entry, 0, target['plan_attempt'], target['ordinal'])))
    plain = prepare_items(source, bindings)
    assert len(plain) == count
    outbox = BatchOutbox(archive)
    old = outbox.prepare(plain, consent_generation=1)
    plain.sort(key=lambda i: positions[i['job_id']][1])
    packed = _pack_items(CitedAnalysisSource(archive), plain, positions=positions)
    assert len(payload(packed)['requests']) == 1
    ident = outbox.prepare(packed, consent_generation=1)
    bind_consent(journal, outbox, ident, read_batch_policy(journal.policy_root))
    journal.reserve_historical_batch(ident)
    clock, posts, terminals = [0], [], {}

    async def send(method, provider_id=None, body=None):
        if method == 'GET':
            return copy.deepcopy(terminals[provider_id])
        posts.append(copy.deepcopy(body))
        current = outbox.read(ident)
        items = current['items'] if len(posts) == 1 else outbox.repair_records(current)[-1]['items']
        result = completed(items)
        result['id'] = f'batch_packed_{len(posts)}'
        if len(posts) == 1:
            message = result['results'][0]['response']['body']['choices'][0]['message']
            frame = json.loads(message['content'])
            if failure == 'truncated':
                message['content'] = message['content'][:-1]
                result['results'][0]['response']['body']['choices'][0]['finish_reason'] = 'length'
            else:
                frame['windows'][count - 1]['analysis']['proposals'][0]['quote'] = 'NONEXISTENT_SYNTHETIC_QUOTE'
                message['content'] = json.dumps(frame)
        terminals[result['id']] = result
        return {k: v for k, v in {**result, 'status': 'validating'}.items() if k not in {'results', 'usage'}}

    def worker():
        return HistoricalBatchWorker(CaptureJournal(archive), authorize_submit=lambda _: True,
                                     send=send, provider_status=ready, clock=lambda: clock[0])
    first = worker()
    await first.step()
    assert first.status['windows'] == count and first.status['provider_requests'] == 1
    clock[0] = 61
    restarted = worker()
    await restarted.step()
    repair_count = count if failure == 'truncated' else 1
    assert len(posts) == 2 and [len(p['requests']) for p in posts] == [1, repair_count]
    children = outbox.repair_records(outbox.read(ident))
    assert len(children[0]['items']) == repair_count
    assert all('pack' not in i for i in children[0]['items'])
    expected_failed = packed if failure == 'truncated' else [packed[count - 1]]
    assert {i['job_id'] for i in children[0]['items']} == {i['job_id'] for i in expected_failed}
    with journal._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM history_analysis_jobs WHERE state='succeeded'").fetchone()[0] == count - repair_count
    clock[0] = 122
    await restarted.step()
    clock[0] = 183
    await restarted.step()
    assert journal.historical_batch_owner()['phase'] == 'passed'
    assert len(posts) == 2
    assert status(journal.policy_root)['daily_cost_usd'] == pytest.approx(0.024)
    with journal._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM history_analysis_jobs WHERE state='succeeded'").fetchone()[0] == count
    assert outbox.read(old)['state'] == 'prepared'  # Nothing deleted or rewritten.


@pytest.mark.parametrize('count', [3, 50])
def test_packed_submission_restore_and_failed_only_unpacked_child(tmp_path, count):
    from muninn.history.secure_archive import SecureHistoryArchive
    archive, source, plain = prepared(tmp_path, count)
    packed = _pack_items(source, plain, positions=source._packing_positions)
    outbox = BatchOutbox(archive)
    parent = outbox.prepare(packed, consent_generation=1)
    outbox.begin_submission(parent, 0)
    response = completed(packed)
    outbox.save_submission(parent, 1, response)
    outbox.save_terminal(parent, 2, response)
    child_item = {**plain[1], 'custom_id': 'e' * 32}
    child = outbox.prepare_repair(parent, [child_item])
    restored = SecureHistoryArchive.restore_from_backup(archive.root, tmp_path / 'restored', PHRASE)
    recovered = BatchOutbox(restored)
    assert payload(recovered.read(parent)['items']) == payload(packed)
    assert 'pack' not in recovered.read(child)['items'][0]
    assert len(payload(recovered.read(child)['items'])['requests']) == 1
    assert recovered.repair_records(recovered.read(parent))[0]['id'] == child
