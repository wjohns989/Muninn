"""Paid reply recovery in isolated encrypted stores; never real inference."""
import json

import pytest

from muninn.history import secure_analysis as analysis
from muninn.history.capture_journal import CaptureJournal, SearchJobError
from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.remote_accounting import reserve, status
from muninn.history.remote_policy import write_policy
from tests.test_analysis_publication_journal import queued, expire


def retained_fixture(tmp_path):
    journal, archive, job, stage, source = queued(tmp_path)
    write_policy(tmp_path, enabled=True, daily_usd=5, monthly_usd=50,
                 override_ceiling=False, fallback=lambda: (False, 5, 50, False))
    with journal._connect() as db:
        db.execute("UPDATE history_analysis_jobs SET remote_policy_generation=1 WHERE job_id=?", (job.job_id,))
    admission = reserve(tmp_path, 1, {"state": "ready", "admission_ready": True,
        "key_limit_usd": 5, "key_remaining_usd": 5, "key_reset": "daily",
        "usage_daily_usd": 0, "usage_monthly_usd": 0})
    admission.mark_unknown()
    assert admission.settle_response({"usage": {"cost": 0.01}})
    assert journal.bind_analysis_window(job.job_id, job.lease_token, stage['window'])
    assert journal.mark_remote_dispatched(job.job_id, job.lease_token)
    body = analysis.Provider('openrouter', '', ['fixture-model'], 'fixture-key').request_body(
        analysis._cited_prompt(source.reopen(stage['window'])))
    body['max_completion_tokens'] = 2048
    body['response_format'] = {'type': 'json_schema', 'json_schema': {
        'name': 'secure_excerpt_analysis', 'strict': True, 'schema': analysis._CITED_SCHEMA}}
    content = json.dumps({**stage['result']['analysis'], 'proposals': stage['proposals']})
    data = {'model': 'fixture-model', 'choices': [{'message': {'content': content}}],
            'usage': {'cost': 0.01}}
    reply = analysis.retained_cited_reply(source, stage['window'], body, data,
                                          admission.identifier, 200)
    return journal, archive, job, source, reply


def test_reply_encrypted_immutable_not_public_and_recovers_without_post(tmp_path, monkeypatch):
    journal, archive, job, source, reply = retained_fixture(tmp_path)
    assert journal.retain_analysis_reply(job.job_id, job.lease_token, reply)
    assert journal.retain_analysis_reply(job.job_id, job.lease_token, reply)
    changed = {**reply, 'response': reply['response'] + ' '}
    with pytest.raises(SearchJobError):
        journal.retain_analysis_reply(job.job_id, job.lease_token, changed)
    assert b'Keep needle citations' not in journal.path.read_bytes()
    assert 'response' not in json.dumps(journal.get_analysis_job(job.job_id))
    expire(journal, job)
    restored = CaptureJournal(archive)
    recovered = restored.claim_analysis()
    assert recovered and recovered.remote_reply == reply
    monkeypatch.setattr(analysis.httpx, 'AsyncClient', lambda *a, **kw: pytest.fail('HTTP replay'))
    outcome = analysis.replay_cited_reply(source, recovered.window, recovered.remote_reply)
    assert restored.stage_analysis(recovered.job_id, recovered.lease_token, outcome['extraction'])
    assert restored.begin_publication(recovered.job_id, recovered.lease_token)
    stage = outcome['extraction']
    refs = source.record_proposals(stage['window'], stage['proposals'], model_identity=stage['model_identity'])
    assert restored.acknowledge_publication(recovered.job_id, recovered.lease_token, refs)
    assert restored.get_analysis_job(job.job_id)['state'] == 'succeeded'
    assert status(tmp_path)['daily_cost_usd'] == 0.01
    assert restored.verify_all() == 0


@pytest.mark.parametrize('field', ['window', 'request_sha256', 'input_sha256', 'contract'])
def test_replay_requires_original_input_request_and_contract(tmp_path, field):
    journal, archive, job, source, reply = retained_fixture(tmp_path)
    changed = dict(reply)
    if field == 'window':
        changed[field] = {**reply[field], 'offset': 1}
    else:
        changed[field] = '0' * 64
    with pytest.raises((analysis.ModelOutputInvalid, ValueError)):
        analysis.replay_cited_reply(source, reply['window'], changed)


def test_unsettled_or_other_generation_cannot_be_retained(tmp_path):
    journal, archive, job, source, reply = retained_fixture(tmp_path)
    with journal._connect() as db:
        db.execute('UPDATE history_analysis_jobs SET remote_policy_generation=2 WHERE job_id=?', (job.job_id,))
    with pytest.raises(SearchJobError):
        journal.retain_analysis_reply(job.job_id, job.lease_token, reply)


@pytest.mark.parametrize('state', ['reserved', 'unknown', 'released'])
def test_unsettled_admission_cannot_authorize_reply(tmp_path, state):
    from muninn.history.remote_accounting import _db
    journal, archive, job, source, reply = retained_fixture(tmp_path)
    with _db(tmp_path) as (db, _):
        db.execute('UPDATE remote_admissions SET state=?,resolution=NULL,cost_micro=NULL WHERE id=?',
                   (state, reply['admission_id']))
    with pytest.raises(SearchJobError):
        journal.retain_analysis_reply(job.job_id, job.lease_token, reply)


@pytest.mark.parametrize('expired', [True, False])
def test_reply_requires_current_lease(tmp_path, expired):
    journal, archive, job, source, reply = retained_fixture(tmp_path)
    if expired:
        expire(journal, job)
    assert not journal.retain_analysis_reply(job.job_id,
        job.lease_token if expired else '0' * 32, reply)


def test_settled_http_rejection_is_retained_but_not_published(tmp_path):
    journal, archive, job, source, reply = retained_fixture(tmp_path)
    reply['http_status'] = 429
    assert journal.retain_analysis_reply(job.job_id, job.lease_token, reply)
    with pytest.raises(analysis.ModelOutputInvalid) as failure:
        analysis.replay_cited_reply(source, reply['window'], reply)
    assert failure.value.code == 'provider_rejected'
    assert journal.fail_retained_reply(job.job_id, job.lease_token, failure.value.code)
    assert journal.get_analysis_job(job.job_id)['error_code'] == 'remote_provider_rejected'
    assert source.ledger.verify_all()['candidates'] == 0


def test_retained_invalid_reply_stops_without_new_dispatch(tmp_path):
    journal, archive, job, source, reply = retained_fixture(tmp_path)
    data = json.loads(reply['response'])
    data['choices'][0]['message']['content'] = 'PRIVATE_INVALID_REPLY'
    reply['response'] = json.dumps(data)
    assert journal.retain_analysis_reply(job.job_id, job.lease_token, reply)
    with pytest.raises(analysis.ModelOutputInvalid):
        analysis.replay_cited_reply(source, reply['window'], reply)
    assert journal.fail_retained_reply(job.job_id, job.lease_token, 'json')
    assert journal.get_analysis_job(job.job_id)['state'] == 'failed'
    assert journal.get_analysis_job(job.job_id)['error_code'] == 'remote_output_json'
    assert journal.claim_analysis() is None
    assert b'PRIVATE_INVALID_REPLY' not in journal.path.read_bytes()


def test_old_uncertain_dispatch_is_not_retried(tmp_path):
    journal, archive, job, source, reply = retained_fixture(tmp_path)
    expire(journal, job)
    restored = CaptureJournal(archive)
    assert restored.get_analysis_job(job.job_id)['state'] == 'outcome_unknown'
    assert restored.claim_analysis() is None


def test_reply_ciphertext_tamper_fails_closed(tmp_path):
    journal, archive, job, source, reply = retained_fixture(tmp_path)
    assert journal.retain_analysis_reply(job.job_id, job.lease_token, reply)
    with journal._connect() as db:
        sealed = db.execute('SELECT sealed_remote_reply FROM history_analysis_jobs').fetchone()[0]
        db.execute('UPDATE history_analysis_jobs SET sealed_remote_reply=?', (sealed[:-1] + bytes([sealed[-1] ^ 1]),))
    with pytest.raises(VaultIntegrityError):
        journal.verify_all()


def test_one_settled_admission_cannot_authorize_two_jobs_or_transplanted_ciphertext(tmp_path):
    from tests.test_secure_analysis_journal import _result
    journal, archive, job, source, reply = retained_fixture(tmp_path)
    assert journal.retain_analysis_reply(job.job_id, job.lease_token, reply)
    original = journal._publication_row(job.job_id)
    target = journal._analysis_row(original).target
    target['terms'] = ['different', 'terms']
    search = journal.enqueue_search('different terms')
    search_job = journal.claim_search()
    assert journal.finish_search(search, search_job.lease_token, _result(),
                                 analysis_target=target, remote_policy_generation=1)
    other = journal.claim_analysis()
    assert other.job_id != job.job_id
    assert journal.bind_analysis_window(other.job_id, other.lease_token, reply['window'])
    assert journal.mark_remote_dispatched(other.job_id, other.lease_token)
    with pytest.raises(SearchJobError):
        journal.retain_analysis_reply(other.job_id, other.lease_token, reply)
    with journal._connect() as db:
        db.execute('UPDATE history_analysis_jobs SET remote_reply_identity=NULL WHERE job_id=?', (job.job_id,))
        db.execute('UPDATE history_analysis_jobs SET sealed_remote_reply=?,remote_reply_identity=? WHERE job_id=?',
                   (original['sealed_remote_reply'], original['remote_reply_identity'], other.job_id))
    with pytest.raises(VaultIntegrityError):
        journal._read_remote_reply(journal._publication_row(other.job_id))


def test_oversized_reply_is_fully_preserved_as_authenticated_ciphertext_chunks(tmp_path):
    import hashlib
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    journal, archive, job, source, reply = retained_fixture(tmp_path)
    data = json.loads(reply['response'])
    data['choices'][0]['message']['content'] = '😀PRIVATE_INVALID_REPLY' * 10000
    reply['response'] = json.dumps(data, ensure_ascii=False)
    original = reply['response'].encode('utf-8')
    assert journal.retain_analysis_reply(job.job_id, job.lease_token, reply)
    assert journal.retain_analysis_reply(job.job_id, job.lease_token, reply)
    row = journal._publication_row(job.job_id)
    recovered = journal._read_remote_reply(row)
    assert recovered['response']['bytes'] == len(original)
    assert recovered['response']['sha256'] == hashlib.sha256(original).hexdigest()
    assert len(row['sealed_remote_reply']) < 262144
    with journal._connect() as db:
        chunks = db.execute('SELECT chunk_index,sealed FROM history_analysis_reply_chunks ORDER BY chunk_index').fetchall()
    raw = []
    for chunk in chunks:
        purpose = journal._remote_reply_purpose(row) + ':chunk:' + str(chunk['chunk_index'])
        sealed = chunk['sealed']
        part = AESGCM(journal._search_key(job.job_id)).decrypt(sealed[:12], sealed[12:],
            journal._search_aad(job.job_id, purpose))
        assert len(part) <= 65536
        raw.append(part)
    assert b''.join(raw) == original
    assert reply['admission_id'].encode('ascii') not in journal.path.read_bytes()
    assert b'PRIVATE_INVALID_REPLY' not in journal.path.read_bytes()
    with pytest.raises(analysis.ModelOutputInvalid) as failure:
        analysis.replay_cited_reply(source, recovered['window'], recovered)
    assert failure.value.code == 'reply_bound'
    assert journal.fail_retained_reply(job.job_id, job.lease_token, failure.value.code)
    assert journal.get_analysis_job(job.job_id)['error_code'] == 'remote_reply_bound'
    assert journal.verify_all() == 0
    with journal._connect() as db:
        db.execute('UPDATE history_analysis_reply_chunks SET sealed=? WHERE chunk_index=0', (b'bad',))
    with pytest.raises(VaultIntegrityError):
        journal.verify_all()


@pytest.mark.asyncio
@pytest.mark.parametrize('valid', [True, False])
async def test_real_worker_replays_after_restart_and_consent_revocation(tmp_path, monkeypatch, valid):
    from muninn.history.service import HistoryService
    journal, archive, job, source, reply = retained_fixture(tmp_path)
    if not valid:
        data = json.loads(reply['response'])
        data['choices'][0]['message']['content'] = 'PRIVATE_INVALID_REPLY'
        reply['response'] = json.dumps(data)
    assert journal.retain_analysis_reply(job.job_id, job.lease_token, reply)
    expire(journal, job)
    restored = CaptureJournal(archive)
    write_policy(tmp_path, enabled=False, daily_usd=5, monthly_usd=50,
                 override_ceiling=False, fallback=lambda: (False, 5, 50, False))
    monkeypatch.setenv('MUNINN_HISTORY_SECURITY', 'strict')
    service = HistoryService(None, tmp_path / 'service', home=tmp_path)
    service._capture_journal = restored
    monkeypatch.setattr(service, '_require_secure_archive', lambda: archive)
    monkeypatch.setattr(analysis, 'analyze_cited_window', lambda *a, **kw: pytest.fail('model replay'))
    monkeypatch.setattr(analysis.httpx, 'AsyncClient', lambda *a, **kw: pytest.fail('HTTP replay'))
    assert await service._process_secure_analysis_once()
    visible = restored.get_analysis_job(job.job_id)
    assert visible['state'] == ('succeeded' if valid else 'failed')
    assert visible['error_code'] == (None if valid else 'remote_output_json')
    assert status(tmp_path)['daily_cost_usd'] == 0.01


@pytest.mark.parametrize('oversized', [False, True])
def test_portable_backup_restores_reply_and_accounting_without_remote_authority(tmp_path, oversized, recovery_copy):
    from muninn.history.secure_archive import SecureHistoryArchive
    from muninn.history.auto_routing import remote_policy_snapshot
    journal, archive, job, source, reply = retained_fixture(tmp_path)
    if oversized:
        reply['response'] += ' ' * 131073
    assert journal.retain_analysis_reply(job.job_id, job.lease_token, reply)
    expected = journal._read_remote_reply(journal._publication_row(job.job_id))
    recovery_copy(archive, tmp_path / 'backup', 'synthetic recovery passphrase')
    restored = SecureHistoryArchive.restore_from_backup(tmp_path / 'backup', tmp_path / 'recovery',
                                                       'synthetic recovery passphrase')
    recovered_journal = CaptureJournal(restored, recover=False)
    recovered = recovered_journal._analysis_row(recovered_journal._publication_row(job.job_id))
    assert recovered.remote_reply == expected
    assert not remote_policy_snapshot(restored.root.parent).enabled
    assert recovered_journal.verify_all() == 0


@pytest.mark.parametrize('mutation', ['missing', 'reordered', 'extra'])
def test_retained_chunk_sequence_must_match_complete_receipt(tmp_path, mutation):
    journal, archive, job, source, reply = retained_fixture(tmp_path)
    reply['response'] += ' ' * 131073
    assert journal.retain_analysis_reply(job.job_id, job.lease_token, reply)
    with journal._connect() as db:
        if mutation == 'missing':
            db.execute('DELETE FROM history_analysis_reply_chunks WHERE job_id=? AND chunk_index=0', (job.job_id,))
        elif mutation == 'reordered':
            db.execute('UPDATE history_analysis_reply_chunks SET chunk_index=99 WHERE job_id=? AND chunk_index=0', (job.job_id,))
        else:
            db.execute('INSERT INTO history_analysis_reply_chunks VALUES(?,?,?)', (job.job_id, 99, b'bad'))
    with pytest.raises(VaultIntegrityError):
        journal.verify_all()


def test_projected_replay_preserves_original_safe_ranges(tmp_path):
    from tests.test_projected_memory_publication import mixed
    from muninn.history.cited_zdr_projection import CitedZDRProjection
    archive, source, descriptor, proposal, view = mixed(tmp_path)
    projection = CitedZDRProjection(source, descriptor)
    body = analysis.Provider('openrouter', '', ['fixture-model'], 'fixture-key').request_body(
        analysis._cited_prompt(projection.reopen(descriptor)))
    body['response_format'] = {'type': 'json_schema', 'json_schema': {
        'name': 'secure_excerpt_analysis', 'strict': True, 'schema': analysis._CITED_SCHEMA}}
    data = {'model': 'fixture-model', 'choices': [{'message': {'content': json.dumps({
        'summary': 'Keep SQLite.', 'decisions': [], 'open_items': [], 'uncertainty': '',
        'proposals': [proposal]})}}]}
    reply = analysis.retained_cited_reply(projection, descriptor, body, data, 'a' * 32, 200)
    assert 'short-value' not in json.dumps(reply['request'])
    outcome = analysis.replay_cited_reply(source, descriptor, reply)
    assert outcome['extraction']['source_view'] == view
    assert outcome['extraction']['proposals'] == [proposal]
    reply['source_view'] = {**view, 'ranges': []}
    with pytest.raises(ValueError):
        analysis.replay_cited_reply(source, descriptor, reply)


@pytest.mark.asyncio
async def test_transport_retains_settled_invalid_reply_before_parse(tmp_path, monkeypatch):
    from tests.test_remote_admission_transport import route, response
    journal, archive, job, stage, source = queued(tmp_path)
    async def post():
        return response(0.02, content='PRIVATE_INVALID_REPLY')
    history = route(tmp_path, monkeypatch, post)
    with journal._connect() as db:
        db.execute('UPDATE history_analysis_jobs SET remote_policy_generation=1 WHERE job_id=?', (job.job_id,))
    assert journal.bind_analysis_window(job.job_id, job.lease_token, stage['window'])
    async def marker():
        return journal.mark_remote_dispatched(job.job_id, job.lease_token)
    async def retain(reply):
        return journal.retain_analysis_reply(job.job_id, job.lease_token, reply)
    with pytest.raises(analysis.ModelOutputInvalid):
        await analysis.analyze_cited_window(history, source, stage['window'],
            allow_remote=True, prefer_remote=True, before_remote=marker, retain_remote_reply=retain)
    retained = journal._analysis_row(journal._publication_row(job.job_id)).remote_reply
    assert 'PRIVATE_INVALID_REPLY' in retained['response']
    assert status(tmp_path)['daily_cost_usd'] == 0.02
    assert b'PRIVATE_INVALID_REPLY' not in journal.path.read_bytes()


@pytest.mark.asyncio
async def test_worker_cancellation_drains_paid_reply_writer_without_publication(tmp_path, monkeypatch):
    import asyncio
    import threading
    from contextlib import asynccontextmanager
    from muninn.history.service import HistoryService
    from tests.test_remote_admission_transport import route, response
    journal, archive, job, stage, source = queued(tmp_path)
    calls = []
    async def post():
        calls.append(True)
        return response(0.02, content=json.dumps({**stage['result']['analysis'], 'proposals': stage['proposals']}))
    route(tmp_path, monkeypatch, post)
    with journal._connect() as db:
        db.execute('UPDATE history_analysis_jobs SET remote_policy_generation=1 WHERE job_id=?', (job.job_id,))
    expire(journal, job)
    monkeypatch.setenv('MUNINN_HISTORY_SECURITY', 'strict')
    service = HistoryService(None, tmp_path / 'service', home=tmp_path)
    service._capture_journal = journal
    monkeypatch.setattr(service, '_require_secure_archive', lambda: archive)
    monkeypatch.setattr(analysis, '_select_local', lambda _: (None, 'gpu_busy'))
    @asynccontextmanager
    async def slot():
        yield
    monkeypatch.setattr('muninn.extraction.ollama_slot.async_ollama_slot', slot)
    entered, release = threading.Event(), threading.Event()
    retain = journal.retain_analysis_reply
    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return retain(*args)
    monkeypatch.setattr(journal, 'retain_analysis_reply', blocked)
    task = asyncio.create_task(service._process_secure_analysis_once())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(calls) == 1
    row = journal._publication_row(job.job_id)
    assert journal._read_remote_reply(row) is not None
    assert row['publication_started'] == 0 and row['sealed_extraction'] is None
    assert source.ledger.verify_all()['candidates'] == 0
    assert status(tmp_path)['daily_cost_usd'] == 0.02
