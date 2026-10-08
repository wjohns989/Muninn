"""Isolated masked ZDR transport/recovery proof, never real secrets or models."""
import json
from types import SimpleNamespace

import httpx
import pytest

from muninn.history import credential_zdr as route
from muninn.history.ambiguity_triage import CandidateForReview
from muninn.history.credential_context import CredentialContextStore
from muninn.history.remote_accounting import AdmissionError, status
from muninn.history.remote_policy import write_policy
from muninn.history.secure_projection_store import ProjectionIntegrityError
from tests.test_credential_context import _fixture

_CLIENT = httpx.Client


def item(context=None):
    return CandidateForReview('local-only-id', 'SERVICE_API_KEY', 'unparsed_value',
                              'fake-value$123', {'provider': 'codex', 'role': 'user',
                              'event_at': '2026-09-30T12:00:00Z', 'time_basis': 'record',
                              'context': context or (
                                  'Documentation uses SERVICE_API_KEY=fake-value$123 as a placeholder.')})


def setup(tmp_path, monkeypatch, *, confidence=1.0, context_revision=2):
    archive, entry = _fixture(tmp_path)
    contexts = CredentialContextStore(archive)
    contexts._parser_revision = context_revision
    attempt = contexts.build_snapshot(entry, 0)
    source = SimpleNamespace(contexts=contexts)
    prepared = (SimpleNamespace(version=0), entry, None, attempt)
    root = tmp_path / 'runtime'
    root.mkdir()
    write_policy(root, enabled=True, daily_usd=5, monthly_usd=50,
                 override_ceiling=False, fallback=lambda: (False, 1, 30, False))
    monkeypatch.setattr(route.llm_settings, 'api_key', lambda: 'synthetic-local-auth')
    monkeypatch.setattr(route, 'openrouter_key_status', lambda **kwargs: {
        'admission_ready': True, 'usage_daily_usd': 0, 'usage_monthly_usd': 0})
    posts = []

    def handler(request):
        assert request.url == 'https://openrouter.ai/api/v1/chat/completions'
        body = json.loads(request.content)
        assert body['provider'] == {'zdr': True, 'data_collection': 'deny', 'require_parameters': True}
        assert body['models'] == [route.llm_settings.DEFAULT_MODEL]
        posts.append(body)
        return httpx.Response(200, json={'model': route.llm_settings.DEFAULT_MODEL,
            'usage': {'cost': .001}, 'choices': [{'finish_reason': 'stop', 'message': {
                'content': json.dumps({'items': [{'index': 0, 'class': 'nonvalue',
                                                  'confidence': confidence(len(posts))
                                                  if callable(confidence) else confidence}]})}}]})

    original = _CLIENT
    monkeypatch.setattr(route.httpx, 'Client', lambda **kwargs: original(
        transport=httpx.MockTransport(handler), **kwargs))
    kwargs = {'policy_root': root, 'generation': 1, 'model': route.llm_settings.DEFAULT_MODEL}
    return source, prepared, kwargs, posts


def test_legacy_mask_recipe_receipt_is_reused_without_another_post(tmp_path, monkeypatch):
    source, prepared, kwargs, posts = setup(tmp_path, monkeypatch, context_revision=1)
    original = route.masked_body
    observed = item(r'Documentation\nSERVICE_API_KEY=fake-value$123 is a placeholder.')
    observed = CandidateForReview(observed.id, 'nSERVICE_API_KEY', observed.reason,
                                  observed.candidate, observed.source_context)
    legacy_body = original(observed, kwargs['model'], _assignment_revision=1)
    assert legacy_body is not None
    legacy_context = json.loads(legacy_body['messages'][1]['content'])['items'][0]['source']['context']
    assert legacy_context == r'Documentation\FIELD=[REDACTED] is a placeholder.'
    assert original(observed, kwargs['model']) != legacy_body
    with monkeypatch.context() as patch:
        patch.setattr(route, 'masked_body', lambda value, model, **kw:
                      original(value, model, _assignment_revision=1))
        assert route.review_context(observed, source, prepared, 0, **kwargs)[1:] == (1, False)
    assert route.review_context(observed, source, prepared, 0, **kwargs)[1:] == (0, True)
    assert len(posts) == 1 and posts[0] == legacy_body
    assert status(kwargs['policy_root'])['daily_cost_usd'] == .001
    from muninn.history.secure_archive import SecureHistoryArchive
    restored = SecureHistoryArchive.restore_from_backup(
        source.contexts.archive.root, tmp_path / 'restored', 'synthetic portable recovery phrase')
    source.contexts = CredentialContextStore(restored)
    monkeypatch.setattr(route.llm_settings, 'api_key', lambda: pytest.fail('reuse requested key'))
    assert route.review_context(observed, source, prepared, 0, **kwargs)[1:] == (0, True)
    assert len(posts) == 1 and status(kwargs['policy_root'])['daily_cost_usd'] == .001


@pytest.mark.parametrize('state', ['unknown', 'unsent', 'wrong_body'])
def test_legacy_intent_never_resends_old_wire_body(tmp_path, monkeypatch, state):
    source, prepared, kwargs, posts = setup(tmp_path, monkeypatch, context_revision=1)
    base = item(r'Documentation\nSERVICE_API_KEY=fake-value$123 is a placeholder.')
    observed = CandidateForReview(base.id, 'nSERVICE_API_KEY', base.reason,
                                  base.candidate, base.source_context)
    legacy_body = route.masked_body(observed, kwargs['model'], _assignment_revision=1)
    digest, identity = route._body_binding(legacy_body)
    admission = route.reserve(kwargs['policy_root'], 1, {'admission_ready': True,
                              'usage_daily_usd': 0, 'usage_monthly_usd': 0})
    if state == 'unsent':
        admission.release_reserved()
        admission.release_unsent()
    else:
        admission.mark_unknown()
    record = {'state': 'intent', 'admission': admission.identifier, 'generation': 1,
              'body_hash': 'f' * 64 if state == 'wrong_body' else digest, 'response': None}
    _, entry, _, attempt = prepared
    source.contexts.save_remote_receipt(entry, 0, attempt, 0, identity, record, expected=None)
    if state == 'unsent':
        assert route.review_context(observed, source, prepared, 0, **kwargs)[1:] == (1, False)
        assert posts == [route.masked_body(observed, kwargs['model'])]
        assert posts[0] != legacy_body
        assert source.contexts.remote_receipt(entry, 0, attempt, 0, identity) == record
    else:
        error = ('credential_receipt_binding_mismatch' if state == 'wrong_body'
                 else 'credential_review_outcome_unknown')
        with pytest.raises(AdmissionError, match=error):
            route.review_context(observed, source, prepared, 0, **kwargs)
        assert posts == []


def test_changed_unknown_receipt_cannot_hide_behind_a_new_identity(tmp_path, monkeypatch):
    source, prepared, kwargs, posts = setup(tmp_path, monkeypatch)
    real_save = source.contexts.save_remote_receipt
    def interrupted(*args, **kw):
        if args[-1]['state'] == 'received':
            raise RuntimeError('simulated interruption')
        return real_save(*args, **kw)
    monkeypatch.setattr(source.contexts, 'save_remote_receipt', interrupted)
    with pytest.raises(RuntimeError):
        route.review_context(item(), source, prepared, 0, **kwargs)
    monkeypatch.setattr(source.contexts, 'save_remote_receipt', real_save)
    original = item()
    changed = CandidateForReview(original.id, original.name, original.reason, original.candidate,
                                 {**original.source_context, 'event_at': '2026-10-01T12:00:00Z'})
    with pytest.raises(AdmissionError, match='credential_receipt_binding_mismatch'):
        route.review_context(changed, source, prepared, 0, **kwargs)
    assert len(posts) == 1


def test_changed_paid_context_body_requires_explicit_resolution_not_another_post(tmp_path, monkeypatch):
    source, prepared, kwargs, posts = setup(tmp_path, monkeypatch)
    original = item()
    route.review_context(original, source, prepared, 0, **kwargs)
    changed = CandidateForReview(original.id, original.name, original.reason, original.candidate,
                                 {**original.source_context, 'event_at': '2026-10-01T12:00:00Z'})
    with pytest.raises(AdmissionError, match='credential_receipt_binding_mismatch'):
        route.review_context(changed, source, prepared, 0, **kwargs)
    assert len(posts) == 1


@pytest.mark.parametrize('state', ['paid', 'unsent'])
def test_new_parser_attempt_cannot_hide_old_paid_work(tmp_path, monkeypatch, state):
    source, old_prepared, kwargs, posts = setup(tmp_path, monkeypatch, context_revision=1)
    if state == 'paid':
        route.review_context(item(), source, old_prepared, 0, **kwargs)
    else:
        mark = route.Admission.mark_unknown
        monkeypatch.setattr(route.Admission, 'mark_unknown', lambda *args, **kw:
                            (_ for _ in ()).throw(AdmissionError('remote_consent_revoked')))
        with pytest.raises(AdmissionError):
            route.review_context(item(), source, old_prepared, 0, **kwargs)
        monkeypatch.setattr(route.Admission, 'mark_unknown', mark)
    contexts = CredentialContextStore(source.contexts.archive)
    source.contexts = contexts
    original, entry, units, old_attempt = old_prepared
    new_attempt = contexts.build_snapshot(entry, 0)
    assert new_attempt != old_attempt
    new_prepared = original, entry, units, new_attempt
    if state == 'paid':
        with pytest.raises(AdmissionError, match='credential_receipt_binding_mismatch'):
            route.review_context(item(), source, new_prepared, 0, **kwargs)
    else:
        assert route.review_context(item(), source, new_prepared, 0, **kwargs)[1:] == (1, False)
    assert len(posts) == 1


@pytest.mark.parametrize('context', [
    'SERVICE_API_KEY=fake-value$123 OTHER_API_KEY=neighbor-value$456',
    json.dumps({'text': 'SERVICE_API_KEY=fake-value$123 OTHER_API_KEY=neighbor-value$456'}),
    json.dumps(json.dumps({'text': 'SERVICE_API_KEY=fake-value$123 OTHER_API_KEY=neighbor-value$456'})),
])
def test_masks_target_neighbors_and_json_escapes(context):
    body = route.masked_body(item(context), route.llm_settings.DEFAULT_MODEL)
    assert body is not None
    content = json.dumps(body)
    assert 'fake-value' not in content and 'neighbor-value' not in content
    data = json.loads(body['messages'][1]['content'])['items'][0]
    assert data['source']['event_at'] == '2026-09-30T12:00:00Z'
    assert data['source']['provider'] == 'codex'
    assert data['value_shape']['has_symbols'] is True
    assert 'local-only-id' not in content


@pytest.mark.parametrize('rows', [
    [{'index': True, 'class': 'nonvalue', 'confidence': 1}],
    [{'index': 1, 'class': 'nonvalue', 'confidence': 1}],
    [{'index': 0, 'class': 'nonvalue', 'confidence': True}],
    [{'index': 0, 'class': 'nonvalue', 'confidence': float('nan')}],
    [{'index': 0, 'class': 'nonvalue', 'confidence': 1}] * 2,
])
def test_invalid_mapping_never_rejects(rows):
    assert route.parse_decisions(json.dumps({'items': rows}), 1) == ['deferred']


def test_duplicate_classifier_keys_defer():
    raw = '{"items":[{"index":0,"class":"uncertain","class":"nonvalue","confidence":1}]}'
    assert route.parse_decisions(raw, 1) == ['deferred']


def test_received_result_is_encrypted_and_reused_without_a_second_post(tmp_path, monkeypatch):
    source, prepared, kwargs, posts = setup(tmp_path, monkeypatch)
    result, calls, reused = route.review_context(item(), source, prepared, 0, **kwargs)
    assert (result.decision, calls, reused) == ('rejected', 1, False)
    assert route.review_context(item(), source, prepared, 0, **kwargs)[1:] == (0, True)
    assert len(posts) == 1 and status(kwargs['policy_root'])['daily_cost_usd'] == .001
    raw = source.contexts.db_path.read_bytes()
    assert b'fake-value' not in raw and b'synthetic-local-auth' not in raw
    assert source.contexts.verify_all() == {'snapshots': 1, 'contexts': 2}


@pytest.mark.parametrize('failure_stage', ['receipt', 'settlement', 'cache'])
def test_restart_does_not_repeat_a_possibly_paid_review(tmp_path, monkeypatch, failure_stage):
    source, prepared, kwargs, posts = setup(tmp_path, monkeypatch)
    method = {'receipt': 'save_remote_receipt', 'cache': 'record_review'}.get(failure_stage)
    target = source.contexts if method else route
    method = method or '_settle'
    original = getattr(target, method)

    def fail(*args, **kw):
        if failure_stage != 'receipt' or args[-1]['state'] == 'received':
            raise RuntimeError('simulated process interruption')
        return original(*args, **kw)

    monkeypatch.setattr(target, method, fail)
    with pytest.raises(RuntimeError, match='simulated'):
        route.review_context(item(), source, prepared, 0, **kwargs)
    monkeypatch.setattr(target, method, original)
    if failure_stage == 'receipt':
        with pytest.raises(AdmissionError, match='credential_review_outcome_unknown'):
            route.review_context(item(), source, prepared, 0, **kwargs)
        assert status(kwargs['policy_root'])['state'] == 'blocked'
    else:
        result, calls, reused = route.review_context(item(), source, prepared, 0, **kwargs)
        assert (result.decision, calls, reused) == ('rejected', 0, True)
        assert status(kwargs['policy_root'])['daily_cost_usd'] == .001
    assert len(posts) == 1


def test_revocation_after_reservation_is_proven_unsent_and_can_resume(tmp_path, monkeypatch):
    source, prepared, kwargs, posts = setup(tmp_path, monkeypatch)
    original = route.Admission.mark_unknown
    monkeypatch.setattr(route.Admission, 'mark_unknown', lambda *args, **kw:
                        (_ for _ in ()).throw(AdmissionError('remote_consent_revoked')))
    with pytest.raises(AdmissionError, match='remote_consent_revoked'):
        route.review_context(item(), source, prepared, 0, **kwargs)
    assert posts == [] and status(kwargs['policy_root'])['state'] == 'ready'
    monkeypatch.setattr(route.Admission, 'mark_unknown', original)
    assert route.review_context(item(), source, prepared, 0, **kwargs)[1] == 1


def test_receipt_tampering_blocks_restore_verification(tmp_path, monkeypatch):
    source, prepared, kwargs, _ = setup(tmp_path, monkeypatch)
    route.review_context(item(), source, prepared, 0, **kwargs)
    with source.contexts._connect() as db:
        db.execute('UPDATE context_remote_calls SET page=1 WHERE page=0')
    with pytest.raises(ProjectionIntegrityError):
        source.contexts.verify_all()


def test_portable_restore_preserves_remote_receipt_without_another_post(tmp_path, monkeypatch):
    from muninn.history.secure_archive import SecureHistoryArchive
    source, prepared, kwargs, posts = setup(tmp_path, monkeypatch)
    route.review_context(item(), source, prepared, 0, **kwargs)
    restored = SecureHistoryArchive.restore_from_backup(
        source.contexts.archive.root, tmp_path / 'restored', 'synthetic portable recovery phrase')
    source.contexts = CredentialContextStore(restored)
    assert route.review_context(item(), source, prepared, 0, **kwargs)[1:] == (0, True)
    assert len(posts) == 1


def test_independent_runtime_restore_reuses_paid_receipt_with_disabled_consent(tmp_path, monkeypatch, recovery_copy):
    from muninn.history.auto_routing import remote_policy_snapshot
    from muninn.history.secure_archive import SecureHistoryArchive
    source, prepared, kwargs, posts = setup(tmp_path, monkeypatch)
    route.review_context(item(), source, prepared, 0, **kwargs)
    backup = tmp_path / 'runtime-backup'
    assert recovery_copy(source.contexts.archive, backup, 'synthetic portable recovery phrase',
                         policy_root=kwargs['policy_root'])['runtime_bundle'] == 1
    restored_root = tmp_path / 'independent-runtime'
    restored = SecureHistoryArchive.restore_from_backup(
        backup, restored_root, 'synthetic portable recovery phrase')
    source.contexts = CredentialContextStore(restored)
    assert remote_policy_snapshot(restored_root).enabled is False
    assert status(restored_root)['daily_cost_usd'] == .001
    monkeypatch.setattr(route.llm_settings, 'api_key', lambda: pytest.fail('restore accessed provider key'))
    kwargs['policy_root'] = restored_root
    assert route.review_context(item(), source, prepared, 0, **kwargs)[1:] == (0, True)
    assert len(posts) == 1


def test_existing_paid_batch_refuses_without_local_probe_or_post(tmp_path, monkeypatch):
    source, prepared, kwargs, posts = setup(tmp_path, monkeypatch)
    paid = route.reserve(kwargs['policy_root'], 1, {'admission_ready': True,
                         'usage_daily_usd': 0, 'usage_monthly_usd': 0})
    paid.mark_unknown()
    with pytest.raises(AdmissionError, match='remote_admission_busy'):
        route.review_context(item(), source, prepared, 0, **kwargs)
    assert posts == []


@pytest.mark.parametrize('kind', ['oversized', 'duplicate_envelope'])
def test_untrusted_response_cannot_clear_unknown_or_cause_resend(tmp_path, monkeypatch, kind):
    source, prepared, kwargs, _ = setup(tmp_path, monkeypatch)
    streamed = []

    class Large(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(1000):
                streamed.append(1)
                yield b'x' * 8192

    def handler(request):
        if kind == 'oversized':
            return httpx.Response(200, stream=Large())
        return httpx.Response(200, content=b'{"usage":{"cost":0},"usage":{"cost":1}}')

    original = _CLIENT
    monkeypatch.setattr(route.httpx, 'Client', lambda **kw: original(
        transport=httpx.MockTransport(handler), **kw))
    with pytest.raises((AdmissionError, ValueError)):
        route.review_context(item(), source, prepared, 0, **kwargs)
    assert status(kwargs['policy_root'])['state'] == 'blocked'
    with pytest.raises(AdmissionError, match='credential_review_outcome_unknown'):
        route.review_context(item(), source, prepared, 0, **kwargs)
    assert len(streamed) <= 9


@pytest.mark.parametrize('candidate', ['SERVICE_API_KEY', 'SERVICE_API'])
def test_variable_name_cannot_reconstruct_withheld_value(candidate):
    value = CandidateForReview('local-only', 'SERVICE_API_KEY', 'unparsed_value',
                               candidate, {'context': 'Documentation reference.'})
    assert route.masked_body(value, route.llm_settings.DEFAULT_MODEL) is None


def test_unparsed_neighbor_does_not_leak_partial_value():
    body = route.masked_body(item('SERVICE_API_KEY=fake-value$123; password my$new$value'),
                             route.llm_settings.DEFAULT_MODEL)
    assert body is None or '$new$value' not in json.dumps(body)


def test_existing_redaction_marker_cannot_hide_an_unmasked_suffix():
    body = route.masked_body(item('SERVICE_API_KEY=fake-value$123; SERVICE_API_KEY=[REDACTED]$new$value'),
                             route.llm_settings.DEFAULT_MODEL)
    assert body is None or '$new$value' not in json.dumps(body)


def test_real_occurrence_join_requires_every_context_and_never_probes_local(tmp_path, monkeypatch):
    from muninn.history.credential_review_source import CredentialReviewSource
    from muninn.history.credential_store import AmbiguousCandidate, CredentialStore, source_fingerprint
    from scripts import triage_credential_ambiguity as runner
    source, _, kwargs, posts = setup(tmp_path, monkeypatch, confidence=lambda n: 1 if n == 1 else .5)
    archive = source.contexts.archive
    path = tmp_path / 'candidate.jsonl'
    rows = [{'type': 'session_meta', 'payload': {'cwd': 'C:/test-project'}}]
    for stamp in ['2026-09-30T12:00:00Z', '2026-09-30T13:00:00Z']:
        rows.append({'type': 'event_msg', 'timestamp': stamp, 'payload': {
            'type': 'user_message', 'message':
                'Documentation uses SERVICE_API_KEY=fake-value$123 as a placeholder.'}})
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8')
    archive.archive_file(path, 'codex')
    entry = archive._load_manifest()['files'][str(path.resolve())][0]
    phrase = 'synthetic portable recovery phrase'
    vault = CredentialStore.create(tmp_path / 'vault', phrase)
    vault.scan_source(passphrase=phrase, project='codex', origin='transcript',
                      source_hash=source_fingerprint(f"{archive.vault_id}:{entry['blob']}:{entry['sha256']}"),
                      findings=[AmbiguousCandidate('SERVICE_API_KEY', 'unparsed_value', 'fake-value$123', '')])
    monkeypatch.setattr(runner, 'probe_gpu', lambda: pytest.fail('ZDR probed GPU'))
    monkeypatch.setattr(runner, 'probe_ollama', lambda *args: pytest.fail('ZDR probed Ollama'))
    params = dict(root=vault.root, passphrase=phrase, limit=1, model_limit=1,
                  model=route.llm_settings.DEFAULT_MODEL, apply=True, base_url='unused-local-url',
                  provider='openrouter', policy_root=kwargs['policy_root'],
                  review_source=CredentialReviewSource(archive))
    first = runner.run(**params)
    assert first['model_calls'] == 1 and first['model_rejected'] == 0
    second = runner.run(**params)
    assert second['contexts_reused'] == 1 and second['model_calls'] == 1
    assert second['model_rejected'] == 0 and second['deferred_for_user'] == 1
    assert vault.ambiguity_status() == {'pending': 1} and len(posts) == 2
    assert all('fake-value' not in json.dumps(body) for body in posts)


def test_readiness_refuses_active_admission_before_passphrase(tmp_path, monkeypatch, capsys):
    import sys

    from muninn.history.credential_store import CredentialStore
    from scripts import triage_credential_ambiguity as runner
    source, _, kwargs, _ = setup(tmp_path, monkeypatch)
    vault = CredentialStore.create(tmp_path / 'vault', 'synthetic portable recovery phrase')
    paid = route.reserve(kwargs['policy_root'], 1, {'admission_ready': True,
                         'usage_daily_usd': 0, 'usage_monthly_usd': 0})
    paid.mark_unknown()
    monkeypatch.setattr(runner.getpass, 'getpass', lambda *a: pytest.fail('busy route requested passphrase'))
    monkeypatch.setattr(sys, 'argv', ['triage', '--root', str(vault.root),
        '--archive-root', str(source.contexts.archive.root), '--policy-root', str(kwargs['policy_root']),
        '--provider', 'openrouter', '--check-readiness'])
    assert runner.main() == 2
    report = json.loads(capsys.readouterr().out)
    assert report['state'] == 'remote_admission_busy' and report['passphrase_needed'] is False


def test_received_receipt_requires_same_admission_generation(tmp_path, monkeypatch):
    import sqlite3
    source, prepared, kwargs, posts = setup(tmp_path, monkeypatch)
    route.review_context(item(), source, prepared, 0, **kwargs)
    with sqlite3.connect(kwargs['policy_root'] / 'remote_policy' / 'policy.sqlite3') as db:
        db.execute('UPDATE remote_admissions SET generation=2')
    with pytest.raises(AdmissionError, match='credential_receipt_binding_mismatch'):
        route.review_context(item(), source, prepared, 0, **kwargs)
    assert len(posts) == 1


def test_unsafe_context_does_not_starve_next_safe_candidate(tmp_path, monkeypatch):
    from scripts import triage_credential_ambiguity as runner
    source, prepared, kwargs, posts = setup(tmp_path, monkeypatch)

    class Store:
        rejected = []
        def list_ambiguities(self, **kw):
            return [{'id': name, 'name': 'SERVICE_API_KEY', 'reason': 'unparsed_value',
                     'created_at': at} for at, name in enumerate(['first', 'second'])]
        def reveal_ambiguity(self, ident, **kw):
            return 'SERVICE_API_KEY' if ident == 'first' else 'fake-value$123'
        def decide_ambiguity(self, ident, **kw):
            self.rejected.append(ident)
        def ambiguity_status(self):
            return {'pending': 2 - len(self.rejected), 'rejected': len(self.rejected)}

    store = Store()
    monkeypatch.setattr(runner, 'CredentialStore', lambda root: store)
    source.prepare = lambda row, *, candidate=None: prepared
    def inputs(prepared, row, value):
        yield 0, CandidateForReview(row['id'], row['name'], row['reason'], value,
                                    {'context': f'Documentation SERVICE_API_KEY={value} is a placeholder.'})
    source.inputs = inputs
    result = runner.run(root=tmp_path, passphrase='synthetic phrase', limit=2,
                         model_limit=2, model=route.llm_settings.DEFAULT_MODEL, apply=True,
                         base_url='unused', review_source=source, provider='openrouter',
                         policy_root=kwargs['policy_root'])
    assert store.rejected == ['second'] and len(posts) == 1
    assert result['next_cursor']['id'] == 'second' and result['deferred_for_user'] == 1
