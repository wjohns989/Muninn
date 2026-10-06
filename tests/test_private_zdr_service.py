"""Private work fairness in the serial batch consumer, without provider calls."""
from types import SimpleNamespace

import pytest


def setup(monkeypatch, *, phase='passed'):
    from muninn.history import auto_routing, batch_activation, historical_batch_worker
    from muninn.history import service as module
    owner = {'id': 'a' * 32, 'phase': phase}
    journal = SimpleNamespace(policy_root=None, historical_batch_owner=lambda: owner)
    monkeypatch.setattr(auto_routing, 'remote_policy_snapshot', lambda _: SimpleNamespace(enabled=True))
    monkeypatch.setattr(batch_activation, 'read_batch_policy',
                        lambda _: {'enabled': True, 'remaining_batches': 10})
    preparations, steps = [], []
    def prepare(*args, **kwargs):
        preparations.append(kwargs)
        return 'fixture-owned'
    monkeypatch.setattr(batch_activation, 'prepare_next_batch', prepare)
    class Worker:
        def __init__(self, *args, **kwargs):
            self.status = {}
        async def step(self):
            steps.append(True)
    monkeypatch.setattr(historical_batch_worker, 'HistoricalBatchWorker', Worker)
    service = module.HistoryService.__new__(module.HistoryService)
    service._require_capture_journal = lambda: journal
    service._historical_batch_worker = None
    service._historical_batch_gather_deadline = None
    service._capture_auto_enabled = lambda: True
    service._capture_remote_enabled = lambda: True
    service._capture_cadence = SimpleNamespace(attempt_ready=lambda: True)
    calls = []
    async def private(**kwargs):
        calls.append(kwargs)
        return True
    service._process_secure_analysis_once = private
    return service, owner, preparations, steps, calls


@pytest.mark.asyncio
async def test_passed_checkpoint_gets_one_private_opportunity_before_next_batch(monkeypatch):
    service, _, preparations, steps, calls = setup(monkeypatch)
    assert await service._process_secure_batch_once()
    assert calls == [{'include_capture': True, 'include_search': False,
                      'capture_remote_only': True, 'capture_private_zdr': True}]
    assert preparations == [] and steps == []
    assert await service._process_secure_batch_once()
    assert len(calls) == 1 and len(preparations) == 1 and len(steps) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', ['owned', 'sent'])
async def test_active_batch_checkpoint_never_launches_private_work(monkeypatch, phase):
    service, _, preparations, steps, calls = setup(monkeypatch, phase=phase)
    assert await service._process_secure_batch_once()
    assert calls == [] and preparations == [] and steps == [True]


@pytest.mark.asyncio
async def test_private_only_work_waits_for_gather_deadline_and_empty_clean_queue(monkeypatch):
    from muninn.history import batch_activation
    service, owner, preparations, _, calls = setup(monkeypatch)
    service._require_capture_journal = lambda: SimpleNamespace(policy_root=None, historical_batch_owner=lambda: None)
    monkeypatch.setattr(batch_activation, 'prepare_next_batch', lambda *a, **k: None)
    assert await service._process_secure_batch_once()
    assert calls == []
    service._historical_batch_gather_deadline = 0
    assert await service._process_secure_batch_once()
    assert len(calls) == 1 and preparations == []


@pytest.mark.asyncio
async def test_revoked_private_capture_defers_without_any_inference(monkeypatch, tmp_path):
    import json
    import sqlite3

    from muninn.history import secure_analysis
    from tests.test_capture_automatic_service import enabled_service
    service, _, path, _ = enabled_service(monkeypatch, tmp_path)
    path.write_text(json.dumps({'type': 'event_msg', 'payload': {'type': 'user_message',
        'message': 'Keep orbital decisions. API_KEY=short-value'}}) + '\n', encoding='utf-8')
    await service.capture(str(path), 'codex')
    journal = service._require_capture_journal()
    journal.queue_capture_windows(journal.pending_enrichment()[0], limit=1, remote_policy_generation=1)
    job = journal.claim_analysis(include_capture=True, include_search=False)
    journal.defer_analysis(job.job_id, job.lease_token, 'source_not_remote_safe')
    with sqlite3.connect(journal.path) as db:
        db.execute('UPDATE history_analysis_jobs SET due_at=0 WHERE job_id=?', (job.job_id,))
    monkeypatch.setattr(service, '_capture_remote_enabled', lambda: True)
    monkeypatch.setattr('muninn.history.auto_routing.remote_policy_snapshot',
                        lambda _: SimpleNamespace(enabled=False, generation=2))
    async def forbidden(*args, **kwargs):
        pytest.fail('inference attempted after revocation')
    monkeypatch.setattr(secure_analysis, 'analyze_cited_window', forbidden)
    assert await service._process_secure_analysis_once(include_capture=True, include_search=False,
        capture_remote_only=True, capture_private_zdr=True)
    assert journal.get_analysis_job(job.job_id)['error_code'] == 'remote_consent_revoked'
