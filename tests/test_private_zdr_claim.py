"""Explicit private lane selection in isolated encrypted capture journals."""
import json
import sqlite3

import pytest

from tests.test_capture_window_jobs import window_fixture


def parked(tmp_path):
    journal, archive, receipt = window_fixture(tmp_path,
        text='Keep orbital decisions.\nAPI_KEY=short-value\nKeep SQLite.')
    journal.queue_capture_windows(receipt, limit=1, remote_policy_generation=1)
    job = journal.claim_analysis(include_capture=True, include_search=False)
    assert journal.defer_analysis(job.job_id, job.lease_token, 'source_not_remote_safe')
    with sqlite3.connect(journal.path) as db:
        db.execute('UPDATE history_analysis_jobs SET due_at=0 WHERE job_id=?', (job.job_id,))
    return journal, archive, job


def test_private_job_needs_separate_remote_only_selection(tmp_path):
    journal, _, job = parked(tmp_path)
    assert journal.claim_analysis(include_capture=True, include_search=False, capture_remote_only=True) is None
    claimed = journal.claim_analysis(include_capture=True, include_search=False,
        capture_remote_only=True, capture_private_zdr=True)
    assert claimed.job_id == job.job_id and claimed.remote_policy_generation == 1
    with sqlite3.connect(journal.path) as db:
        assert db.execute('SELECT remote_dispatched FROM history_analysis_jobs WHERE job_id=?',
                          (claimed.job_id,)).fetchone() == (0,)


@pytest.mark.parametrize('kwargs', [{}, {'include_capture': True},
    {'include_capture': True, 'capture_remote_only': True}])
def test_private_selection_cannot_be_used_as_general_or_local_claim(tmp_path, kwargs):
    journal, _, _ = parked(tmp_path)
    with pytest.raises(ValueError):
        journal.claim_analysis(capture_private_zdr=True, **kwargs)


def test_private_lane_does_not_consume_unrelated_clean_jobs(tmp_path):
    journal, _, receipt = window_fixture(tmp_path)
    journal.queue_capture_windows(receipt, limit=1, remote_policy_generation=1)
    assert journal.claim_analysis(include_capture=True, include_search=False,
        capture_remote_only=True, capture_private_zdr=True) is None
    assert journal.claim_analysis(include_capture=True, include_search=False, capture_remote_only=True)


def test_sent_private_job_is_not_readmitted_after_restart(tmp_path):
    from muninn.history.capture_journal import CaptureJournal
    journal, archive, old = parked(tmp_path)
    job = journal.claim_analysis(include_capture=True, include_search=False,
        capture_remote_only=True, capture_private_zdr=True)
    assert journal.mark_remote_dispatched(job.job_id, job.lease_token)
    with sqlite3.connect(journal.path) as db:
        db.execute('UPDATE history_analysis_jobs SET lease_until=0 WHERE job_id=?', (job.job_id,))
    restarted = CaptureJournal(archive, recover=False)
    assert restarted.claim_analysis(include_capture=True, include_search=False,
        capture_remote_only=True, capture_private_zdr=True) is None
    assert restarted.get_analysis_job(old.job_id)['state'] == 'outcome_unknown'


def test_authenticated_owned_checkpoint_blocks_unrelated_private_job(tmp_path):
    from tests.test_historical_batch_jobs import fixture
    journal, archive, _, ident, _ = fixture(tmp_path)
    journal.reserve_historical_batch(ident)
    path = tmp_path / 'private.jsonl'
    path.write_text(json.dumps({'type': 'event_msg', 'payload': {'type': 'user_message',
        'message': 'Keep orbital decisions. API_KEY=short-value'}}) + '\n', encoding='utf-8')
    receipt = archive.archive_file(path, 'codex', include_snapshot_receipt=True)['snapshot_receipt']
    journal.enqueue_enrichment_receipt(receipt)
    journal.queue_capture_windows(receipt, limit=1, remote_policy_generation=1)
    job = journal.claim_analysis(include_capture=True, include_search=False)
    assert journal.defer_analysis(job.job_id, job.lease_token, 'source_not_remote_safe')
    with sqlite3.connect(journal.path) as db:
        db.execute('UPDATE history_analysis_jobs SET due_at=0 WHERE job_id=?', (job.job_id,))
    assert journal.claim_analysis(include_capture=True, include_search=False,
        capture_remote_only=True, capture_private_zdr=True) is None


@pytest.mark.parametrize('legacy', [False, True])
def test_projected_stage_restarts_and_publishes_without_redispatch(tmp_path, legacy):
    from muninn.history import secure_analysis
    from muninn.history.capture_journal import CaptureJournal
    from muninn.history.cited_analysis_source import CitedAnalysisSource
    from muninn.history.cited_zdr_projection import CitedZDRProjection
    from muninn.history.remote_accounting import reserve
    from tests.test_remote_accounting import READY, policy
    journal, archive, old = parked(tmp_path)
    policy(journal.policy_root)
    job = journal.claim_analysis(include_capture=True, include_search=False,
        capture_remote_only=True, capture_private_zdr=True)
    from muninn.history.cited_windows import CitedWindowPlanStore
    plans = CitedWindowPlanStore(archive)
    entry = plans.source.ledger._entries[(job.target['blob'], job.target['version'])]
    descriptor = plans.window_at(entry, job.target['version'], job.target['plan_attempt'], job.target['ordinal'])
    source = CitedAnalysisSource(archive)
    projection = CitedZDRProjection(source, descriptor)
    quote = 'Keep SQLite.'
    output = json.dumps({'summary': 'SQLite requested.', 'decisions': [], 'open_items': [],
        'uncertainty': '', 'proposals': [{'type': 'decision', 'text': 'Keep SQLite.',
            'quote': quote, 'start': source.reopen(descriptor)['text'].index(quote)}]})
    stage = secure_analysis._cited_outcome(output, projection, descriptor,
                                         'openrouter', 'fixture-model')['extraction']
    assert stage['source_view'] == projection.source_view()
    if legacy:
        stage.pop('source_view')  # Previously installed stages remain unchanged.
    assert journal.bind_analysis_window(job.job_id, job.lease_token, descriptor)
    paid = reserve(journal.policy_root, 1, READY)
    assert journal.mark_remote_dispatched(job.job_id, job.lease_token)
    paid.mark_unknown()
    assert paid.settle_response({'usage': {'cost': 0.001}})  # Isolated simulated reply; no HTTP.
    stage['admission_id'] = paid.identifier
    assert journal.stage_analysis(job.job_id, job.lease_token, stage)
    assert journal.begin_publication(job.job_id, job.lease_token)
    with sqlite3.connect(journal.path) as db:
        db.execute('UPDATE history_analysis_jobs SET lease_until=0 WHERE job_id=?', (job.job_id,))
    restarted = CaptureJournal(archive, recover=False)
    replay = restarted.claim_analysis(include_capture=True, include_search=False,
        capture_remote_only=True, capture_private_zdr=True)
    assert replay.state == 'publishing' and replay.extraction == stage
    refs = source.record_proposals(descriptor, stage['proposals'], model_identity=stage['model_identity'],
                                  **({'source_view': stage['source_view']} if not legacy else {}))
    assert restarted.acknowledge_publication(replay.job_id, replay.lease_token, refs)
    assert restarted.verify_publications(job_ids=[old.job_id]) == 1
    assert restarted.get_analysis_job(old.job_id)['memory_refs'] == refs
    assert ('text' in source.ledger.get(refs[0])) is not legacy
    assert b'short-value' not in restarted.path.read_bytes()
