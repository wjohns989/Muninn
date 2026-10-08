"""Observational batch health never grants resubmission or changes checkpoint state."""
import pytest

from muninn.history.batch_health import BatchHealth


def reply(completed=0, failed=0, **extra):
    return {'created_at': 1000, 'status': 'in_progress',
            'request_counts': {'total': 60, 'completed': completed, 'failed': failed}, **extra}


def test_four_hour_zero_progress_is_unknown_not_success_or_failure():
    health = BatchHealth()
    health.observe(reply(), expected=60, now=15400)
    result = health.snapshot(15400)
    assert result['state'] == 'degraded_unknown'
    assert result['completed_requests'] == 0 and result['failed_requests'] == 0
    assert result['deadline_at'] == 87400
    assert result['last_successful_poll_at'] == 15400
    assert result['progress_basis'] == 'provider_creation_zero_reported_outcomes'


def test_progress_and_stall_clock_then_terminal_and_missed_poll():
    health = BatchHealth()
    health.observe(reply(2), expected=60, now=15400)
    assert health.snapshot(15400)['state'] == 'progress_observed'
    health.observe(reply(2), expected=60, now=19001)
    assert health.snapshot(19001)['state'] == 'degraded_unknown'
    health.observe(reply(3), expected=60, now=19002)
    assert health.snapshot(19002)['state'] == 'progress_observed'
    assert health.snapshot(19200)['state'] == 'polling_stale'
    health.observe(reply(60, status='completed'), expected=60, now=19201)
    assert health.snapshot(20000)['state'] == 'provider_completed'


def test_missing_or_malformed_metrics_stay_unknown_and_errors_do_not_hide_last_poll():
    health = BatchHealth()
    health.observe(reply(), expected=60, now=15400)
    health.poll_failed()
    assert health.snapshot(15401)['state'] == 'polling_error'
    assert health.snapshot(15401)['last_successful_poll_at'] == 15400
    assert health.snapshot(15401)['consecutive_poll_errors'] == 1
    for metrics in ({'total': 60}, {'total': 60, 'completed': True, 'failed': 0},
                    {'total': 61, 'completed': 0, 'failed': 0},
                    {'total': 60, 'completed': 60, 'failed': 1}):
        health.observe(reply(request_counts=metrics), expected=60, now=15402)
        result = health.snapshot(15402)
        assert result['state'] == 'metrics_unknown'
        assert result['completed_requests'] is None
    health.observe(reply(created_at=float('nan')), expected=60, now=15403)
    assert health.snapshot(15403)['deadline_at'] is None


def test_deadline_overdue_is_not_invented_provider_failure_and_restart_keeps_age():
    health = BatchHealth()
    health.observe(reply(), expected=60, now=87401)
    assert health.snapshot(87401)['state'] == 'deadline_overdue'
    assert health.snapshot(87401)['provider_state'] == 'in_progress'
    restarted = BatchHealth()
    restarted.observe(reply(), expected=60, now=15400)
    assert restarted.snapshot(15400)['state'] == 'degraded_unknown'


@pytest.mark.asyncio
async def test_owned_batch_poll_failure_reports_health_without_resubmission(tmp_path):
    from tests.test_historical_batch_jobs import fixture
    from tests.test_historical_batch_worker import responses, ready
    from muninn.history.historical_batch_worker import HistoricalBatchWorker
    journal, archive, outbox, ident, _ = fixture(tmp_path)
    journal.reserve_historical_batch(ident)
    submitted, _terminal = responses(archive, outbox, ident)
    submitted['created_at'] = 1000
    clock, wall, calls = [0], [1001], []
    async def send(method, **kwargs):
        calls.append(method)
        if method == 'GET':
            raise TimeoutError('synthetic private diagnostic must not appear in status')
        return submitted
    worker = HistoricalBatchWorker(journal, authorize_submit=lambda _: True,
        send=send, provider_status=ready, clock=lambda: clock[0], wall_clock=lambda: wall[0])
    await worker.step()
    clock[0], wall[0] = 61, 1062
    with pytest.raises(TimeoutError):
        await worker.step()
    snapshot = worker.snapshot()
    assert snapshot['health']['state'] == 'polling_error'
    assert 'private diagnostic' not in str(snapshot)
    assert outbox.read(ident)['state'] == 'submitted'
    assert calls == ['POST', 'GET']


@pytest.mark.asyncio
@pytest.mark.parametrize('bad_identity', [False, True])
async def test_candidate_poll_error_is_visible_without_ownership_or_post(tmp_path, bad_identity):
    from tests.test_historical_batch_jobs import fixture, admission
    from tests.test_historical_batch_worker import responses
    from muninn.history.historical_batch_worker import HistoricalBatchWorker
    journal, archive, outbox, ident, _ = fixture(tmp_path)
    journal.reserve_historical_batch(ident)
    paid = admission(journal, ident)
    outbox.begin_submission(ident, 0)
    journal.mark_historical_batch_dispatched(ident, paid.identifier)
    candidate = responses(archive, outbox, ident)[0]['id']
    outbox.set_recovery_candidate(ident, 1, candidate)
    calls = []
    async def send(method, **kwargs):
        calls.append(method)
        if bad_identity:
            return {'id': 'batch_wrong_identity'}
        raise TimeoutError('synthetic candidate')
    worker = HistoricalBatchWorker(journal, send=send)
    from muninn.history.historical_batch import BatchError
    with pytest.raises(BatchError if bad_identity else TimeoutError):
        await worker.step()
    assert worker.snapshot()['health']['state'] == 'polling_error'
    assert outbox.read(ident)['state'] == 'submission_unknown'
    assert calls == ['GET']


@pytest.mark.asyncio
async def test_repair_poll_error_reaches_parent_without_repeating_successes(tmp_path):
    from tests.test_historical_batch_repairs import _setup, _worker, FakeProvider
    journal, archive, outbox, parent, _ = _setup(tmp_path)
    provider = FakeProvider(journal, archive, outbox, parent)
    clock = [0]
    worker = _worker(journal, provider, clock)
    await worker.step()  # Original submission.
    clock[0] = 61
    await worker.step()  # Original results and one failed-only repair submission.
    assert len(provider.posts) == 2
    async def fail_child_poll(method, **kwargs):
        assert method == 'GET'
        raise TimeoutError('synthetic repair')
    worker.repair_worker.send = fail_child_poll
    clock[0] = 122
    with pytest.raises(TimeoutError):
        await worker.step()
    snapshot = worker.snapshot()
    assert snapshot['health']['state'] == 'polling_error'
    assert snapshot['repair_only'] is True and snapshot['items'] == 1
    assert len(provider.posts) == 2
    assert outbox.repair_records(outbox.read(parent))[0]['state'] == 'submitted'
