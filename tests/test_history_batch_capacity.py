"""Bulk batching stays bounded and does not consume foreground search slots."""
import pytest

from muninn.history.batch_activation import configure_batch, prepare_next_batch
from muninn.history.historical_batch import BatchOutbox
from muninn.history.remote_policy import write_policy
from tests.test_capture_window_jobs import window_fixture
from tests.test_secure_analysis_journal import _result, _target


def bulk_history(tmp_path):
    journal, archive, receipt = window_fixture(tmp_path, text="Ordinary historic observation. " * 16000)
    write_policy(journal.policy_root, enabled=True, daily_usd=5, monthly_usd=50,
                 override_ceiling=False, fallback=lambda: (False, 1, 30, False))
    journal.enroll_historical_latest(limit=128)
    configure_batch(journal.policy_root, enabled=True, max_batches=2)
    return journal, archive, receipt


def fill(journal, receipt, count=128):
    queued = 0
    for _ in range((count + 31) // 32):
        queued += journal.queue_capture_windows(receipt, limit=min(32, count - queued),
                                                remote_policy_generation=1)["queued"]
    return queued


def test_opt_in_bulk_queue_is_bounded_at_128(tmp_path):
    journal, _archive, receipt = bulk_history(tmp_path)
    assert fill(journal, receipt) == 128
    before = journal.capture_window_status(receipt)
    assert journal.queue_capture_windows(receipt, remote_policy_generation=1)["state"] == "queue_full"
    assert journal.capture_window_status(receipt)["queued"] == before["queued"]


def test_revoke_preserves_bulk_work_and_eight_real_foreground_admissions(tmp_path):
    journal, archive, receipt = bulk_history(tmp_path)
    assert fill(journal, receipt) == 128
    configure_batch(journal.policy_root, enabled=False)
    assert journal.queue_capture_windows(receipt, remote_policy_generation=1)["state"] == "queue_full"
    for index in range(9):
        search_id = journal.enqueue_search("needle")
        search = journal.claim_search()
        target = {**_target(archive), "blob": f"{index + 1:032x}"}
        assert journal.finish_search(search_id, search.lease_token, _result(), analysis_target=target)
        row = journal.get_search_job(search_id)
        assert row["analysis_state"] == ("queued" if index < 8 else "not_queued")
    # Admission and ordinary CPU retrieval do not authorize model dispatch.
    assert journal.claim_analysis().lane == 0
    with journal._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM history_analysis_jobs WHERE lane=1").fetchone()[0] == 128


@pytest.mark.asyncio
async def test_bulk_checkpoint_publishes_more_than_24_and_settles_cost_once(tmp_path):
    from muninn.history.historical_batch_worker import HistoricalBatchWorker
    from muninn.history.remote_accounting import status
    from tests.test_historical_batch_worker import ready, responses

    journal, archive, receipt = bulk_history(tmp_path)
    assert fill(journal, receipt, 28) == 28
    ident = prepare_next_batch(journal)
    outbox = BatchOutbox(archive)
    assert len(outbox.read(ident)["items"]) == 28
    submitted, terminal = responses(archive, outbox, ident)
    calls, clock = [], [0]
    async def send(method, **kwargs):
        calls.append(method)
        return submitted if method == "POST" else terminal
    worker = HistoricalBatchWorker(journal, authorize_submit=lambda _: True,
                                   send=send, provider_status=ready, clock=lambda: clock[0])
    await worker.step()
    clock[0] = 61
    await worker.step()
    assert worker.status["state"] == "passed"
    ids = [item["job_id"] for item in outbox.read(ident)["items"]]
    assert journal.verify_publications(job_ids=ids) == 28
    assert status(journal.policy_root)["daily_cost_usd"] == 0.001
    assert not await worker.step()
    assert calls == ["POST", "GET"]
    assert outbox.read(ident)["state"] == "terminal_saved"


def test_byte_bound_splits_batch_without_changing_remaining_jobs(tmp_path, monkeypatch):
    from muninn.history import historical_batch
    from tests.test_batch_activation import history

    journal, archive = history(tmp_path)
    configure_batch(journal.policy_root, enabled=True)
    with journal._connect() as db:
        before = {row["job_id"]: tuple(row) for row in db.execute("SELECT * FROM history_analysis_jobs")}
    original = historical_batch.payload
    def one_item_only(items):
        if len(items) > 1:
            raise historical_batch.BatchError("batch_request_bound")
        return original(items)
    monkeypatch.setattr(historical_batch, "payload", one_item_only)
    ident = prepare_next_batch(journal)
    selected = {item["job_id"] for item in BatchOutbox(archive).read(ident)["items"]}
    assert len(selected) == 1
    with journal._connect() as db:
        for row in db.execute("SELECT * FROM history_analysis_jobs"):
            if row["job_id"] not in selected:
                assert tuple(row) == before[row["job_id"]]


def test_exact_post_byte_bound_and_128_item_bound(monkeypatch):
    from copy import deepcopy

    from muninn.history import historical_batch as batch
    from tests.test_historical_batch import items

    rows = [deepcopy(items()[0]) for _ in range(129)]
    for index, row in enumerate(rows):
        row.update(custom_id=f"{index:032x}", job_id=f"{index + 200:032x}")
    assert len(batch.payload(rows[:128])["requests"]) == 128
    with pytest.raises(batch.BatchError, match="count"):
        batch.payload(rows)
    size = len(batch._wire_json(batch.payload(rows[:2])))
    monkeypatch.setattr(batch, "MAX_REQUEST_BYTES", size)
    assert len(batch._wire_json(batch.payload(rows[:2]))) == size
    monkeypatch.setattr(batch, "MAX_REQUEST_BYTES", size - 1)
    with pytest.raises(batch.BatchError, match="batch_request_bound"):
        batch.payload(rows[:2])


def test_bulk_result_transport_accepts_more_than_old_four_mib():
    import json

    from muninn.history.historical_batch_worker import _decode

    raw = json.dumps({"padding": "x" * (4 * 1024 * 1024 + 1)}).encode()
    assert len(_decode(raw)["padding"]) == 4 * 1024 * 1024 + 1
