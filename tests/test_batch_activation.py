"""Retained batch activation on isolated encrypted historical archives."""
import pytest

from muninn.history.batch_activation import authorize_batch, configure_batch, prepare_next_batch, read_batch_policy
from muninn.history.historical_batch import BatchOutbox
from muninn.history.remote_policy import write_policy
from tests.test_capture_historical_enrollment import legacy_fixture


def history(tmp_path):
    journal, archive = legacy_fixture(tmp_path)
    write_policy(journal.policy_root, enabled=True, daily_usd=5, monthly_usd=50,
                 override_ceiling=False, fallback=lambda: (False, 1, 30, False))
    journal.enroll_historical_latest(limit=128)
    for receipt in journal.pending_enrichment():
        journal.queue_capture_windows(receipt, limit=4, remote_policy_generation=1)
    return journal, archive


def test_disabled_by_default_does_not_create_outbox(tmp_path):
    journal, archive = history(tmp_path)
    assert not read_batch_policy(journal.policy_root)["enabled"]
    assert prepare_next_batch(journal) is None
    assert not (archive.root / "historical-batches.db").exists()


def test_exact_binding_and_revoke_reenable_does_not_revive_old_batch(tmp_path):
    journal, archive = history(tmp_path)
    configure_batch(journal.policy_root, enabled=True)
    ident = prepare_next_batch(journal)
    assert ident and authorize_batch(journal, 1)
    assert read_batch_policy(journal.policy_root)["remaining_batches"] == 0
    assert prepare_next_batch(journal) is None
    configure_batch(journal.policy_root, enabled=False)
    assert not authorize_batch(journal, 1)
    configure_batch(journal.policy_root, enabled=True)
    assert not authorize_batch(journal, 1)
    assert BatchOutbox(archive).read(ident)["state"] == "prepared"


def test_remote_revoke_blocks_dispatch_without_dropping_owner(tmp_path):
    journal, _archive = history(tmp_path)
    configure_batch(journal.policy_root, enabled=True)
    ident = prepare_next_batch(journal)
    write_policy(journal.policy_root, enabled=False, daily_usd=5, monthly_usd=50,
                 override_ceiling=False, fallback=lambda: (False, 1, 30, False))
    assert not authorize_batch(journal, 1)
    assert journal.historical_batch_owner()["id"] == ident


@pytest.mark.parametrize("window", [None, {"text": "   "}])
def test_private_or_empty_history_stays_local(tmp_path, monkeypatch, window):
    from muninn.history.cited_analysis_source import CitedAnalysisSource
    journal, archive = history(tmp_path)
    configure_batch(journal.policy_root, enabled=True)
    monkeypatch.setattr(CitedAnalysisSource, "remote_input", lambda *_: window)
    assert prepare_next_batch(journal) is None
    assert journal.historical_batch_owner() is None
    assert not (archive.root / "historical-batches.db").exists()


def test_batch_has_supported_output_cap_and_keeps_budget(tmp_path):
    from muninn.history.auto_routing import remote_policy_snapshot
    journal, archive = history(tmp_path)
    before = remote_policy_snapshot(journal.policy_root)
    configure_batch(journal.policy_root, enabled=True)
    ident = prepare_next_batch(journal)
    assert all(i["body"]["max_tokens"] == 2048 for i in BatchOutbox(archive).read(ident)["items"])
    assert remote_policy_snapshot(journal.policy_root) == before


@pytest.mark.asyncio
async def test_revoke_after_last_precheck_before_fence_never_posts(tmp_path):
    from muninn.history.batch_activation import authorize_transaction
    from muninn.history.historical_batch_worker import HistoricalBatchWorker
    from muninn.history.remote_accounting import AdmissionError, status
    from tests.test_historical_batch_worker import ready
    journal, _archive = history(tmp_path)
    configure_batch(journal.policy_root, enabled=True)
    ident = prepare_next_batch(journal)
    calls = 0
    def last_check(generation):
        nonlocal calls
        allowed = authorize_batch(journal, generation)
        calls += 1
        if calls == 2:
            configure_batch(journal.policy_root, enabled=False)
        return allowed  # Revocation lands AFTER the old check returned true.
    async def forbidden(*args, **kwargs):
        pytest.fail("Revoked retention must not POST")
    worker = HistoricalBatchWorker(journal, authorize_submit=last_check,
        authorize_transaction=authorize_transaction, send=forbidden, provider_status=ready)
    with pytest.raises(AdmissionError):
        await worker.step()
    assert status(journal.policy_root)["unresolved"] == 0
    assert BatchOutbox(journal.archive).read(ident)["state"] == "prepared"


def test_unsent_local_failure_is_readmitted_but_never_empty_failure(tmp_path):
    journal, _archive = history(tmp_path)
    job = journal.claim_analysis(include_capture=True, include_search=False)
    assert journal.fail_analysis(job.job_id, job.lease_token, "model_unavailable")
    empty = journal.claim_analysis(include_capture=True, include_search=False)
    assert journal.fail_analysis(empty.job_id, empty.lease_token, "insufficient_context")
    configure_batch(journal.policy_root, enabled=True)
    ident = prepare_next_batch(journal)
    items = BatchOutbox(journal.archive).read(ident)["items"]
    assert job.job_id in {item["job_id"] for item in items}
    assert empty.job_id not in {item["job_id"] for item in items}
