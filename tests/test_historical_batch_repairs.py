"""Failed-only batch repair contracts; synthetic encrypted fixtures, no network."""

import json
import sqlite3
from copy import deepcopy

import pytest

from muninn.history.batch_activation import (
    authorize_batch,
    authorize_transaction,
    bind_consent,
    configure_batch,
    read_batch_policy,
)
from muninn.history.capture_journal import CaptureJournal
from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.historical_batch import BatchError, BatchOutbox
from muninn.history.historical_batch_worker import HistoricalBatchWorker
from muninn.history.remote_accounting import Admission, status
from muninn.history.secure_archive import SecureHistoryArchive
from tests.test_historical_batch_jobs import fixture
from tests.test_historical_batch_worker import ready, responses


def _invalid_quote(terminal, index=0):
    message = terminal["results"][index]["response"]["body"]["choices"][0]["message"]
    content = json.loads(message["content"])
    content["proposals"][0].update(quote="NONEXISTENT_SYNTHETIC_QUOTE", start=0)
    message["content"] = json.dumps(content)


def _setup(tmp_path):
    journal, archive, outbox, parent, bindings = fixture(tmp_path)
    configure_batch(journal.policy_root, enabled=True, max_batches=10)
    policy = read_batch_policy(journal.policy_root)
    bind_consent(journal, outbox, parent, policy)
    journal.reserve_historical_batch(parent)
    return journal, archive, outbox, parent, bindings


class FakeProvider:
    def __init__(self, journal, archive, outbox, parent, *, invalid_repairs=False,
                 lose_repair=False, revoke_after_parent=False):
        self.journal, self.archive, self.outbox, self.parent = journal, archive, outbox, parent
        self.invalid_repairs = invalid_repairs
        self.lose_repair = lose_repair
        self.revoke_after_parent = revoke_after_parent
        self.posts, self.gets, self.terminals = [], [], {}
        self.original_items = deepcopy(outbox.read(parent)["items"])
        self.original_terminal = None

    async def send(self, method, provider_id=None, body=None):
        if method == "GET":
            assert body is None and provider_id in self.terminals
            self.gets.append(provider_id)
            terminal = self.terminals[provider_id]
            if self.revoke_after_parent and terminal == self.original_terminal:
                configure_batch(self.journal.policy_root, enabled=False, max_batches=10)
            return deepcopy(terminal)
        assert method == "POST" and provider_id is None
        request_ids = {item["custom_id"] for item in body["requests"]}
        parent = self.outbox.read(self.parent)
        records = [parent] + self.outbox.repair_records(parent)
        matching = [record for record in records
                    if {item["custom_id"] for item in record["items"]} == request_ids]
        assert len(matching) == 1
        record = matching[0]
        assert record["state"] == "submission_unknown"
        assert self.journal.historical_batch_owner()["id"] == self.parent
        assert self.journal.historical_batch_owner()["phase"] == "sent"
        if record["id"] != self.parent:
            assert len(record["items"]) == 1
            child_item = record["items"][0]
            failed = self.original_items[0]
            assert child_item["job_id"] == failed["job_id"]
            assert child_item["window"] == failed["window"]
            assert child_item["body"] == failed["body"]
            assert child_item["custom_id"] not in {i["custom_id"] for i in self.original_items}
        submitted, terminal = responses(self.archive, self.outbox, record["id"])
        number = len(self.posts) + 1
        submitted["id"] = terminal["id"] = f"batch_repair_fixture_{number}"
        terminal["usage"]["cost"] = number / 1000
        if record["id"] == self.parent or self.invalid_repairs:
            _invalid_quote(terminal)
        self.posts.append((record["id"], deepcopy(body)))
        self.terminals[submitted["id"]] = deepcopy(terminal)
        if record["id"] == self.parent:
            self.original_terminal = deepcopy(terminal)
        elif self.lose_repair:
            raise TimeoutError("synthetic lost repair response")
        return submitted


def _worker(journal, provider, clock):
    return HistoricalBatchWorker(
        journal, authorize_submit=lambda generation: authorize_batch(journal, generation),
        authorize_transaction=authorize_transaction, send=provider.send,
        provider_status=ready, clock=lambda: clock[0],
    )


async def _steps(worker, clock, predicate, *, limit=12):
    for _ in range(limit):
        await worker.step()
        if predicate():
            return
        clock[0] += 61
    pytest.fail("repair did not reach the bounded expected checkpoint")


def _publication(journal, job_id):
    with journal._connect() as db:
        row = db.execute("SELECT state,sealed_extraction,sealed_receipt,sealed_result "
                         "FROM history_analysis_jobs WHERE job_id=?", (job_id,)).fetchone()
    return tuple(row)


async def _completed_repair(tmp_path):
    journal, archive, outbox, parent, bindings = _setup(tmp_path)
    provider = FakeProvider(journal, archive, outbox, parent)
    clock = [0]
    worker = _worker(journal, provider, clock)
    await _steps(worker, clock, lambda: _publication(journal, bindings[1][0])[0] == "succeeded")
    sibling = _publication(journal, bindings[1][0])
    assert journal.historical_batch_owner()["phase"] == "sent"
    clock[0] += 61
    await _steps(worker, clock, lambda: journal.historical_batch_owner()["phase"] == "passed")
    assert _publication(journal, bindings[1][0]) == sibling
    return journal, archive, outbox, parent, bindings, provider


@pytest.mark.asyncio
async def test_partial_repair_preserves_original_bill_reply_and_successful_sibling(tmp_path):
    journal, _archive, outbox, parent, bindings, provider = await _completed_repair(tmp_path)
    record = outbox.read(parent)
    children = outbox.repair_records(record)
    assert record["items"] == provider.original_items
    assert record["terminal"] == provider.original_terminal
    assert len(children) == 1 and len(provider.posts) == 2
    assert [ident for ident, _body in provider.posts] == [parent, children[0]["id"]]
    assert len({request["custom_id"] for _ident, body in provider.posts
                for request in body["requests"]}) == 3
    assert status(journal.policy_root)["daily_cost_usd"] == pytest.approx(0.003)
    assert status(journal.policy_root)["unresolved"] == 0
    with sqlite3.connect(journal.policy_root / "remote_policy" / "policy.sqlite3") as db:
        rows = db.execute("SELECT id,batch_owner,generation,state,resolution,cost_micro "
                          "FROM remote_admissions ORDER BY cost_micro").fetchall()
    assert [(r[1], r[2], r[3], r[4], r[5]) for r in rows] == [
        (parent, 1, "settled", "response", 1000),
        (children[0]["id"], 1, "settled", "response", 2000),
    ]
    assert children[0]["repair_admission"] == rows[1][0]
    assert journal.verify_publications(job_ids=[job for job, _ in bindings]) == 2
    journal.verify_all()


@pytest.mark.asyncio
async def test_unknown_child_survives_restart_without_any_resend(tmp_path):
    journal, archive, outbox, parent, bindings = _setup(tmp_path)
    provider = FakeProvider(journal, archive, outbox, parent, lose_repair=True)
    clock = [0]
    worker = _worker(journal, provider, clock)
    with pytest.raises(TimeoutError, match="synthetic lost repair"):
        await _steps(worker, clock, lambda: False)
    child = outbox.repair_records(outbox.read(parent))[0]
    assert child["state"] == "submission_unknown"
    assert len(provider.posts) == 2 and status(journal.policy_root)["unresolved"] == 1
    assert status(journal.policy_root)["daily_cost_usd"] == pytest.approx(0.001)
    sibling = _publication(journal, bindings[1][0])
    recovered = CaptureJournal(archive)
    worker = _worker(recovered, provider, clock)
    for _ in range(3):
        clock[0] += 61
        await worker.step()
    assert len(provider.posts) == 2
    assert len(outbox.repair_records(outbox.read(parent))) == 1
    assert recovered.historical_batch_owner()["id"] == parent
    assert recovered.historical_batch_owner()["phase"] == "sent"
    assert _publication(recovered, bindings[1][0]) == sibling
    assert status(journal.policy_root)["unresolved"] == 1


@pytest.mark.asyncio
async def test_saved_child_terminal_resumes_settlement_after_restart(tmp_path, monkeypatch):
    journal, archive, outbox, parent, bindings = _setup(tmp_path)
    provider = FakeProvider(journal, archive, outbox, parent)
    clock = [0]
    worker = _worker(journal, provider, clock)
    original_settle = Admission.settle_response

    def interrupt_child_settlement(admission, reply):
        children = outbox.repair_records(outbox.read(parent))
        if children and admission.identifier == children[-1]["repair_admission"]:
            raise RuntimeError("synthetic interruption before child settlement")
        return original_settle(admission, reply)

    with monkeypatch.context() as patch:
        patch.setattr(Admission, "settle_response", interrupt_child_settlement)
        with pytest.raises(RuntimeError, match="synthetic interruption"):
            await _steps(worker, clock, lambda: False)
    child = outbox.repair_records(outbox.read(parent))[0]
    assert child["state"] == "terminal_saved"
    assert len(provider.posts) == 2
    assert status(journal.policy_root)["unresolved"] == 1
    sibling = _publication(journal, bindings[1][0])
    recovered = CaptureJournal(archive)
    restarted = _worker(recovered, provider, clock)
    clock[0] += 61
    await _steps(restarted, clock, lambda: recovered.historical_batch_owner()["phase"] == "passed")
    assert len(provider.posts) == 2
    assert _publication(recovered, bindings[1][0]) == sibling
    assert status(recovered.policy_root)["unresolved"] == 0
    assert status(recovered.policy_root)["daily_cost_usd"] == pytest.approx(0.003)


@pytest.mark.asyncio
async def test_two_failed_repair_rounds_hold_checkpoint_without_fourth_post(tmp_path):
    journal, archive, outbox, parent, bindings = _setup(tmp_path)
    provider = FakeProvider(journal, archive, outbox, parent, invalid_repairs=True)
    clock = [0]
    worker = _worker(journal, provider, clock)
    for _ in range(12):
        await worker.step()
        clock[0] += 61
    children = outbox.repair_records(outbox.read(parent))
    assert len(children) == 2 and len(provider.posts) == 3
    assert all(child["state"] == "terminal_saved" for child in children)
    assert journal.historical_batch_owner()["phase"] == "sent"
    assert worker.status["state"] not in {"passed", "idle"}
    assert journal.verify_publications(job_ids=[bindings[1][0]]) == 1
    assert status(journal.policy_root)["daily_cost_usd"] == pytest.approx(0.006)
    assert status(journal.policy_root)["unresolved"] == 0


@pytest.mark.asyncio
async def test_retention_revocation_blocks_child_post_but_keeps_original_publication(tmp_path):
    journal, archive, outbox, parent, bindings = _setup(tmp_path)
    provider = FakeProvider(journal, archive, outbox, parent, revoke_after_parent=True)
    clock = [0]
    worker = _worker(journal, provider, clock)
    for _ in range(5):
        await worker.step()
        clock[0] += 61
    assert [ident for ident, _body in provider.posts] == [parent]
    assert not read_batch_policy(journal.policy_root)["enabled"]
    assert journal.historical_batch_owner()["phase"] == "sent"
    assert journal.verify_publications(job_ids=[bindings[1][0]]) == 1
    assert status(journal.policy_root)["daily_cost_usd"] == pytest.approx(0.001)
    assert status(journal.policy_root)["unresolved"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["parent", "window", "admission"])
async def test_repaired_checkpoint_rejects_authenticated_cross_store_mismatch(tmp_path, defect):
    journal, _archive, outbox, parent, _bindings, _provider = await _completed_repair(tmp_path)
    child = outbox.repair_records(outbox.read(parent))[0]
    if defect == "parent":
        child["repair_parent"] = "f" * 32
    elif defect == "window":
        child["items"][0]["window"]["offset"] += 1
    else:
        # Correctly shaped and settled, but paid for the original, not the child.
        with sqlite3.connect(journal.policy_root / "remote_policy" / "policy.sqlite3") as db:
            child["repair_admission"] = db.execute(
                "SELECT id FROM remote_admissions WHERE batch_owner=?", (parent,),
            ).fetchone()[0]
    with sqlite3.connect(outbox.path) as db:
        db.execute("UPDATE batches SET sealed=? WHERE id=?", (outbox._seal(child), child["id"]))
    with pytest.raises((BatchError, VaultIntegrityError)):
        journal.verify_all()


@pytest.mark.asyncio
async def test_passed_repair_checkpoint_verifies_after_portable_restore(tmp_path):
    journal, archive, outbox, parent, bindings, _provider = await _completed_repair(tmp_path)
    other = tmp_path / "fresh-parent"
    other.mkdir()
    archive.backup_to(other / "backup")
    restored = SecureHistoryArchive.restore_from_backup(
        other / "backup", other / "restored", "test-only portable passphrase",
    )
    recovered = CaptureJournal(restored, recover=False)
    assert recovered.historical_batch_owner()["id"] == parent
    assert recovered.historical_batch_owner()["phase"] == "passed"
    recovered.verify_all()
    assert recovered.verify_publications(job_ids=[job for job, _ in bindings]) == 2
    assert BatchOutbox(restored).read(parent)["terminal"] == outbox.read(parent)["terminal"]
    assert BatchOutbox(restored).verify_all() == {"batches": 2}
    assert status(recovered.policy_root)["daily_cost_usd"] == pytest.approx(0.003)
    assert not read_batch_policy(recovered.policy_root)["enabled"]
