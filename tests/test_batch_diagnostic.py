"""Tiny synthetic temp-policy fixtures; no account, live archive or HTTP."""
import asyncio
import copy
import json

import pytest

from muninn.history.batch_activation import configure_batch
from muninn.history.batch_diagnostic import DiagnosticStore, request_body, verify_price, validate_results
from muninn.history.historical_batch import BatchError, MODEL
from muninn.history.portable_accounting import _validate
from muninn.history.remote_accounting import AdmissionError, _db, reserve, status
from muninn.history.run_accounting import run_status
from muninn.history.secure_archive import SecureHistoryArchive
from tests.test_remote_accounting import policy, READY


def setup(tmp_path):
    root = tmp_path / "policy"
    from muninn.history.private_acl import create_private_directory
    create_private_directory(root)
    remote = policy(root)
    retention = configure_batch(root, enabled=True, max_batches=10)
    parent = "a" * 32
    admission = reserve(root, remote.generation, READY, batch_owner=parent)
    admission.mark_unknown()
    archive = SecureHistoryArchive.create(tmp_path / "archive", "isolated synthetic recovery phrase")
    return DiagnosticStore(archive, root), admission, parent, retention["generation"]


def receipt(body, state="in_progress"):
    return {"id": "batch_diagnostic_fixture", "model": MODEL, "endpoint": "/v1/chat/completions",
        "completion_window": "24h", "status": state,
        "request_counts": {"total": 2, "completed": 0, "failed": 0}}


def terminal(body):
    result = receipt(body, "completed")
    result.update(request_counts={"total": 2, "completed": 2, "failed": 0},
        usage={"cost": .000123, "is_byok": False}, results=[])
    values = [{"ok": True}, {"summary": "Keep citations.", "decisions": [], "open_items": [],
        "uncertainty": "", "proposals": [{"type": "preference", "text": "Keep citations.",
            "quote": "keeping source citations", "start": 31}]}]
    for request, value in zip(body["requests"], values):
        result["results"].append({"custom_id": request["custom_id"], "error": None, "response": {
            "status_code": 200, "body": {"model": MODEL, "choices": [{"finish_reason": "stop",
                "message": {"content": json.dumps(value)}}]}}})
    return result


def test_fixed_request_order_shape_price_and_no_private_context():
    body = request_body()
    assert list(body) == ["endpoint", "model", "provider", "completion_window", "requests"]
    assert body["provider"] == {"only": ["openai"]}
    assert len({r["custom_id"] for r in body["requests"]}) == 2
    assert all(r["body"]["max_tokens"] == 512 for r in body["requests"])
    assert body["requests"][1]["body"]["response_format"]["type"] == "json_schema"
    catalog = {"data": {"endpoints": [{"provider_name": "OpenAI", "pricing": {
        "prompt": "0.0000004", "completion": "0.0000015"}}]}}
    assert 0 < verify_price(body, catalog) < .01
    catalog["data"]["endpoints"][0]["pricing"]["prompt"] = "1"
    with pytest.raises(BatchError, match="price_bound"):
        verify_price(body, catalog)


def test_one_scoped_probe_preserves_ordinary_guard_and_recovery(tmp_path):
    store, parent_admission, parent, retention = setup(tmp_path)
    body = request_body()
    ident = store.prepare(parent, 1, retention, READY, body)
    assert store.read(ident)["body"] == body
    assert status(store.root)["unresolved"] == 2
    with pytest.raises(AdmissionError):
        reserve(store.root, 1, READY)
    with pytest.raises(AdmissionError):
        store.prepare(parent, 1, retention, READY, body)
    with _db(store.root) as (db, _):
        assert _validate(db)
        assert "The synthetic project" not in str(db.execute("SELECT sealed FROM batch_diagnostics").fetchall())
    async def send(method, **kwargs):
        if method == "POST":
            assert store.read(ident)["state"] == "submission_unknown"
            assert status(store.root)["unresolved"] == 2
            return receipt(body)
        return terminal(body)
    assert asyncio.run(store.submit(ident, send, provider_status=lambda: READY))["provider_state"] == "in_progress"
    with pytest.raises((BatchError, AdmissionError)):
        asyncio.run(store.submit(ident, send, provider_status=lambda: READY))
    result = asyncio.run(store.poll(ident, send))
    assert result["validated_requests"] == 2
    assert result["actual_aggregate_cost_usd"] == .000123
    assert store.read(ident)["state"] == "terminal_saved"
    assert status(store.root)["unresolved"] == 1
    totals = run_status(store.root, since=0)
    assert totals["settled_cost_usd"] == .000123
    assert totals["diagnostics"]["settled_cost_usd"] == .000123
    assert totals["diagnostics"]["backlog_publications"] == 0
    asyncio.run(store.poll(ident, send))  # no aggregate bill duplication
    assert run_status(store.root, since=0)["settled_cost_usd"] == .000123
    parent_admission.settle_response({"usage": {"cost": .001}})
    with pytest.raises(AdmissionError):
        store.prepare(parent, 1, retention, READY, body)
    ordinary = reserve(store.root, 1, READY)
    ordinary.release_reserved()


def test_unknown_post_never_retries_and_parent_settlement_still_blocks(tmp_path):
    store, parent_admission, parent, retention = setup(tmp_path)
    ident = store.prepare(parent, 1, retention, READY, request_body())
    calls = []
    async def fail(method, **kwargs):
        calls.append(method)
        raise TimeoutError
    with pytest.raises(TimeoutError):
        asyncio.run(store.submit(ident, fail, provider_status=lambda: READY))
    with pytest.raises((BatchError, AdmissionError)):
        asyncio.run(store.submit(ident, fail, provider_status=lambda: READY))
    with pytest.raises(BatchError):
        asyncio.run(store.poll(ident, fail))
    assert calls == ["POST"]
    parent_admission.settle_response({"usage": {"cost": 0}})
    with pytest.raises(AdmissionError):
        reserve(store.root, 1, READY)


def test_failed_preimage_no_schema_or_new_admission(tmp_path, monkeypatch):
    store, _admission, parent, retention = setup(tmp_path)
    def fail(*args):
        raise OSError
    monkeypatch.setattr("muninn.history.batch_activation._backup_batch_policy", fail)
    with pytest.raises(AdmissionError):
        store.prepare(parent, 1, retention, READY, request_body())
    assert status(store.root)["unresolved"] == 1
    with _db(store.root) as (db, _):
        assert "diagnostic_parent" not in {r[1] for r in db.execute("PRAGMA table_info(remote_admissions)")}


def test_budget_and_revocation_fail_before_dispatch(tmp_path):
    store, _admission, parent, retention = setup(tmp_path)
    with pytest.raises(AdmissionError, match="headroom"):
        store.prepare(parent, 1, retention, {**READY, "usage_daily_usd": 4.999}, request_body())
    ident = store.prepare(parent, 1, retention, READY, request_body())
    configure_batch(store.root, enabled=False)
    async def never(*args, **kwargs):
        pytest.fail("No POST after revocation")
    with pytest.raises(AdmissionError):
        asyncio.run(store.submit(ident, never, provider_status=lambda: READY))


def test_exact_result_ids_and_quotes_not_acceptance(tmp_path):
    body = request_body()
    record = {"body": body, "response": terminal(body)}
    assert validate_results(record) == 2
    bad = copy.deepcopy(record)
    bad["response"]["results"][0]["custom_id"] = "f" * 32
    with pytest.raises(BatchError, match="binding"):
        validate_results(bad)
    bad = copy.deepcopy(record)
    content = json.loads(bad["response"]["results"][1]["response"]["body"]["choices"][0]["message"]["content"])
    content["proposals"][0]["start"] = 0
    bad["response"]["results"][1]["response"]["body"]["choices"][0]["message"]["content"] = json.dumps(content)
    assert validate_results(bad) == 1


def test_changed_index_and_arbitrary_input_fail_closed(tmp_path):
    from muninn.history.credential_crypto import VaultIntegrityError
    store, _admission, parent, retention = setup(tmp_path)
    arbitrary = request_body()
    arbitrary["requests"][0]["body"]["messages"][0]["content"] = "Not the fixed synthetic probe"
    with pytest.raises(BatchError, match="fixed_input"):
        store.prepare(parent, 1, retention, READY, arbitrary)
    store.prepare(parent, 1, retention, READY, request_body())
    with _db(store.root) as (db, _):
        db.execute("DROP INDEX one_diagnostic_per_parent")
    with _db(store.root) as (db, _):
        with pytest.raises(VaultIntegrityError, match="fences"):
            _validate(db)


def test_portable_snapshot_retains_probe_and_disables_dispatch(tmp_path):
    from muninn.history.portable_accounting import snapshot_into, restore_into, verify_snapshot
    from muninn.history.private_acl import create_private_directory
    from muninn.history.remote_policy import read_policy
    store, _admission, parent, retention = setup(tmp_path)
    ident = store.prepare(parent, 1, retention, READY, request_body())
    destination = tmp_path / "bundle"
    create_private_directory(destination)
    snapshot_into(store.archive, store.root, destination)
    verify_snapshot(store.archive, destination)
    restore = tmp_path / "restored"
    create_private_directory(restore)
    restore_into(store.archive, restore)
    assert not read_policy(restore, lambda: (False, 5, 50, False)).enabled
    assert DiagnosticStore(store.archive, restore).read(ident) == store.read(ident)
    assert status(restore)["unresolved"] == 2


def test_fresh_usage_and_received_mismatch_never_retry(tmp_path):
    store, _admission, parent, retention = setup(tmp_path)
    ident = store.prepare(parent, 1, retention, READY, request_body())
    calls = []
    async def send(method, **kwargs):
        calls.append(method)
        return {**receipt(request_body()), "model": "wrong-model"}
    with pytest.raises(AdmissionError):
        asyncio.run(store.submit(ident, send, provider_status=lambda: {**READY, "usage_daily_usd": 4.999}))
    assert not calls and store.read(ident)["state"] == "prepared"
    with pytest.raises(BatchError, match="identity"):
        asyncio.run(store.submit(ident, send, provider_status=lambda: READY))
    record = store.read(ident)
    assert record["state"] == "submission_unknown"
    assert record["response"]["model"] == "wrong-model"
    assert record["recovery_candidate"] == "batch_diagnostic_fixture"
    assert "provider_id" not in store.summary(record)
    with pytest.raises((AdmissionError, BatchError)):
        asyncio.run(store.submit(ident, send, provider_status=lambda: READY))
    assert calls == ["POST"]


def test_expired_watcher_does_not_even_get(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import probe_luna_batch as cli
    store, _admission, parent, retention = setup(tmp_path)
    ident = store.prepare(parent, 1, retention, READY, request_body())
    record = store.read(ident)
    monkeypatch.setattr(cli, "_local_setting", lambda name: str(store.root) if name == "MUNINN_DATA_DIR" else str(store.archive.root))
    monkeypatch.setattr(cli, "DiagnosticStore", lambda *_args: store)
    monkeypatch.setattr(cli.time, "time", lambda: record["created_at"] + 86401)
    async def never(*args, **kwargs):
        pytest.fail("No GET after watcher deadline")
    monkeypatch.setattr(cli, "transport", never)
    assert asyncio.run(cli.run(SimpleNamespace(status=None, poll=ident, watch=True))) == 2
