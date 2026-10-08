"""Synthetic isolated migration, stream framing and accounting tests only."""
import asyncio
import json

import pytest

from muninn.history.historical_batch import BatchError, MODEL
from muninn.history.portable_accounting import _validate
from muninn.history.remote_accounting import AdmissionError, _db, reserve, status
from muninn.history.run_accounting import run_status
from muninn.history.streaming_diagnostic import StreamReceipt, StreamingStore, request_body, MAX_RECEIPT_BYTES
from tests.test_batch_diagnostic import setup, receipt
from tests.test_remote_accounting import READY


def accepted_batch(tmp_path):
    from muninn.history.batch_diagnostic import request_body as batch_body
    base, parent_admission, parent, retention = setup(tmp_path)
    ident = base.prepare(parent, 1, retention, READY, batch_body())
    async def send(*args, **kwargs):
        return receipt(batch_body())
    asyncio.run(base.submit(ident, send, provider_status=lambda: READY))
    return StreamingStore(base.archive, base.root), parent, retention, ident


def streamed(*, text="1 2 3 4 5", usage=True, done=True):
    collector = StreamReceipt()
    collector.value["http_status"] = 200
    events = [{"id": "gen-fixture", "model": MODEL,
        "choices": [{"delta": {"content": text}, "finish_reason": "stop"}]}]
    if usage:
        events.append({"id": "gen-fixture", "model": MODEL, "choices": [], "usage": {"cost": .000123, "is_byok": False}})
    wire = ": keepalive\r\n\r\n" + "".join("data: " + json.dumps(e) + "\r\n\r\n" for e in events)
    if done:
        wire += "data: [DONE]\r\n\r\n"
    for byte in wire.encode():  # Every byte boundary, including CRLF framing.
        collector.feed(bytes([byte]))
    return collector.finish()


def test_split_sse_usage_keepalive_and_bound():
    assert request_body()["provider"]["only"] == ["azure"]
    response = streamed()
    assert response["done"] and response["identity_valid"]
    assert response["comments"] == 1 and len(response["content_chunks"]) == 1
    assert response["usage"]["cost"] == .000123
    collector = StreamReceipt()
    with pytest.raises(BatchError, match="bound"):
        collector.feed(b"a" * (MAX_RECEIPT_BYTES + 1))
    collector = StreamReceipt()
    collector.feed(b'data: {"id":"gen-fixture",\ndata: "model":"' + MODEL.encode() + b'", "choices":[]}\n\n')
    assert collector.finish()["identity_valid"]


def test_legacy_batch_migration_portable_fences_and_kind_binding(tmp_path):
    store, parent, retention, batch = accepted_batch(tmp_path)
    old = store.read(batch)
    old.pop("kind")
    with _db(store.root) as (db, _):
        store._save(db, old)
        db.execute("DROP INDEX one_diagnostic_admission")
        db.execute("DROP INDEX one_diagnostic_per_parent")
        db.execute("ALTER TABLE remote_admissions DROP COLUMN diagnostic_kind")
        db.execute("CREATE UNIQUE INDEX one_diagnostic_admission ON remote_admissions((1)) WHERE state IN ('reserved','unknown') AND diagnostic_parent IS NOT NULL")
        db.execute("CREATE UNIQUE INDEX one_diagnostic_per_parent ON remote_admissions(diagnostic_parent) WHERE diagnostic_parent IS NOT NULL")
    ident = store.prepare(parent, 1, retention, READY, request_body(), kind="streaming")
    assert store.read(batch)["kind"] == "batch" and store.read(batch)["response"] == old["response"]
    assert store.read(ident)["kind"] == "streaming"
    with _db(store.root) as (db, _):
        assert _validate(db)
    from muninn.history.portable_accounting import snapshot_into, restore_into, verify_snapshot
    from muninn.history.private_acl import create_private_directory
    bundle, restored = tmp_path / "bundle", tmp_path / "restored"
    create_private_directory(bundle)
    snapshot_into(store.archive, store.root, bundle)
    verify_snapshot(store.archive, bundle)
    create_private_directory(restored)
    restore_into(store.archive, restored)
    assert StreamingStore(store.archive, restored).read(ident) == store.read(ident)
    record = store.read(ident)
    record["kind"] = "batch"
    with _db(store.root) as (db, _):
        store._save(db, record)
    with pytest.raises(AdmissionError):
        store.read(ident)


def test_one_stream_no_resubmit_and_cost_once_even_bad_content(tmp_path):
    store, parent, retention, batch = accepted_batch(tmp_path)
    ident = store.prepare(parent, 1, retention, READY, request_body(), kind="streaming")
    assert status(store.root)["unresolved"] == 3
    with pytest.raises(AdmissionError):
        reserve(store.root, 1, READY)
    calls = []
    async def send(body):
        calls.append(body)
        assert store.read(ident)["state"] == "submission_unknown"
        return streamed(text="wrong content")
    result = asyncio.run(store.submit_stream(ident, send, provider_status=lambda: READY))
    assert result["billing_settled"] and not result["validated_output"]
    assert result["actual_cost_usd"] == .000123
    store.reconcile(ident)
    totals = run_status(store.root, since=0)
    assert totals["settled_cost_usd"] == .000123
    assert totals["diagnostics"]["by_kind"]["streaming"]["settled_cost_usd"] == .000123
    assert totals["diagnostics"]["by_kind"]["batch"]["admission_states"] == {"unknown": 1}
    with pytest.raises((BatchError, AdmissionError)):
        asyncio.run(store.submit_stream(ident, send, provider_status=lambda: READY))
    with pytest.raises(AdmissionError):
        store.prepare(parent, 1, retention, READY, request_body(), kind="streaming")
    with pytest.raises(AdmissionError):
        reserve(store.root, 1, READY)
    assert len(calls) == 1


def test_missing_usage_midstream_error_and_timeout_unknown(tmp_path):
    store, parent, retention, batch = accepted_batch(tmp_path)
    ident = store.prepare(parent, 1, retention, READY, request_body(), kind="streaming")
    async def send(body):
        value = streamed(usage=False, done=False)
        value["error"] = "stream_timeout"
        return value
    result = asyncio.run(store.submit_stream(ident, send, provider_status=lambda: READY))
    assert not result["billing_settled"] and not result["validated_output"]
    assert result["content_chunks"] == 1 and status(store.root)["unresolved"] == 3
    with pytest.raises((BatchError, AdmissionError)):
        asyncio.run(store.submit_stream(ident, send, provider_status=lambda: READY))
    collector = StreamReceipt()
    collector.feed(b'data: {"error":{"message":"fixture"}}\n\ndata: [DONE]\n\n')
    assert collector.finish()["error"] == "provider_stream_error"


def test_unaccepted_or_unrelated_batch_hold_and_preimage_failure(tmp_path, monkeypatch):
    store, parent, retention, batch = accepted_batch(tmp_path)
    with _db(store.root) as (db, _):
        record = store._read(db, batch)
        record["state"] = "submission_unknown"
        store._save(db, record)
    with pytest.raises(AdmissionError):
        store.prepare(parent, 1, retention, READY, request_body(), kind="streaming")
    record["state"] = "submitted"
    with _db(store.root) as (db, _):
        store._save(db, record)
        db.execute("UPDATE remote_admissions SET generation=2 WHERE id=?", (batch,))
    with pytest.raises(AdmissionError):
        store.prepare(parent, 1, retention, READY, request_body(), kind="streaming")


def test_stream_transport_preserves_received_chunks_on_timeout(monkeypatch):
    from muninn.history import llm_settings
    from muninn.history.streaming_diagnostic import transport
    monkeypatch.setattr(llm_settings, "api_key", lambda: "synthetic-fixture-only")
    class Reply:
        status_code = 200
        headers = {"content-type": "text/event-stream", "X-Generation-Id": "header-correlation"}
        async def __aenter__(self): return self
        async def __aexit__(self, *_): pass
        async def aiter_bytes(self):
            yield ('data: ' + json.dumps({"id": "gen-fixture", "model": MODEL,
                "choices": [{"delta": {"content": "1 2"}}]}) + '\n\n').encode()
            raise TimeoutError
    class Client:
        def __init__(self, **kwargs): assert not kwargs["trust_env"] and not kwargs["follow_redirects"]
        async def __aenter__(self): return self
        async def __aexit__(self, *_): pass
        def stream(self, method, url, **kwargs):
            assert method == "POST" and url == "https://openrouter.ai/api/v1/chat/completions"
            assert json.loads(kwargs["content"])["provider"]["zdr"]
            return Reply()
    monkeypatch.setattr("muninn.history.streaming_diagnostic.httpx.AsyncClient", Client)
    response = asyncio.run(transport(request_body()))
    assert response["content_chunks"] == ["1 2"] and response["error"] == "stream_timeout"
    assert response["usage"] is None
    assert response["transport_generation_id"] == "header-correlation"
    assert response["generation_id"] == "gen-fixture" and response["identity_valid"]


def test_byok_missing_or_true_keeps_unknown(tmp_path):
    store, parent, retention, _batch = accepted_batch(tmp_path)
    ident = store.prepare(parent, 1, retention, READY, request_body(), kind="streaming")
    async def send(body):
        value = streamed()
        value["usage"].pop("is_byok")
        return value
    result = asyncio.run(store.submit_stream(ident, send, provider_status=lambda: READY))
    assert result["validated_output"] and not result["billing_settled"]
    assert status(store.root)["unresolved"] == 3
    assert not store.reconcile(ident)["billing_settled"]


def test_two_allowed_models_cannot_settle(tmp_path):
    from muninn.history.historical_batch import MODEL_IDENTITIES
    collector = StreamReceipt()
    for model in sorted(MODEL_IDENTITIES):
        collector.feed(("data: " + json.dumps({"id": "gen-fixture", "model": model,
            "choices": [], "usage": {"cost": .000123, "is_byok": False}}) + "\n\n").encode())
    value = collector.finish()
    assert not value["identity_valid"]
    store, parent, retention, _batch = accepted_batch(tmp_path)
    ident = store.prepare(parent, 1, retention, READY, request_body(), kind="streaming")
    async def send(body): return value
    result = asyncio.run(store.submit_stream(ident, send, provider_status=lambda: READY))
    assert not result["billing_settled"] and status(store.root)["unresolved"] == 3


def test_parent_finishes_while_accepted_diagnostic_waits(tmp_path):
    from muninn.history.remote_accounting import Admission
    store, parent, retention, _batch = accepted_batch(tmp_path)
    with _db(store.root) as (db, _):
        parent_id = db.execute("SELECT id FROM remote_admissions WHERE batch_owner=?", (parent,)).fetchone()[0]
    Admission(store.root, parent_id, 1).settle_response({"usage": {"cost": .001}})
    assert status(store.root)["unresolved"] == 1
    ident = store.prepare(parent, 1, retention, READY, request_body(), kind="streaming")
    async def send(body): return streamed()
    result = asyncio.run(store.submit_stream(ident, send, provider_status=lambda: READY))
    assert result["validated_output"] and result["billing_settled"]
    assert status(store.root)["unresolved"] == 1
    with pytest.raises(AdmissionError):
        reserve(store.root, 1, READY)
