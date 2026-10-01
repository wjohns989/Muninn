"""Actual strict transport with temporary policy/accounting, no live providers."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from muninn.history import auto_routing, secure_analysis as analysis
from muninn.history.remote_policy import write_policy
from muninn.history.remote_accounting import status


def route(tmp_path, monkeypatch, post):
    write_policy(tmp_path, enabled=True, daily_usd=5, monthly_usd=50,
                 override_ceiling=False, fallback=lambda: (False, 5, 50, False))
    monkeypatch.setattr(analysis, "guarded_openrouter_available", lambda **kwargs: True)
    status = lambda **kwargs: {"state": "ready", "admission_ready": True,
        "key_limit_usd": 5, "key_remaining_usd": 5, "key_reset": "daily",
        "usage_daily_usd": 0, "usage_monthly_usd": 0}
    monkeypatch.setattr(auto_routing, "openrouter_key_status", status)
    monkeypatch.setattr(analysis, "openrouter_key_status", status, raising=False)
    monkeypatch.setattr(analysis.Provider, "from_env", lambda *args: analysis.Provider(
        "openrouter", "https://openrouter.ai/api/v1", ["fixture-model"], "fixture-key"))

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["trust_env"] is False
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def post(self, *args, **kwargs):
            assert kwargs["json"]["max_completion_tokens"] == 2048
            return await post()

    monkeypatch.setattr(analysis.httpx, "AsyncClient", Client)
    return SimpleNamespace(data_dir=tmp_path)


def response(cost=0.01, *, content=None):
    body = {"model": "fixture-model", "usage": {"cost": cost}, "choices": [{"message": {
        "content": content or json.dumps({"summary": "Use source citations.", "decisions": [],
            "open_items": [], "uncertainty": "No further evidence."})}}]}
    return SimpleNamespace(raise_for_status=lambda: None, json=lambda **kwargs: body)


async def run(history, **kwargs):
    return await analysis._analyze_window(history, "Use source citations.",
        allow_remote=True, prefer_remote=True, **kwargs)


@pytest.mark.asyncio
async def test_parallel_calls_cannot_spend_same_allowance(tmp_path, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def post():
        calls.append(True)
        if len(calls) == 1:
            entered.set()
            await release.wait()
        return response()
    history = route(tmp_path, monkeypatch, post)
    first = asyncio.create_task(run(history))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        second = await asyncio.wait_for(run(history), timeout=2)
        assert second["status"] == "deferred"
        assert second["reason"] == "remote_admission_busy"
        assert len(calls) == 1
    finally:
        release.set()
        await first


@pytest.mark.asyncio
async def test_unknown_post_outcome_blocks_next_call(tmp_path, monkeypatch):
    import httpx
    calls = []
    async def post():
        calls.append(True)
        raise httpx.ReadTimeout("fixture-only lost reply")
    history = route(tmp_path, monkeypatch, post)
    with pytest.raises(httpx.ReadTimeout):
        await run(history)
    second = await run(history)
    assert second["status"] == "deferred" and second["reason"] == "remote_admission_busy"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_valid_cost_settles_before_invalid_model_output(tmp_path, monkeypatch):
    async def post():
        return response(0.2, content="not valid model JSON")
    history = route(tmp_path, monkeypatch, post)
    with pytest.raises(analysis.ModelOutputInvalid):
        await run(history)
    assert status(tmp_path)["state"] == "ready"
    assert status(tmp_path)["daily_cost_usd"] == 0.2


@pytest.mark.asyncio
async def test_real_http_json_decimal_cost_never_rounds_down(tmp_path, monkeypatch):
    import httpx
    async def post():
        return httpx.Response(200, request=httpx.Request("POST", "https://fixture.invalid"),
            content=b'{"model":"fixture-model","usage":{"cost":4.9999990000000001},'
                    b'"choices":[{"message":{"content":"invalid model JSON"}}]}')
    history = route(tmp_path, monkeypatch, post)
    with pytest.raises(analysis.ModelOutputInvalid):
        await run(history)
    assert status(tmp_path)["daily_cost_usd"] == 5
    assert (await run(history))["reason"] == "remote_admission_threshold_reached"


@pytest.mark.asyncio
async def test_missing_cost_cannot_return_publishable_result_or_free_allowance(tmp_path, monkeypatch):
    calls = []
    async def post():
        calls.append(True)
        return response(None)
    history = route(tmp_path, monkeypatch, post)
    first = await run(history)
    assert first["status"] == "deferred"
    assert first["reason"] == "remote_cost_unresolved"
    assert status(tmp_path)["state"] == "blocked"
    assert (await run(history))["reason"] == "remote_admission_busy"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_capture_opt_out_after_marker_still_prevents_remote_post(tmp_path, monkeypatch):
    calls = []
    async def post():
        calls.append(True)
        return response()
    history = route(tmp_path, monkeypatch, post)
    monkeypatch.setattr(analysis, "_select_local", lambda base: (None, "gpu_busy"))
    marker = []
    async def before_remote():
        marker.append(True)
        return True
    async def remote_not_sent():
        return True
    result = await analysis._analyze_window(history, "Use source citations.",
        allow_remote=True, before_remote=before_remote,
        remote_not_sent=remote_not_sent, remote_gate=lambda: False)
    assert result["status"] == "deferred" and result["reason"] == "remote_consent_revoked"
    assert marker == [True] and calls == []
    assert status(tmp_path)["unresolved"] == 0


@pytest.mark.asyncio
async def test_revocation_during_post_still_settles_returned_cost(tmp_path, monkeypatch):
    async def post():
        write_policy(tmp_path, enabled=False, daily_usd=5, monthly_usd=50,
                     override_ceiling=False, fallback=lambda: (False, 5, 50, False))
        return response(0.3)
    history = route(tmp_path, monkeypatch, post)
    assert (await run(history))["status"] == "ok"
    assert status(tmp_path)["unresolved"] == 0
    assert status(tmp_path)["daily_cost_usd"] == 0.3


@pytest.mark.asyncio
async def test_revocation_during_lease_marker_proven_unsent_releases(tmp_path, monkeypatch):
    calls, cleared = [], []
    async def post():
        calls.append(True)
        return response()
    history = route(tmp_path, monkeypatch, post)
    async def marker():
        write_policy(tmp_path, enabled=False, daily_usd=5, monthly_usd=50,
                     override_ceiling=False, fallback=lambda: (False, 5, 50, False))
        return True
    async def unsent():
        cleared.append(True)
        return True
    result = await run(history, before_remote=marker, remote_not_sent=unsent)
    assert result["reason"] == "remote_consent_revoked"
    assert not calls and cleared == [True]
    assert status(tmp_path)["unresolved"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmed", [True, False])
async def test_lease_exception_releases_only_if_dispatch_marker_confirmed_unsent(tmp_path, monkeypatch, confirmed):
    async def post():
        pytest.fail("A failed lease must not dispatch")
    history = route(tmp_path, monkeypatch, post)
    async def marker():
        raise RuntimeError("fixture-only ambiguous lease update")
    async def unsent():
        return confirmed
    with pytest.raises(RuntimeError, match="ambiguous lease"):
        await run(history, before_remote=marker, remote_not_sent=unsent)
    assert status(tmp_path)["unresolved"] == (0 if confirmed else 1)


@pytest.mark.asyncio
async def test_cancellation_during_post_retains_unknown(tmp_path, monkeypatch):
    entered = asyncio.Event()
    async def post():
        entered.set()
        await asyncio.Future()
    history = route(tmp_path, monkeypatch, post)
    pending = asyncio.create_task(run(history))
    await asyncio.wait_for(entered.wait(), timeout=2)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert status(tmp_path)["unresolved"] == 1


@pytest.mark.asyncio
async def test_accounting_api_is_local_authenticated_and_has_no_identifier_or_key(tmp_path, monkeypatch):
    import httpx
    import server
    from muninn.history import llm_settings
    from muninn.history.remote_accounting import reserve
    write_policy(tmp_path, enabled=True, daily_usd=5, monthly_usd=50,
                 override_ceiling=False, fallback=lambda: (False, 5, 50, False))
    admission = reserve(tmp_path, 1, {"admission_ready": True, "usage_daily_usd": 0, "usage_monthly_usd": 0})
    token = "test-main-auth-token-aaaaaaaaaaaaaaaaaaaaaaaa"
    monkeypatch.setenv("MUNINN_AUTH_TOKEN", token)
    monkeypatch.setenv("MUNINN_NO_AUTH", "0")
    monkeypatch.setattr(server, "is_security_enabled", lambda: True)
    monkeypatch.setattr(server, "_require_history", lambda: SimpleNamespace(data_dir=tmp_path))
    monkeypatch.setattr(llm_settings, "api_key", lambda: pytest.fail("Accounting status must not read keys"))
    url = "/history/secure/remote-policy/accounting"
    local = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 1234))
    async with httpx.AsyncClient(transport=local, base_url="http://localhost") as client:
        assert (await client.get(url)).status_code == 401
        assert (await client.get(url, headers={"Authorization": "Bearer wrong"})).status_code == 401
        result = await client.get(url, headers={"Authorization": f"Bearer {token}"})
        assert result.status_code == 200 and result.json()["data"]["unresolved"] == 1
        assert result.headers["cache-control"] == "no-store"
        assert admission.identifier not in result.text and token not in result.text
    remote = httpx.ASGITransport(app=server.app, client=("192.168.1.2", 1234))
    async with httpx.AsyncClient(transport=remote, base_url="http://localhost") as client:
        assert (await client.get(url, headers={"Authorization": f"Bearer {token}"})).status_code == 404
