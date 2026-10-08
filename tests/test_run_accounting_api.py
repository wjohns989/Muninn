"""New cost reader requires actual local main authentication; no live effects."""
from types import SimpleNamespace

import httpx
import pytest

import server
from tests.test_run_accounting import READY, setup
from muninn.history.remote_accounting import reserve

TOKEN = "test-main-run-accounting-token-aaaaaaaaaaaaaaaa"
ROUTE = "/history/secure/remote-policy/accounting/run"


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setenv("MUNINN_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv("MUNINN_NO_AUTH", "0")
    monkeypatch.setattr(server, "is_security_enabled", lambda: True)
    monkeypatch.setattr(server, "_require_history", lambda: SimpleNamespace(data_dir=tmp_path))
    setup(tmp_path)
    paid = reserve(tmp_path, 1, READY)
    paid.mark_unknown()
    paid.settle_response({"usage": {"cost": .1234567}})
    return tmp_path


@pytest.mark.asyncio
async def test_main_authenticated_local_get_is_value_free_and_read_only(configured):
    path = configured / "remote_policy" / "policy.sqlite3"
    before = path.read_bytes()
    headers = {"Authorization": f"Bearer {TOKEN}"}
    transport = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 1234))
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        assert (await client.get(ROUTE, params={"since": 0})).status_code == 401
        response = await client.get(ROUTE, params={"since": 0}, headers=headers)
        assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
        data = response.json()["data"]
        assert data["settled_cost_usd"] == .123457
        assert "id" not in data and "admissions" not in data and TOKEN not in response.text
        for since in ("nan", "inf", "-1", "999999999999999"):
            assert (await client.get(ROUTE, params={"since": since}, headers=headers)).status_code == 422
    assert path.read_bytes() == before
    remote = httpx.ASGITransport(app=server.app, client=("192.168.1.2", 1234))
    async with httpx.AsyncClient(transport=remote, base_url="http://localhost") as client:
        assert (await client.get(ROUTE, params={"since": 0}, headers=headers)).status_code == 404


@pytest.mark.asyncio
async def test_failure_never_echoes_private_paths_or_provider_detail(configured, monkeypatch):
    from muninn.history import run_accounting
    from muninn.history.remote_accounting import AdmissionError
    def unavailable(*args, **kwargs):
        raise AdmissionError("PRIVATE_PATH_MUST_NOT_APPEAR")
    monkeypatch.setattr(run_accounting, "run_status", unavailable)
    transport = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 1234))
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        response = await client.get(ROUTE, params={"since": 0}, headers={"Authorization": f"Bearer {TOKEN}"})
        assert response.status_code == 503
        assert "PRIVATE_PATH" not in response.text


@pytest.mark.asyncio
async def test_missing_and_nonnumeric_cutoff_validation_is_static_and_no_store(configured):
    transport = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 1234))
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        for params in ({}, {"since": "PRIVATE_NOT_NUMERIC<img>"}):
            response = await client.get(ROUTE, params=params, headers={"Authorization": f"Bearer {TOKEN}"})
            assert response.status_code == 422
            assert response.json() == {"detail": "Invalid run interval"}
            assert response.headers["cache-control"] == "no-store"
