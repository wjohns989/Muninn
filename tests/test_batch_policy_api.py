"""Owner-local batch consent; isolated policy files, never live providers."""
import sqlite3
from types import SimpleNamespace

import httpx
import pytest

import server
from muninn.history.batch_activation import configure_batch, read_batch_policy
from muninn.history.remote_policy import write_policy, read_policy

TOKEN = "test-main-auth-token-aaaaaaaaaaaaaaaaaaaaaaaa"
ROUTE = "/history/secure/batch-policy"


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setenv("MUNINN_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv("MUNINN_NO_AUTH", "0")
    monkeypatch.setattr(server, "is_security_enabled", lambda: True)
    monkeypatch.setattr(server, "_require_history", lambda: SimpleNamespace(data_dir=tmp_path))
    write_policy(tmp_path, enabled=True, daily_usd=5, monthly_usd=50,
                 override_ceiling=False, fallback=lambda: (False, 1, 30, False))
    configure_batch(tmp_path, enabled=True, max_batches=10000)
    return tmp_path


@pytest.mark.asyncio
async def test_get_is_main_authenticated_local_read_only(configured):
    path = configured / "remote_policy" / "policy.sqlite3"
    before = path.read_bytes()
    transport = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 1234))
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        assert (await client.get(ROUTE)).status_code == 401
        response = await client.get(ROUTE, headers={"Authorization": f"Bearer {TOKEN}"})
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.json()["data"] == {"enabled": True, "generation": 1,
            "remaining_batches": 10000, "max_batches": 10000}
    assert path.read_bytes() == before
    assert not list(configured.glob("batch-policy-preimage-*"))
    remote = httpx.ASGITransport(app=server.app, client=("192.168.1.2", 1234))
    async with httpx.AsyncClient(transport=remote, base_url="http://localhost") as client:
        assert (await client.get(ROUTE, headers={"Authorization": f"Bearer {TOKEN}"})).status_code == 404


@pytest.mark.asyncio
async def test_revoke_backups_current_policy_and_stale_save_never_reapplies(configured):
    remote_before = read_policy(configured, lambda: (False, 1, 30, False))
    transport = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 1234))
    headers = {"Authorization": f"Bearer {TOKEN}", "Origin": "http://localhost"}
    body = {"enabled": False, "max_batches": 10000, "expected_generation": 1}
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        denied = await client.post(ROUTE, json=body, headers={**headers, "Origin": "http://evil.test"})
        assert denied.status_code == 403
        assert not list(configured.glob("batch-policy-preimage-*"))
        response = await client.post(ROUTE, json=body, headers=headers)
        assert response.status_code == 200
        assert response.json()["data"]["enabled"] is False
        assert response.json()["data"]["generation"] == 2
        stale = await client.post(ROUTE, json=body, headers=headers)
        assert stale.status_code == 409
    backups = list(configured.glob("batch-policy-preimage-*/policy.sqlite3"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as copy:
        assert copy.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert copy.execute("SELECT enabled,generation,max_batches FROM batch_policy").fetchone() == (1, 1, 10000)
    assert read_policy(configured, lambda: (False, 1, 30, False)) == remote_before
    assert read_batch_policy(configured)["generation"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", [{"enabled": "false"}, {"max_batches": True},
    {"max_batches": 0}, {"max_batches": 10001}, {"expected_generation": True},
    {"expected_generation": -1}, {"daily_usd": 10}])
async def test_policy_request_is_strict_and_has_no_budget_fields(configured, extra):
    transport = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 1234))
    body = {"enabled": False, "max_batches": 10000, "expected_generation": 1, **extra}
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        assert (await client.post(ROUTE, json=body,
            headers={"Authorization": f"Bearer {TOKEN}"})).status_code == 422
    assert read_batch_policy(configured)["generation"] == 1
    assert not list(configured.glob("batch-policy-preimage-*"))


def test_backup_failure_does_not_change_consent(configured, monkeypatch):
    from muninn.history import batch_activation
    before = read_batch_policy(configured)
    def unavailable(*_):
        raise OSError("synthetic private path MUST_NOT_EXPOSE")
    monkeypatch.setattr(batch_activation, "_backup_batch_policy", unavailable)
    with pytest.raises(OSError):
        configure_batch(configured, enabled=False, max_batches=10000,
                        expected_generation=1, backup_before=True)
    assert read_batch_policy(configured) == before


def test_remote_revoke_reenable_interleaving_blocks_stale_batch_enable(configured, monkeypatch):
    from muninn.history import batch_activation
    from muninn.history.remote_policy import PolicyError
    before = read_batch_policy(configured)
    original_snapshot = batch_activation.remote_policy_snapshot

    def intervening_remote_edit(root):
        observed = original_snapshot(root)
        for enabled in (False, True):
            write_policy(root, enabled=enabled, daily_usd=5, monthly_usd=50,
                         override_ceiling=False, fallback=lambda: (False, 1, 30, False))
        return observed

    monkeypatch.setattr(batch_activation, "remote_policy_snapshot", intervening_remote_edit)
    with pytest.raises(PolicyError):
        configure_batch(configured, enabled=True, max_batches=128,
                        expected_generation=1, backup_before=True)
    monkeypatch.setattr(batch_activation, "remote_policy_snapshot", original_snapshot)
    assert read_batch_policy(configured) == before
    assert not list(configured.glob("batch-policy-preimage-*"))


def test_first_policy_preimage_precedes_schema_creation(configured):
    database = configured / "remote_policy" / "policy.sqlite3"
    with sqlite3.connect(database) as db:
        for table in ("batch_policy", "batch_policy_audit", "batch_consent"):
            db.execute(f"DROP TABLE {table}")  # Isolated fixture only.
    result = configure_batch(configured, enabled=True, max_batches=128,
                             expected_generation=0, backup_before=True)
    assert result["generation"] == 1 and result["max_batches"] == 128
    backup = next(configured.glob("batch-policy-preimage-*/policy.sqlite3"))
    with sqlite3.connect(backup) as db:
        assert db.execute("SELECT 1 FROM sqlite_master WHERE name='batch_policy'").fetchone() is None


@pytest.mark.asyncio
async def test_live_remote_revocation_blocks_enable_but_not_batch_revoke(configured):
    write_policy(configured, enabled=False, daily_usd=5, monthly_usd=50,
                 override_ceiling=False, fallback=lambda: (False, 1, 30, False))
    transport = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 1234))
    headers = {"Authorization": f"Bearer {TOKEN}"}
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        body = {"enabled": True, "max_batches": 10000, "expected_generation": 1}
        assert (await client.post(ROUTE, json=body, headers=headers)).status_code == 503
        assert not list(configured.glob("batch-policy-preimage-*"))
        body["enabled"] = False
        assert (await client.post(ROUTE, json=body, headers=headers)).status_code == 200
