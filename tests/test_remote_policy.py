"""Persisted ZDR consent is local, auditable, and fail-closed."""

import sqlite3
from contextlib import closing
from types import SimpleNamespace

import httpx
import pytest

import server
from muninn.history.private_acl import verify_private
from muninn.history.remote_policy import PolicyError, read_policy, write_policy


def test_legacy_settings_only_apply_before_policy_is_managed(tmp_path):
    def fallback():
        return True, 1.0, 20.0, False
    assert not read_policy(tmp_path, fallback).enabled
    assert read_policy(tmp_path, fallback).source == "unconfigured"
    disabled = write_policy(tmp_path, enabled=False, daily_usd=1, monthly_usd=20,
                            override_ceiling=False, fallback=fallback)
    assert disabled.generation == 1
    assert not read_policy(tmp_path, fallback).enabled
    assert not read_policy(tmp_path, lambda: (True, 9.0, 90.0, True)).enabled
    for path in (tmp_path / "remote_policy", tmp_path / "remote_policy" / "managed",
                 tmp_path / "remote_policy" / "policy.sqlite3"):
        verify_private(path)
    with sqlite3.connect(tmp_path / "remote_policy" / "policy.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM audit").fetchone()[0] == 1


def test_missing_or_corrupt_managed_policy_never_revives_legacy_consent(tmp_path):
    def fallback():
        return True, 1.0, 20.0, False
    write_policy(tmp_path, enabled=False, daily_usd=1, monthly_usd=20,
                 override_ceiling=False, fallback=fallback)
    database = tmp_path / "remote_policy" / "policy.sqlite3"
    database.unlink()
    with pytest.raises(PolicyError):
        read_policy(tmp_path, fallback)


def test_whole_policy_directory_loss_and_invalid_cap_row_fail_closed(tmp_path):
    from muninn.history import auto_routing

    def fallback():
        return True, 1.0, 20.0, False

    write_policy(tmp_path, enabled=False, daily_usd=1, monthly_usd=20,
                 override_ceiling=False, fallback=fallback)
    database = tmp_path / "remote_policy" / "policy.sqlite3"
    with closing(sqlite3.connect(database)) as db:
        db.execute("UPDATE policy SET daily_usd=11 WHERE id=1")
        db.commit()
    with pytest.raises(PolicyError):
        read_policy(tmp_path, fallback)
    assert not auto_routing.remote_policy_snapshot(tmp_path).enabled
    # Simulate loss of the entire managed directory without a recursive delete.
    database.unlink()
    (tmp_path / "remote_policy" / "managed").unlink()
    (tmp_path / "remote_policy").rmdir()
    assert not read_policy(tmp_path, fallback).enabled
    assert not auto_routing.remote_policy_snapshot(tmp_path).enabled


def test_managed_revocation_closes_both_consent_and_budget_with_legacy_env_enabled(tmp_path, monkeypatch):
    from muninn.history import auto_routing, llm_settings, secure_analysis

    monkeypatch.setattr(auto_routing, "_local_setting", lambda _name: "1")
    monkeypatch.setattr(llm_settings, "api_key", lambda: "fixture-only-key")
    write_policy(tmp_path, enabled=False, daily_usd=1, monthly_usd=20,
                 override_ceiling=False, fallback=lambda: (True, 1.0, 20.0, False))
    monkeypatch.setattr(auto_routing.httpx, "Client",
                        lambda **_kwargs: pytest.fail("revoked policy must not query provider"))
    assert not auto_routing.remote_policy_snapshot(tmp_path).enabled
    assert auto_routing.openrouter_budget_ceiling(tmp_path) == (0.0, 0.0)
    assert not auto_routing.guarded_openrouter_available(policy_root=tmp_path)
    assert not secure_analysis._remote_eligible("pertinent transcript", allow_remote=True,
                                                 policy_root=tmp_path)


def test_budget_above_default_ceiling_needs_explicit_override(tmp_path):
    def fallback():
        return False, 1.0, 20.0, False
    with pytest.raises(ValueError):
        write_policy(tmp_path, enabled=True, daily_usd=11, monthly_usd=20,
                     override_ceiling=False, fallback=fallback)
    saved = write_policy(tmp_path, enabled=True, daily_usd=11, monthly_usd=120,
                         override_ceiling=True, fallback=fallback)
    assert (saved.daily_usd, saved.monthly_usd, saved.generation) == (11, 120, 1)
    assert read_policy(tmp_path, fallback) == saved


def test_relative_data_directory_uses_same_policy_file(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.chdir(tmp_path)
    Path("data").mkdir()

    def fallback():
        return False, 1.0, 20.0, False

    written = write_policy(Path("data"), enabled=True, daily_usd=1,
                           monthly_usd=20, override_ceiling=False, fallback=fallback)
    assert read_policy(tmp_path / "data", fallback) == written


@pytest.mark.asyncio
async def test_policy_api_requires_main_token_loopback_and_same_browser_origin(tmp_path, monkeypatch):
    token = "test-main-auth-token-aaaaaaaaaaaaaaaaaaaaaaaa"
    monkeypatch.setenv("MUNINN_AUTH_TOKEN", token)
    monkeypatch.setenv("MUNINN_NO_AUTH", "0")
    monkeypatch.setenv("MUNINN_ALLOWED_ORIGINS", "*")
    monkeypatch.setattr(server, "is_security_enabled", lambda: True)
    monkeypatch.setattr(server, "_require_history", lambda: SimpleNamespace(data_dir=tmp_path))
    route = "/history/secure/remote-policy"
    body = {"enabled": False, "daily_usd": 1, "monthly_usd": 20,
            "override_ceiling": False}
    local = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 1234))
    async with httpx.AsyncClient(transport=local, base_url="http://localhost") as client:
        assert (await client.get(route)).status_code == 401
        headers = {"Authorization": f"Bearer {token}"}
        denied = await client.post(route, json=body,
                                   headers={**headers, "Origin": "http://evil.test"})
        assert denied.status_code == 403
        assert denied.headers["cache-control"] == "no-store"
        saved = await client.post(route, json=body,
                                  headers={**headers, "Origin": "http://localhost"})
        assert saved.status_code == 200
        assert saved.json()["data"]["enabled"] is False
        assert saved.headers["cache-control"] == "no-store"
        assert (await client.get(route, headers=headers)).json()["data"]["source"] == "managed"
    remote = httpx.ASGITransport(app=server.app, client=("192.168.1.2", 1234))
    async with httpx.AsyncClient(transport=remote, base_url="http://localhost") as client:
        assert (await client.post(route, json=body,
                                  headers={"Authorization": f"Bearer {token}"})).status_code == 404
