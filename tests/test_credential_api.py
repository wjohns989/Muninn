"""Synthetic ASGI tests for explicit local credential API access."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

import server
from muninn.history.credential_api import RevealLimiter
from muninn.history.credential_store import CredentialStore, source_fingerprint
from muninn.mcp.definitions import TOOLSETS

_TOKEN = "synthetic-local-credential-api-token-123456"
_PASSPHRASE = "synthetic local passphrase with enough entropy"
_VALUE = "synthetic-api-secret-456789"


def _client(peer: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app, client=(peer, 1234)),
                             base_url="http://localhost")


@pytest.mark.asyncio
async def test_api_requires_dedicated_token_and_loopback(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(server, "memory", SimpleNamespace(config=SimpleNamespace(data_dir=str(tmp_path))))
    async with _client("127.0.0.1") as client:
        disabled = await client.get("/credentials/search", params={"query": "example"})
        assert disabled.status_code == 404 and disabled.headers["cache-control"] == "no-store"
    monkeypatch.setenv("MUNINN_CREDENTIAL_API_TOKEN", _TOKEN)
    async with _client("127.0.0.1") as client:
        assert (await client.get("/credentials/search", params={"query": "example"})).status_code == 401
        assert (await client.get("/credentials/search")).status_code == 401
        assert (await client.get("/credentials/search", params={"limit": "abc"})).status_code == 401
        assert (await client.get("/credentials/search", params={"query": "example"},
                                 headers={"Authorization": "Bearer wrong"})).status_code == 401
        assert (await client.get("/credentials/search", params={"limit": "abc"},
                                 headers={"Authorization": f"Bearer {_TOKEN}"})).status_code == 400
    for remote in ("192.168.1.2", "203.0.113.5"):
        async with _client(remote) as client:
            assert (await client.get("/credentials/search", params={"query": "example"},
                                     headers={"Authorization": f"Bearer {_TOKEN}"})).status_code == 404


@pytest.mark.asyncio
async def test_metadata_and_reveal_are_separate_audited_and_not_logged(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setenv("MUNINN_CREDENTIAL_API_TOKEN", _TOKEN)
    monkeypatch.setattr(server, "memory", SimpleNamespace(config=SimpleNamespace(data_dir=str(tmp_path))))
    monkeypatch.setattr(server, "_credential_reveal_limiter", RevealLimiter())
    store = CredentialStore.create(tmp_path / "credential_vault", _PASSPHRASE)
    record_id = store.add(passphrase=_PASSPHRASE, value=_VALUE, service="example", project="test",
                          source_hash=source_fingerprint("source"))
    store.scan_source(passphrase=_PASSPHRASE, source_hash=source_fingerprint("project-config"),
                      project="test", origin="project",
                      findings=[("SERVICE_API_KEY", _VALUE, "config.yaml")])
    headers = {"Authorization": f"Bearer {_TOKEN}"}
    with caplog.at_level(logging.INFO):
        for peer in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
            async with _client(peer) as client:
                searched = await client.get("/credentials/search", params={"query": "example"}, headers=headers)
                assert searched.status_code == 200, (peer, searched.text)
                assert searched.json()["data"][0]["id"] == record_id
                assert _VALUE not in searched.text
                assert searched.headers["cache-control"] == "no-store"
                candidate = await client.get("/credentials/search", params={"query": "SERVICE_API_KEY"},
                                             headers=headers)
                assert candidate.status_code == 200
                assert candidate.json()["data"][0]["candidate_status"] == "unverified"
                assert candidate.json()["data"][0]["source_hint"] == "config.yaml"
                assert _VALUE not in candidate.text
        async with _client("127.0.0.1") as client:
            bad = await client.post(f"/credentials/reveal/{record_id}", headers=headers,
                                    json={"passphrase": "wrong passphrase"})
            assert bad.status_code == 404 and bad.headers["cache-control"] == "no-store"
            good = await client.post(f"/credentials/reveal/{record_id}", headers=headers,
                                     json={"passphrase": _PASSPHRASE})
            assert good.status_code == 200 and good.json()["value"] == _VALUE
            assert good.headers["cache-control"] == "no-store"
            assert (await client.post(f"/credentials/reveal/{record_id}", headers=headers,
                                      content=b"x" * 4097)).status_code == 415
            oversized = await client.post(f"/credentials/reveal/{record_id}",
                                          headers={**headers, "Content-Type": "application/json"},
                                          content=b"x" * 4097)
            assert oversized.status_code == 413
    assert _PASSPHRASE not in caplog.text and _VALUE not in caplog.text
    with store._connect(readonly=True) as db:
        assert db.execute("SELECT COUNT(*) FROM reveal_audit").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_opt_in_agent_metadata_search_never_reveals_values(tmp_path, monkeypatch) -> None:
    main_token = "test-main-auth-token-aaaaaaaaaaaaaaaaaaaaaaaa"
    monkeypatch.setenv("MUNINN_AUTH_TOKEN", main_token)
    monkeypatch.setenv("MUNINN_API_KEY", "different-api-key-bbbbbbbbbbbbbbbbbbbbbbbb")
    monkeypatch.setenv("MUNINN_NO_AUTH", "0")
    monkeypatch.setattr(server, "memory", SimpleNamespace(config=SimpleNamespace(data_dir=str(tmp_path))))
    monkeypatch.setattr(server, "is_security_enabled", lambda: True)
    monkeypatch.setattr(server, "_credential_agent_search_times", deque())
    store = CredentialStore.create(tmp_path / "credential_vault", _PASSPHRASE)
    store.add(passphrase=_PASSPHRASE, value=_VALUE, service="openrouter", project="example",
              source_hash=source_fingerprint("private-source"), source_hint="config/.env.local")
    store.scan_source(
        passphrase=_PASSPHRASE, source_hash=source_fingerprint("historical-chat"),
        project="codex", origin="transcript",
        findings=[("SERVICE_API_KEY", _VALUE, "")],
    )
    headers = {"Authorization": f"Bearer {main_token}"}
    async with _client("127.0.0.1") as client:
        disabled = await client.post("/credentials/agent-search", json={"query": "openrouter"}, headers=headers)
        assert disabled.status_code == 404
        monkeypatch.setenv("MUNINN_CREDENTIAL_AGENT_SEARCH", "1")
        assert (await client.post("/credentials/agent-search", json={"query": "openrouter"})).status_code == 401
        assert (await client.post("/credentials/agent-search", json={"query": "openrouter"},
                                  headers={"Authorization": "Bearer different-api-key-bbbbbbbbbbbbbbbbbbbbbbbb"}
                                  )).status_code == 401
        found = await client.post("/credentials/agent-search", json={"query": ".env.local"}, headers=headers)
        assert found.status_code == 200
        assert found.json()["success"] is True
        assert found.json()["data"][0]["source_hint"] == "config/.env.local"
        assert set(found.json()["data"][0]) == {"id", "service", "project", "source_hash", "source_hint"}
        assert _VALUE not in found.text
        assert found.headers["cache-control"] == "no-store"
        historical = await client.post("/credentials/agent-search", json={"query": "SERVICE_API_KEY"}, headers=headers)
        assert historical.status_code == 200
        assert historical.json()["data"][0]["origin"] == "transcript"
        assert historical.json()["data"][0]["candidate_status"] == "unverified"
        assert _VALUE not in historical.text
    async with _client("192.168.1.2") as remote:
        assert (await remote.post("/credentials/agent-search", json={"query": "openrouter"},
                                  headers=headers)).status_code == 404
    assert "search_credential_metadata" in TOOLSETS["core"]
    assert not any("reveal" in tool for tool in TOOLSETS["core"])


def test_limiter_is_concurrent_and_expires() -> None:
    current = [1000.0]
    limiter = RevealLimiter(clock=lambda: current[0])
    key = ("principal", "127.0.0.1", "vault", "record")
    for _ in range(5):
        assert limiter.begin(key)
        limiter.finish(key, success=False)
    assert not limiter.begin(key)
    other = ("principal", "127.0.0.1", "vault", "other-record")
    assert limiter.begin(other)
    limiter.finish(other, success=True)
    current[0] += 301
    assert limiter.begin(key)
    limiter.finish(key, success=True)
    for index in range(50):
        another = ("principal", "127.0.0.1", "vault", str(index))
        assert limiter.begin(another)
        limiter.finish(another, success=False)
    fresh = ("principal", "127.0.0.1", "vault", "new")
    assert not limiter.begin(fresh)
    current[0] += 301
    assert limiter.begin(fresh)
    limiter.finish(fresh, success=True)
    reservations: list[bool] = []
    slots = [threading.Thread(target=lambda: reservations.append(limiter.begin(key))) for _ in range(8)]
    for thread in slots:
        thread.start()
    for thread in slots:
        thread.join()
    assert reservations.count(True) == 1
    limiter.finish(key, success=True)
    assert limiter.begin(key)
    limiter.finish(key, success=False)


@pytest.mark.asyncio
async def test_cancelled_client_keeps_slot_until_worker_finishes(tmp_path, monkeypatch) -> None:
    started, release = threading.Event(), threading.Event()
    limiter = RevealLimiter()
    monkeypatch.setattr(server, "_credential_reveal_limiter", limiter)
    monkeypatch.setattr(server, "authenticate_local", lambda _request: ("principal", "127.0.0.1"))
    monkeypatch.setattr(server, "read_passphrase", AsyncMock(return_value=_PASSPHRASE))

    class SlowStore:
        root = tmp_path

        def reveal(self, _record_id: str, *, passphrase: str) -> str:
            assert passphrase == _PASSPHRASE
            started.set()
            assert release.wait(5)
            return _VALUE

    monkeypatch.setattr(server, "_credential_store_for_api", SlowStore)
    key = ("principal", "127.0.0.1", str(tmp_path), "a" * 32)
    request_task = asyncio.create_task(server.credential_reveal_endpoint("a" * 32, object()))
    assert await asyncio.to_thread(started.wait, 5)
    request_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request_task
    assert not limiter.begin(key)
    release.set()
    for _ in range(100):
        await asyncio.sleep(0.01)
        if limiter.begin(key):
            limiter.finish(key, success=True)
            break
    else:
        pytest.fail("reveal slot was not released after canceled request's worker finished")
