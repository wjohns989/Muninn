"""A queued search receipt and poll never expose query or transcript content."""

from __future__ import annotations

import httpx
import pytest

import server


@pytest.mark.asyncio
async def test_secure_search_jobs_require_main_local_auth_and_no_store(monkeypatch) -> None:
    token = "test-main-auth-token-aaaaaaaaaaaaaaaaaaaaaaaa"
    monkeypatch.setenv("MUNINN_AUTH_TOKEN", token)
    monkeypatch.setenv("MUNINN_NO_AUTH", "0")
    monkeypatch.setattr(server, "is_security_enabled", lambda: True)

    class FakeHistory:
        def queue_secure_search(self, query, *, limit):
            assert (query, limit) == ("private query marker", 1)
            return "a" * 32

        def secure_search_job_status(self, job_id):
            if job_id != "a" * 32:
                return None
            return {"job_id": job_id, "state": "pending", "result": None}

        def cancel_secure_search_job(self, job_id):
            return job_id == "a" * 32

    monkeypatch.setattr(server, "_require_history", lambda: FakeHistory())
    base = "/history/secure/search/jobs"
    local = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 1234))
    async with httpx.AsyncClient(transport=local, base_url="http://localhost") as client:
        assert (await client.post(base, json={"query": "private query marker", "limit": 1})).status_code == 401
        headers = {"Authorization": f"Bearer {token}"}
        queued = await client.post(base, json={"query": "private query marker", "limit": 1},
                                   headers=headers)
        assert queued.status_code == 202
        assert queued.headers["cache-control"] == "no-store"
        assert "private query marker" not in queued.text
        assert queued.json()["data"]["job_id"] == "a" * 32
        polled = await client.get(base + "/" + "a" * 32, headers=headers)
        assert polled.status_code == 200
        assert polled.headers["cache-control"] == "no-store"
        assert polled.json()["data"]["state"] == "pending"
        assert (await client.get(base + "/" + "b" * 32,
                                 headers=headers)).status_code == 404
        cancelled = await client.delete(base + "/" + "a" * 32, headers=headers)
        assert cancelled.status_code == 200
        assert cancelled.headers["cache-control"] == "no-store"
    remote = httpx.ASGITransport(app=server.app, client=("192.168.1.2", 1234))
    async with httpx.AsyncClient(transport=remote, base_url="http://localhost") as client:
        assert (await client.post(base, json={"query": "private query marker"},
                                  headers={"Authorization": f"Bearer {token}"})).status_code == 404
