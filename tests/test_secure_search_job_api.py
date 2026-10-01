"""A queued search receipt and poll never expose query or transcript content."""

from __future__ import annotations

import httpx
import pytest
import sqlite3

import server
from muninn.history.capture_journal import CaptureJournal
from muninn.history.secure_archive import SecureHistoryArchive


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

        def secure_analysis_job_status(self, job_id):
            if job_id != "c" * 32:
                return None
            return {"job_id": job_id, "state": "succeeded", "provisional": True,
                    "result": {"status": "ok", "provider": "ollama", "model": "test-model",
                               "analysis": {"summary": "Safe context", "decisions": [],
                                            "open_items": [], "uncertainty": "Unverified"}}}

        def cancel_secure_analysis_job(self, job_id):
            return job_id == "c" * 32

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
        analysis_base = "/history/secure/analysis/jobs"
        assert (await client.get(analysis_base + "/" + "c" * 32)).status_code == 401
        analyzed = await client.get(analysis_base + "/" + "c" * 32, headers=headers)
        assert analyzed.status_code == 200 and analyzed.headers["cache-control"] == "no-store"
        assert analyzed.json()["data"]["provisional"] is True
        assert (await client.get(analysis_base + "/" + "b" * 32,
                                 headers=headers)).status_code == 404
        assert (await client.delete(analysis_base + "/" + "c" * 32,
                                    headers=headers)).status_code == 200
    remote = httpx.ASGITransport(app=server.app, client=("192.168.1.2", 1234))
    async with httpx.AsyncClient(transport=remote, base_url="http://localhost") as client:
        assert (await client.post(base, json={"query": "private query marker"},
                                  headers={"Authorization": f"Bearer {token}"})).status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["search", "analysis"])
async def test_poll_busy_is_retryable_without_changing_durable_job(tmp_path, monkeypatch, kind):
    token = "test-main-auth-token-aaaaaaaaaaaaaaaaaaaaaaaa"
    monkeypatch.setenv("MUNINN_AUTH_TOKEN", token)
    monkeypatch.setenv("MUNINN_NO_AUTH", "0")
    monkeypatch.setattr(server, "is_security_enabled", lambda: True)
    archive = SecureHistoryArchive.create(tmp_path / "archive", "test-only recovery passphrase")
    journal = CaptureJournal(archive)
    job_id = journal.enqueue_search("private query marker")

    class History:
        # Both endpoints must handle the same real SQLite read contention;
        # analysis payload semantics are covered by the normal endpoint test.
        secure_search_job_status = staticmethod(journal.get_search_job)
        secure_analysis_job_status = staticmethod(journal.get_search_job)

    monkeypatch.setattr(server, "_require_history", History)
    url = f"/history/secure/{kind}/jobs/{job_id}"
    local = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 1234))
    async with httpx.AsyncClient(transport=local, base_url="http://localhost") as client:
        headers = {"Authorization": f"Bearer {token}"}
        with sqlite3.connect(journal.path) as writer:
            writer.execute("BEGIN EXCLUSIVE")
            assert (await client.get(url)).status_code == 401
            response = await client.get(url, headers=headers)
            assert response.status_code == 503
            assert response.headers["cache-control"] == "no-store"
            assert response.headers["retry-after"] == "1"
            assert "private query marker" not in response.text
            assert str(journal.path) not in response.text
            writer.rollback()
        response = await client.get(url, headers=headers)
        assert response.status_code == 200
        assert response.json()["data"] == {
            "job_id": job_id, "state": "pending", "result": None}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["search", "analysis"])
async def test_nonbusy_sql_failure_is_not_advertised_as_retryable(monkeypatch, kind):
    token = "test-main-auth-token-aaaaaaaaaaaaaaaaaaaaaaaa"
    monkeypatch.setenv("MUNINN_AUTH_TOKEN", token)
    monkeypatch.setenv("MUNINN_NO_AUTH", "0")
    monkeypatch.setattr(server, "is_security_enabled", lambda: True)

    def fail(job_id):
        error = sqlite3.OperationalError("private database error")
        error.sqlite_errorcode = sqlite3.SQLITE_ERROR
        raise error

    class History:
        secure_search_job_status = staticmethod(fail)
        secure_analysis_job_status = staticmethod(fail)

    monkeypatch.setattr(server, "_require_history", History)
    local = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 1234))
    async with httpx.AsyncClient(transport=local, base_url="http://localhost") as client:
        response = await client.get(f"/history/secure/{kind}/jobs/" + "b" * 32,
                                    headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 503
        assert "retry-after" not in response.headers
        assert "private database error" not in response.text
