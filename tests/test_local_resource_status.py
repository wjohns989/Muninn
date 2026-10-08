"""On-demand resource status must be local, authenticated, and explicit about unknowns."""

import httpx
import pytest

import server
from muninn.history import auto_routing


def test_resource_status_reports_residency_without_paths_or_raw_ollama_fields(monkeypatch):
    monkeypatch.setattr(auto_routing, "probe_gpu", lambda: auto_routing.GpuState(
        free_mib=9000, total_mib=16000, utilization_percent=2, sampled_at=123.0))
    monkeypatch.setattr(auto_routing, "inspect_ollama", lambda _url: auto_routing.OllamaState(
        ({"name": "local:7b", "size": 1234, "digest": "private-digest"},),
        (), 124.0))
    report = auto_routing.local_resource_status()
    assert report == {
        "gpu": {"state": "ready", "free_mib": 9000, "total_mib": 16000,
                "utilization_percent": 2, "sampled_at": 123.0},
        "ollama": {"state": "ready", "sampled_at": 124.0,
                   "installed": [{"name": "local:7b", "size_bytes": 1234}],
                   "resident_models": []},
    }
    assert "private-digest" not in str(report)


def test_resource_status_distinguishes_failed_probes_from_idle(monkeypatch):
    monkeypatch.setattr(auto_routing, "probe_gpu", lambda: None)
    monkeypatch.setattr(auto_routing, "inspect_ollama", lambda _url: None)
    assert auto_routing.local_resource_status() == {
        "gpu": {"state": "unavailable"}, "ollama": {"state": "unavailable"}}


def test_malformed_ollama_response_is_unknown_not_empty(monkeypatch):
    class Response:
        status_code = 200

        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            return None

        def json(self):
            return self.data

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, url):
            return Response({"models": "not-a-list"} if url.endswith("/tags") else {"models": []})

    monkeypatch.setattr(auto_routing.httpx, "Client", Client)
    assert auto_routing.inspect_ollama("http://127.0.0.1:11434") is None
    monkeypatch.setattr(auto_routing.httpx, "Client", lambda **_kwargs: (_ for _ in ()).throw(
        httpx.TimeoutException("timeout")))
    assert auto_routing.inspect_ollama("http://127.0.0.1:11434") is None


@pytest.mark.asyncio
async def test_resource_api_requires_main_local_token_and_never_caches(monkeypatch):
    token = "test-main-auth-token-aaaaaaaaaaaaaaaaaaaaaaaa"
    monkeypatch.setenv("MUNINN_AUTH_TOKEN", token)
    monkeypatch.setenv("MUNINN_NO_AUTH", "0")
    monkeypatch.setattr(server, "is_security_enabled", lambda: True)
    expected = {"gpu": {"state": "unavailable"}, "ollama": {"state": "unavailable"}}
    monkeypatch.setattr(auto_routing, "local_resource_status", lambda: expected)
    route = "/history/secure/resources"
    local = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 1234))
    async with httpx.AsyncClient(transport=local, base_url="http://localhost") as client:
        assert (await client.get(route)).status_code == 401
        response = await client.get(route, headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.json() == {"success": True, "data": expected}
    remote = httpx.ASGITransport(app=server.app, client=("192.168.1.2", 1234))
    async with httpx.AsyncClient(transport=remote, base_url="http://localhost") as client:
        assert (await client.get(route, headers={"Authorization": f"Bearer {token}"})).status_code == 404
