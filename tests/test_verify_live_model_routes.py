"""Bounded model-probe input retrieval; no live service or model calls."""

import json
from types import SimpleNamespace

import pytest

from scripts import verify_live_model_routes as probe


class _Response:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return {"data": self.data}


def test_private_case_polls_projection_and_reads_bounded_page(monkeypatch):
    calls = []
    states = [
        {"matches": [{"fetch_capability": "opaque-capability"}]},
        {"state": "pending"},
        {"state": "ready", "cursor": "opaque-cursor"},
        {"redacted_text": "Actual archived discussion of Muninn."},
    ]

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["timeout"] == 30.0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, **kwargs):
            calls.append((url, kwargs["json"]))
            return _Response(states.pop(0))

    monkeypatch.setattr(probe.httpx, "Client", Client)
    monkeypatch.setattr(probe, "_user_token", lambda: "test-token")
    monkeypatch.setattr(probe.time, "sleep", lambda _: None)
    text = probe._private_case("Muninn")
    assert "Actual archived discussion of Muninn." in text
    assert [url.rsplit("/", 1)[-1] for url, _ in calls] == ["search", "start", "poll", "page"]
    assert calls[-1][1] == {"cursor": "opaque-cursor"}


@pytest.mark.asyncio
async def test_input_failure_never_starts_inference(monkeypatch, capsys):
    monkeypatch.setattr(probe, "_private_case", lambda _: (_ for _ in ()).throw(RuntimeError("private detail")))
    monkeypatch.setattr("sys.argv", ["verify_live_model_routes.py", "--archive-query", "Muninn"])
    assert await probe.main() == 2
    result = json.loads(capsys.readouterr().out)
    assert result == {"state": "input_unavailable", "error_type": "RuntimeError",
                      "inference_sent": False}


@pytest.mark.asyncio
async def test_skipped_local_model_is_not_reported_as_pass(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["verify_live_model_routes.py", "--models", "test-model"])
    monkeypatch.setattr(probe, "_public_case", lambda: "checked-in code case")
    monkeypatch.setattr(probe, "probe_gpu", lambda: None)
    monkeypatch.setattr(probe, "probe_ollama", lambda _: ([], []))
    monkeypatch.setattr(probe, "choose_route", lambda *args, **kwargs:
                        SimpleNamespace(provider="deferred", model=None, reason="gpu_busy"))
    assert await probe.main() == 2
    assert json.loads(capsys.readouterr().out)["skipped"] == "gpu_busy"


@pytest.mark.asyncio
async def test_local_model_error_is_not_reported_as_pass(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["verify_live_model_routes.py", "--models", "test-model"])
    monkeypatch.setattr(probe, "_public_case", lambda: "checked-in code case")
    monkeypatch.setattr(probe, "probe_gpu", lambda: None)
    monkeypatch.setattr(probe, "probe_ollama", lambda _: ([], []))
    monkeypatch.setattr(probe, "choose_route", lambda *args, **kwargs:
                        SimpleNamespace(provider="ollama", model="test-model", reason="idle"))

    async def fail(*args, **kwargs):
        raise RuntimeError("do not disclose response")

    monkeypatch.setattr(probe, "_one", fail)
    assert await probe.main() == 1
    assert json.loads(capsys.readouterr().out) == {"model": "test-model", "error_type": "RuntimeError"}
