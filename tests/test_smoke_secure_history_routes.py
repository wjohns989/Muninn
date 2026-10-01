"""The live smoke command reports only bounded metadata, never transcript text."""

from scripts import smoke_secure_history_routes as smoke


def test_local_smoke_uses_real_capability_without_printing_content(monkeypatch):
    monkeypatch.setattr(smoke, "_token", lambda: "fixture-token")
    calls = []
    options = {}

    class Response:
        headers = {"cache-control": "no-store"}

        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return {"success": True, "data": self.data}

    class Client:
        def __init__(self, **kwargs):
            options.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def post(self, url, json):
            calls.append((url, json))
            if url.endswith("/search"):
                return Response({"matches": [{"fetch_capability": "opaque"}]})
            if url.endswith("/fetch"):
                return Response({"redacted_text": "private transcript " * 20})
            return Response({"status": "ok", "provider": "ollama", "model": "local",
                             "analysis": {"summary": "secret-value", "decisions": ["one"],
                                          "open_items": []}})

    monkeypatch.setattr(smoke.httpx, "Client", Client)
    report = smoke.check("Muninn", remote=False)
    assert report["route_matched"] is True
    assert report["summary_chars"] == len("secret-value")
    assert "secret-value" not in str(report)
    assert "private transcript" not in str(report)
    assert calls[-1][1] == {"capability": "opaque", "allow_remote": False,
                            "prefer_remote": False}
    assert options["trust_env"] is False


def test_remote_smoke_requires_eligible_redacted_hit(monkeypatch):
    monkeypatch.setattr(smoke, "_token", lambda: "fixture-token")
    monkeypatch.setattr(smoke, "_remote_eligible", lambda span, allow_remote: False)
    calls = []

    class Response:
        headers = {"cache-control": "no-store"}

        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return {"success": True, "data": self.data}

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def post(self, url, json):
            calls.append(url)
            if url.endswith("/search"):
                return Response({"matches": [{"fetch_capability": "opaque"}]})
            return Response({"redacted_text": "private transcript " * 20})

    monkeypatch.setattr(smoke.httpx, "Client", Client)
    report = smoke.check("Muninn", remote=True)
    assert report["status"] == "no_eligible_real_hit"
    assert not any(url.endswith("/analyze") for url in calls)
