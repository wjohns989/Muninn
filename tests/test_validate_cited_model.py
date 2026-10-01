"""The real-data probe reports only counts and typed status."""

import json
from types import SimpleNamespace

from scripts import validate_cited_model as probe


def test_probe_selects_citation_capable_source_without_printing_model_text(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("sys.argv", ["validate_cited_model.py", "--root", str(tmp_path),
                                  "--query", "Muninn", "--model", "model-x"])
    monkeypatch.setattr(probe, "SecureHistoryArchive", lambda root: object())

    class Index:
        def __init__(self, archive):
            pass

        def search(self, query, **kwargs):
            return {"matches": [
                {"fetch_capability": "opaque-large", "size_bucket_kib": 100},
                {"fetch_capability": "opaque-small", "size_bucket_kib": 1},
            ]}

    class Source:
        def __init__(self, archive):
            pass

        def prepare(self, capability):
            return None if capability == "opaque-small" else {"opaque": "descriptor"}

    async def analyze(history, source, descriptor, *, allow_remote, prefer_remote,
                      expected_remote_generation):
        assert not allow_remote
        assert not prefer_remote
        assert expected_remote_generation is None
        return {"status": "ok", "provider": "ollama", "model": "model-x",
                "extraction": {"proposals": [{"text": "private model text"}]}}

    monkeypatch.setattr(probe, "SecureHistoryBlindIndex", Index)
    monkeypatch.setattr(probe, "CitedAnalysisSource", Source)
    monkeypatch.setattr(probe, "analyze_cited_window", analyze)
    assert probe.main() == 0
    output = capsys.readouterr().out
    assert "private model text" not in output
    report = json.loads(output)
    assert report["candidates_checked"] == 2
    assert report["proposals"] == 1
    assert report["provider"] == "ollama"
    assert report["requested_model_used"] is True
    assert report["status"] == "ok"


def test_probe_bounds_only_local_request(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("sys.argv", ["validate_cited_model.py", "--root", str(tmp_path),
                                  "--query", "Muninn", "--model", "model-x",
                                  "--max-output-tokens", "512"])
    monkeypatch.setattr(probe, "SecureHistoryArchive", lambda root: object())
    monkeypatch.setattr(probe, "SecureHistoryBlindIndex", lambda archive: type(
        "Index", (), {"search": lambda self, query, **kwargs: {"matches": [
            {"fetch_capability": "opaque", "size_bucket_kib": 1}]}})())
    monkeypatch.setattr(probe, "CitedAnalysisSource", lambda archive: type(
        "Source", (), {"prepare": lambda self, capability: {"opaque": "descriptor"}})())

    async def analyze(history, source, descriptor, *, allow_remote, prefer_remote,
                      expected_remote_generation):
        local = probe.Provider("ollama", "http://127.0.0.1:11434/v1", ["model-x"])
        remote = probe.Provider("openrouter", "https://openrouter.ai/api/v1", ["model-y"])
        assert local.request_body([])["options"]["num_predict"] == 512
        assert "options" not in remote.request_body([])
        return {"status": "ok", "provider": "ollama", "model": "model-x",
                "extraction": {"proposals": []}}

    monkeypatch.setattr(probe, "analyze_cited_window", analyze)
    assert probe.main() == 0
    assert json.loads(capsys.readouterr().out)["max_output_tokens"] == 512


def test_remote_probe_requires_screened_source_and_reports_only_metadata(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("sys.argv", ["validate_cited_model.py", "--root", str(tmp_path),
                                  "--query", "Muninn", "--provider", "openrouter",
                                  "--acknowledge-private-zdr"])
    monkeypatch.setattr(probe, "SecureHistoryArchive", lambda root: object())
    monkeypatch.setattr(probe, "SecureHistoryBlindIndex", lambda archive: type(
        "Index", (), {"search": lambda self, query, **kwargs: {"matches": [
            {"fetch_capability": "unsafe", "size_bucket_kib": 1},
            {"fetch_capability": "safe", "size_bucket_kib": 2}]}})())

    class Source:
        def __init__(self, archive):
            pass

        def prepare(self, capability):
            return {"opaque": capability}

        def remote_input(self, descriptor):
            return None if descriptor["opaque"] == "unsafe" else {"screened": True}

    monkeypatch.setattr(probe, "CitedAnalysisSource", Source)
    monkeypatch.setattr(probe, "read_policy", lambda root, fallback: SimpleNamespace(
        enabled=True, generation=2))
    monkeypatch.setattr(probe, "remote_accounting_status", lambda root: {
        "state": "ready", "unresolved": 0, "daily_cost_usd": 0.0008})

    async def analyze(history, source, descriptor, *, allow_remote, prefer_remote,
                      expected_remote_generation):
        assert descriptor == {"opaque": "safe"}
        assert allow_remote and prefer_remote and expected_remote_generation == 2
        remote = probe.Provider("openrouter", "https://openrouter.ai/api/v1", ["model-y"])
        assert remote.request_body([])["max_completion_tokens"] == 2048
        return {"status": "ok", "provider": "openrouter", "model": "model-y",
                "extraction": {"proposals": [{"text": "private model text"}]}}

    monkeypatch.setattr(probe, "analyze_cited_window", analyze)
    assert probe.main() == 0
    output = capsys.readouterr().out
    assert "private model text" not in output
    report = json.loads(output)
    assert report["candidates_checked"] == 2
    assert report["accounting_state"] == "ready"
    assert report["provider"] == "openrouter"
    assert report["model"] == "model-y"


def test_remote_probe_reports_unresolved_accounting_after_exception(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("sys.argv", ["validate_cited_model.py", "--root", str(tmp_path),
                                  "--query", "Muninn", "--provider", "openrouter",
                                  "--acknowledge-private-zdr"])
    monkeypatch.setattr(probe, "SecureHistoryArchive", lambda root: object())
    monkeypatch.setattr(probe, "SecureHistoryBlindIndex", lambda archive: type(
        "Index", (), {"search": lambda self, query, **kwargs: {"matches": [
            {"fetch_capability": "safe", "size_bucket_kib": 1}]}})())
    monkeypatch.setattr(probe, "CitedAnalysisSource", lambda archive: type(
        "Source", (), {"prepare": lambda self, capability: {"opaque": "safe"},
                       "remote_input": lambda self, descriptor: {"screened": True}})())
    monkeypatch.setattr(probe, "read_policy", lambda root, fallback: SimpleNamespace(
        enabled=True, generation=2))
    monkeypatch.setattr(probe, "remote_accounting_status", lambda root: {
        "state": "blocked", "unresolved": 1, "daily_cost_usd": 0})

    async def analyze(history, source, descriptor, **kwargs):
        raise TimeoutError("private text must not appear")

    monkeypatch.setattr(probe, "analyze_cited_window", analyze)
    assert probe.main() == 1
    output = capsys.readouterr().out
    assert "private text" not in output
    report = json.loads(output)
    assert report["error_type"] == "TimeoutError"
    assert report["accounting_state"] == "blocked"
    assert report["unresolved_admissions"] == 1
