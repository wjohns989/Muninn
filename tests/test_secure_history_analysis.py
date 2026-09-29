"""Strict on-demand analysis never accepts caller-supplied transcript text."""

import json
from contextlib import asynccontextmanager

import httpx
import pytest

from muninn.history import secure_analysis as analysis


def test_installed_model_pool_is_dynamic(monkeypatch):
    monkeypatch.setattr(analysis, "_local_setting", lambda _name: "")
    installed = [{"name": "another-chat-model:latest"}, {"name": "qwen2.5:7b"}]
    assert analysis._candidate_names(installed) == ["qwen2.5:7b", "another-chat-model:latest"]


def test_remote_requires_local_setting_and_request_opt_in(monkeypatch):
    span = "This real transcript excerpt contains useful project context. " * 4
    monkeypatch.setattr(analysis, "_local_setting", lambda _name: "1")
    assert not analysis._remote_eligible(span, allow_remote=False)
    assert analysis._remote_eligible(span, allow_remote=True)
    monkeypatch.setattr(analysis, "_local_setting", lambda _name: "0")
    assert not analysis._remote_eligible(span, allow_remote=True)


def test_model_output_is_bounded_and_secret_scrubbed():
    raw = json.dumps({
        "summary": "The key API_KEY=abcdefghijklmnopqrstuvwxyz1234567890 was used.",
        "decisions": ["Keep local transcript capture."],
        "open_items": [], "uncertainty": "No execution evidence was provided.",
    })
    cleaned = analysis._clean_result(raw)
    assert "abcdefghijklmnopqrstuvwxyz1234567890" not in str(cleaned)
    assert "Keep local transcript capture" in cleaned["decisions"][0]


def test_model_cannot_echo_unlabeled_secret_from_its_input():
    marker = "CANARY-SECRET-91919"
    content = json.dumps({
        "summary": f"Use {marker} for the service.",
        "decisions": [], "open_items": [], "uncertainty": "Not verified.",
    })
    assert marker not in str(analysis._clean_result(content, source_span=f"API_KEY={marker}"))


def test_model_echo_scrub_covers_every_field_and_multiple_values():
    first = "CANARY-SECRET-91919"
    second = "CANARY-SECRET-91919-EXTRA"
    content = json.dumps({
        "summary": second,
        "decisions": [first], "open_items": [second], "uncertainty": first,
    })
    cleaned = analysis._clean_result(content, source_span=f"API_KEY={first}\nTOKEN={second}")
    assert first not in str(cleaned) and second not in str(cleaned)
    assert "[REDACTED_SOURCE_VALUE]" in str(cleaned)


def test_model_echo_scrub_errors_do_not_repeat_source():
    marker = "CANARY-SECRET-91919"
    with pytest.raises(ValueError) as failure:
        analysis._clean_result("not json", source_span=f"API_KEY={marker}")
    assert marker not in str(failure.value)


@pytest.mark.parametrize("label", ["TOKEN", "private key"])
def test_model_echo_scrub_covers_standalone_credential_labels(label):
    marker = "CANARY-SECRET-91919"
    content = json.dumps({
        "summary": marker, "decisions": [], "open_items": [], "uncertainty": "Unknown.",
    })
    assert marker not in str(analysis._clean_result(content, source_span=f"{label}={marker}"))


@pytest.mark.asyncio
@pytest.mark.parametrize("short", [False, True])
async def test_local_route_uses_capability_and_unloads_model(monkeypatch, short):
    seen = {}

    class History:
        def _secure_model_window(self, capability):
            seen["capability"] = capability
            return ("Use SQLite for the local cache." if short else
                    "A real project decision was made to keep SQLite. " * 5 + "CANARY-SECRET-91919")

        def secure_fetch_span(self, *_args, **_kwargs):
            pytest.fail("ordinary redacted fetch must not supply model input")

    @asynccontextmanager
    async def slot():
        yield

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": json.dumps({
                "summary": "SQLite remains the project cache.",
                "decisions": ["Keep SQLite."], "open_items": [],
                "uncertainty": "No deployment proof is shown.",
            })}}

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, *, json, **_kwargs):
            seen["url"] = url
            seen["body"] = json
            return Response()

    monkeypatch.setattr(analysis, "_select_local", lambda _base: ("qwen2.5:7b", "idle"))
    monkeypatch.setattr("muninn.extraction.ollama_slot.async_ollama_slot", slot)
    monkeypatch.setattr(httpx, "AsyncClient", Client)
    monkeypatch.setenv("MUNINN_OLLAMA_KEEP_ALIVE", "30m")
    result = await analysis.analyze_secure_hit(History(), "opaque-capability")
    assert seen["capability"] == "opaque-capability"
    assert seen["body"]["keep_alive"] == 0
    assert "untrusted_transcript" in seen["body"]["messages"][1]["content"]
    if not short:
        assert "CANARY-SECRET-91919" in seen["body"]["messages"][1]["content"]
    assert "CANARY-SECRET-91919" not in str(result)
    assert result["provider"] == "ollama" and result["analysis"]["summary"]


@pytest.mark.asyncio
async def test_no_remote_fallback_after_local_failure(monkeypatch):
    class History:
        def _secure_model_window(self, *_args):
            return "A project decision was made. " * 10

    @asynccontextmanager
    async def slot():
        yield

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            raise httpx.ConnectError("local model unavailable")

    monkeypatch.setattr(analysis, "_select_local", lambda _base: ("qwen2.5:7b", "idle"))
    monkeypatch.setattr("muninn.extraction.ollama_slot.async_ollama_slot", slot)
    monkeypatch.setattr(httpx, "AsyncClient", Client)
    monkeypatch.setattr(analysis, "guarded_openrouter_available",
                        lambda: pytest.fail("remote fallback must not be attempted"))
    with pytest.raises(httpx.ConnectError):
        await analysis.analyze_secure_hit(History(), "opaque-capability", allow_remote=True)


@pytest.mark.asyncio
async def test_explicit_remote_route_requires_both_opt_ins_and_zdr(monkeypatch):
    seen = {}

    class History:
        def _secure_model_window(self, capability):
            return "The project decided to keep SQLite for local caching. " * 6 + "CANARY-SECRET-91919"

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"model": "test-zdr-model", "choices": [{"message": {"content": json.dumps({
                "summary": "SQLite was selected for caching.",
                "decisions": ["Keep SQLite."], "open_items": [],
                "uncertainty": "No deployment proof.",
            })}}]}

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, *, json, headers):
            seen["post_count"] = seen.get("post_count", 0) + 1
            seen["url"] = url
            seen["body"] = json
            assert headers["Authorization"] == "Bearer fixture-key"
            return Response()

    monkeypatch.setattr(analysis, "_local_setting", lambda _name: "1")
    monkeypatch.setattr(analysis, "guarded_openrouter_available", lambda: True)
    monkeypatch.setattr(analysis, "_select_local", lambda _base: pytest.fail("local was forced off"))
    monkeypatch.setattr(analysis.Provider, "from_env",
                        lambda *_args: analysis.Provider("openrouter", "https://openrouter.ai/api/v1",
                                                         ["test-zdr-model"], "fixture-key"))
    monkeypatch.setattr(httpx, "AsyncClient", Client)
    with pytest.raises(ValueError):
        await analysis.analyze_secure_hit(History(), "cap", prefer_remote=True)
    result = await analysis.analyze_secure_hit(History(), "cap", allow_remote=True,
                                               prefer_remote=True)
    assert result["provider"] == "openrouter"
    assert seen["body"]["provider"] == {
        "zdr": True, "data_collection": "deny", "require_parameters": True,
    }
    assert "CANARY-SECRET-91919" in seen["body"]["messages"][1]["content"]
    assert "CANARY-SECRET-91919" not in str(result)
    async def veto_remote():
        return False

    with pytest.raises(RuntimeError, match="lease unavailable"):
        await analysis.analyze_secure_hit(History(), "cap", allow_remote=True,
                                          prefer_remote=True, before_remote=veto_remote)
    assert seen["post_count"] == 1
