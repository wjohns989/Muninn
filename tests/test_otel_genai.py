import importlib

import pytest

import muninn.observability.otel_genai as otel_mod


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for key in [
        "MUNINN_OTEL_CAPTURE_CONTENT",
        "MUNINN_OTEL_CAPTURE_CONTENT_MAX_CHARS",
    ]:
        monkeypatch.delenv(key, raising=False)


def _reload_module():
    return importlib.reload(otel_mod)


def test_maybe_content_disabled_by_default():
    mod = _reload_module()
    tracer = mod.OTelGenAITracer(enabled=False)
    assert tracer.maybe_content("secret text") is None


def test_maybe_content_respects_max_chars(monkeypatch):
    monkeypatch.setenv("MUNINN_OTEL_CAPTURE_CONTENT", "1")
    monkeypatch.setenv("MUNINN_OTEL_CAPTURE_CONTENT_MAX_CHARS", "4")
    mod = _reload_module()
    tracer = mod.OTelGenAITracer(enabled=False)
    assert tracer.maybe_content("abcdefgh") == "abcd"


def test_maybe_content_invalid_max_chars_falls_back(monkeypatch):
    monkeypatch.setenv("MUNINN_OTEL_CAPTURE_CONTENT", "1")
    monkeypatch.setenv("MUNINN_OTEL_CAPTURE_CONTENT_MAX_CHARS", "not-a-number")
    mod = _reload_module()
    tracer = mod.OTelGenAITracer(enabled=False)
    assert tracer.capture_content_max_chars == 1000
    assert tracer.maybe_content("x" * 1200) == "x" * 1000


def test_maybe_content_zero_max_chars_disables_content(monkeypatch):
    monkeypatch.setenv("MUNINN_OTEL_CAPTURE_CONTENT", "1")
    monkeypatch.setenv("MUNINN_OTEL_CAPTURE_CONTENT_MAX_CHARS", "0")
    mod = _reload_module()
    tracer = mod.OTelGenAITracer(enabled=False)
    assert tracer.maybe_content("any") is None
