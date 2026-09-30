"""Resource policy tests never load models or make network requests."""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture(autouse=True)
def _legacy_history_test_mode(monkeypatch):
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "legacy")

from muninn.history import auto_routing
from muninn.history.auto_routing import GpuState, choose_route, model_hints_for_thread


def test_ollama_probe_rejects_nonloopback_without_network(monkeypatch):
    monkeypatch.setattr(auto_routing.httpx, "Client", lambda **_kwargs: pytest.fail("network attempted"))
    assert auto_routing.probe_ollama("http://remote.example:11434") == ([], ())
    assert auto_routing.probe_ollama("http://127.0.0.1:11434@remote.example") == ([], ())


def test_ollama_probe_ignores_proxy_environment_and_redirects(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://remote.example:8080")
    called = []

    class Response:
        def __init__(self, data):
            self.data = data
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return self.data

    class Client:
        def __init__(self, **kwargs):
            called.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, url):
            assert url.startswith("http://127.0.0.1:11434/api/")
            return Response({"models": [{"name": "qwen2.5:7b", "size": 1000}]} if url.endswith("/tags")
                            else {"models": []})

    monkeypatch.setattr(auto_routing.httpx, "Client", Client)
    assert auto_routing.probe_ollama("http://127.0.0.1:11434") == (
        [{"name": "qwen2.5:7b", "size": 1000}], ())
    assert auto_routing.probe_ollama("http://localhost:11434") == (
        [{"name": "qwen2.5:7b", "size": 1000}], ())
    assert called == [{"timeout": 3.0, "trust_env": False, "follow_redirects": False}] * 2


def test_openrouter_budget_ceiling_requires_explicit_override(monkeypatch):
    from muninn.history import auto_routing

    values = {"MUNINN_OPENROUTER_MAX_DAILY_USD": "25",
              "MUNINN_OPENROUTER_MAX_MONTHLY_USD": "250"}
    monkeypatch.setattr(auto_routing, "_local_setting", lambda name: values.get(name, ""))
    assert auto_routing.openrouter_budget_ceiling() == (10.0, 100.0)
    values["MUNINN_OPENROUTER_BUDGET_OVERRIDE"] = "1"
    assert auto_routing.openrouter_budget_ceiling() == (25.0, 250.0)
    values["MUNINN_OPENROUTER_MAX_DAILY_USD"] = "not-a-number"
    assert auto_routing.openrouter_budget_ceiling() == (0.0, 0.0)


@pytest.mark.parametrize(("reset", "limit", "daily_used", "monthly_used", "allowed"), [
    ("daily", 5, 0, 0, True),
    ("daily", 11, 0, 0, False),
    ("monthly", 50, 0, 0, True),
    ("monthly", 101, 0, 0, False),
    ("monthly", 50, 10, 0, False),
    ("daily", 5, 0, 100, False),
    (None, 5, 0, 0, False),
])
def test_openrouter_key_must_fit_both_periods(
    monkeypatch, reset, limit, daily_used, monthly_used, allowed,
):
    from muninn.history import auto_routing, llm_settings

    monkeypatch.setattr(llm_settings, "api_key", lambda: "fixture-key")
    monkeypatch.setattr(auto_routing, "_local_setting", lambda _name: "")
    info = {"limit_reset": reset, "limit": limit, "limit_remaining": 1,
            "usage_daily": daily_used, "usage_monthly": monthly_used}

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, *_args, **_kwargs):
            return SimpleNamespace(raise_for_status=lambda: None,
                                   json=lambda: {"data": info})

    monkeypatch.setattr(auto_routing.httpx, "Client", FakeClient)
    assert auto_routing.guarded_openrouter_available() is allowed


def test_openrouter_budget_probe_ignores_proxy_environment(monkeypatch):
    from muninn.history import auto_routing, llm_settings

    monkeypatch.setenv("HTTPS_PROXY", "http://untrusted-proxy.invalid:8080")
    monkeypatch.setattr(llm_settings, "api_key", lambda: "fixture-key")
    monkeypatch.setattr(auto_routing, "_local_setting", lambda _name: "")
    client_options = {}

    class FakeClient:
        def __init__(self, **kwargs):
            client_options.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, *_args, **_kwargs):
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"data": {"limit_reset": "daily", "limit": 1,
                                       "limit_remaining": 1, "usage_daily": 0,
                                       "usage_monthly": 0}},
            )

    monkeypatch.setattr(auto_routing.httpx, "Client", FakeClient)
    assert auto_routing.guarded_openrouter_available() is True
    assert client_options["trust_env"] is False


def test_openrouter_budget_probe_rejects_non_https_endpoint(monkeypatch):
    from muninn.history import auto_routing, llm_settings

    monkeypatch.setattr(llm_settings, "api_key", lambda: "fixture-key")
    monkeypatch.setattr(llm_settings, "OPENROUTER_API", "http://openrouter.ai/api/v1")
    monkeypatch.setattr(auto_routing.httpx, "Client",
                        lambda **_kwargs: pytest.fail("unsafe endpoint must not be contacted"))
    assert auto_routing.guarded_openrouter_available() is False


@pytest.mark.parametrize("data, expected", [
    ({"limit": 1, "limit_remaining": 0.75, "limit_reset": "daily",
      "usage_daily": 0.25, "usage_monthly": 2}, "ready"),
    ({"limit": 6, "limit_remaining": 6, "limit_reset": "daily",
      "usage_daily": 0, "usage_monthly": 0}, "key_cap_exceeds_local_threshold"),
    ({"limit": 1, "limit_remaining": 0, "limit_reset": "daily",
      "usage_daily": 1, "usage_monthly": 2}, "key_exhausted"),
    ({"limit": 1, "limit_remaining": 0.5, "limit_reset": "daily",
      "usage_daily": 5, "usage_monthly": 10}, "local_threshold_reached"),
    ({"limit": None, "limit_remaining": 1, "limit_reset": "daily",
      "usage_daily": 0, "usage_monthly": 0}, "invalid_provider_data"),
    ({"limit": float("inf"), "limit_remaining": 1, "limit_reset": "daily",
      "usage_daily": 0, "usage_monthly": 0}, "invalid_provider_data"),
    ({"limit": 10 ** 1000, "limit_remaining": 1, "limit_reset": "daily",
      "usage_daily": 0, "usage_monthly": 0}, "invalid_provider_data"),
    ({"limit": 1, "limit_remaining": -1, "limit_reset": "daily",
      "usage_daily": 0, "usage_monthly": 0}, "invalid_provider_data"),
    ({"limit": 1, "limit_remaining": 1, "limit_reset": None,
      "usage_daily": 0, "usage_monthly": 0}, "invalid_provider_data"),
    ({"limit": 1, "limit_remaining": 1, "limit_reset": "daily",
      "usage_daily": 0, "usage_monthly": 0, "disabled": "false"}, "invalid_provider_data"),
])
def test_openrouter_key_status_is_sanitized_and_fail_closed(monkeypatch, data, expected):
    from muninn.history import auto_routing, llm_settings

    monkeypatch.setattr(llm_settings, "api_key", lambda: "secret-fixture-key")
    monkeypatch.setattr(auto_routing, "_local_setting", lambda _name: "")
    options = {}

    class FakeClient:
        def __init__(self, **kwargs):
            options.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, *_args, **_kwargs):
            return SimpleNamespace(raise_for_status=lambda: None,
                                   json=lambda: {"data": {**data, "label": "private-label"}})

    monkeypatch.setattr(auto_routing.httpx, "Client", FakeClient)
    status = auto_routing.openrouter_key_status(5, 50)
    assert status["state"] == expected
    assert status["admission_ready"] is (expected == "ready")
    assert "secret-fixture-key" not in str(status)
    assert "private-label" not in str(status)
    assert options["trust_env"] is False
    assert options["follow_redirects"] is False


def test_openrouter_key_status_does_not_probe_when_disabled_or_key_missing(monkeypatch):
    from muninn.history import auto_routing, llm_settings

    monkeypatch.setattr(auto_routing.httpx, "Client",
                        lambda **_kwargs: pytest.fail("provider must not be contacted"))
    monkeypatch.setattr(auto_routing, "openrouter_budget_ceiling", lambda _root: (0.0, 0.0))
    monkeypatch.setattr(llm_settings, "api_key",
                        lambda: pytest.fail("key must not be read when disabled"))
    assert auto_routing.openrouter_key_status(policy_root=object())["state"] == "disabled"

    monkeypatch.setattr(auto_routing, "openrouter_budget_ceiling", lambda _root: (5.0, 50.0))
    monkeypatch.setattr(llm_settings, "api_key", lambda: None)
    assert auto_routing.openrouter_key_status(policy_root=object())["state"] == "key_missing"


def _gpu(free=13_600, used=1, loaded=()):
    return GpuState(free, 16_376, used, 100.0, loaded)


MODELS = [
    {"name": "muninn-qwen35-defiant-q8-test:latest", "size": 10_000 * 1024 * 1024},
    {"name": "qwen2.5:7b", "size": 4_700 * 1024 * 1024},
]


def test_real_history_quality_keeps_small_model_first_until_q8_is_proven():
    for turns in (0, 59, 60, 61):
        assert choose_route(_gpu(), MODELS, model_hints=model_hints_for_thread(turns),
                            now=100.0).model == "qwen2.5:7b"
    assert choose_route(_gpu(), MODELS, model_hints=("qwen35",),
                        now=100.0).model == MODELS[0]["name"]
    assert model_hints_for_thread(60, configured=("custom",)) == ("custom",)


def test_complex_thread_falls_back_to_smaller_model_when_headroom_falls():
    smaller = choose_route(_gpu(free=7_500), MODELS,
                           model_hints=model_hints_for_thread(60), now=100.0)
    assert (smaller.provider, smaller.model) == ("ollama", "qwen2.5:7b")


def test_busy_or_occupied_gpu_never_loads_another_local_model():
    for gpu in (_gpu(used=70), _gpu(loaded=("other-model",))):
        assert choose_route(gpu, MODELS, now=100.0).provider == "deferred"
        assert choose_route(gpu, MODELS, cloud_allowed=True, cloud_available=True,
                            now=100.0).provider == "openrouter"


def test_missing_stale_or_insufficient_gpu_fails_closed():
    assert choose_route(None, MODELS, now=100.0).provider == "deferred"
    assert choose_route(_gpu(free=5_000), MODELS, now=100.0).provider == "deferred"
    stale = GpuState(13_600, 16_376, 0, 80.0)
    assert choose_route(stale, MODELS, now=100.0).provider == "deferred"


def test_cloud_requires_both_explicit_policy_and_available_key():
    assert choose_route(None, MODELS, cloud_allowed=False, cloud_available=True,
                        now=100.0).provider == "deferred"
    assert choose_route(None, MODELS, cloud_allowed=True, cloud_available=False,
                        now=100.0).provider == "deferred"


def test_auto_activation_excludes_existing_history(monkeypatch):
    from muninn.history.service import HistoryService

    values = {}
    service = HistoryService.__new__(HistoryService)
    service.memory = SimpleNamespace(_metadata=SimpleNamespace(
        get_meta=lambda key: values.get(key),
        set_meta=lambda key, value: values.__setitem__(key, value),
    ))
    monkeypatch.setenv("MUNINN_INSIGHTS_AUTO", "1")
    monkeypatch.setattr("muninn.history.service.time.time", lambda: 1234.0)
    assert service._auto_since() == 1234.0
    assert service._auto_since() == 1234.0
    assert values == {"history_auto_insights_since": "1234.0"}


def test_auto_analysis_is_off_by_default_without_touching_provider(monkeypatch):
    from muninn.history.service import HistoryService

    monkeypatch.delenv("MUNINN_INSIGHTS_AUTO", raising=False)
    service = HistoryService.__new__(HistoryService)
    service.memory = SimpleNamespace(_metadata=SimpleNamespace(
        get_meta=lambda _key: pytest.fail("auto-off must not read analysis state"),
    ))
    assert service._auto_since() is None


@pytest.mark.asyncio
async def test_auto_analysis_routes_only_new_threads_to_idle_local_model(monkeypatch):
    from muninn.history import auto_routing
    from muninn.history.service import HistoryService

    service = HistoryService.__new__(HistoryService)
    service._auto_since = lambda: 100.0
    service.memory = SimpleNamespace(_metadata=SimpleNamespace(
        list_history_threads=lambda *_args, **_kwargs: [
            {"thread_key": "test-thread", "turns_imported": 12}],
    ))
    service.run_analysis = AsyncMock(return_value={})
    service.last_auto_route = None
    monkeypatch.setattr(auto_routing, "probe_gpu", lambda: GpuState(13_600, 16_376, 1, time.time()))
    monkeypatch.setattr(auto_routing, "probe_ollama", lambda _base: (MODELS, ()))
    monkeypatch.setattr(auto_routing, "guarded_openrouter_available",
                        lambda: pytest.fail("local route must not query OpenRouter"))
    await service._auto_analyze()
    assert service.run_analysis.await_args.kwargs == {
        "apply": True, "provider": "ollama", "model": "qwen2.5:7b",
        "since": 100.0, "limit": 1, "concurrency": 1,
        "thread_key": "test-thread",
    }


@pytest.mark.asyncio
async def test_auto_analysis_defers_when_gpu_busy(monkeypatch):
    from muninn.history import auto_routing
    from muninn.history.service import HistoryService

    service = HistoryService.__new__(HistoryService)
    service._auto_since = lambda: 100.0
    service.memory = SimpleNamespace(_metadata=SimpleNamespace(
        list_history_threads=lambda *_args, **_kwargs: [
            {"thread_key": "test-thread", "turns_imported": 12}],
    ))
    service.run_analysis = AsyncMock()
    service.last_auto_route = None
    monkeypatch.setattr(auto_routing, "probe_gpu", lambda: GpuState(13_600, 16_376, 90, time.time()))
    monkeypatch.setattr(auto_routing, "probe_ollama", lambda _base: (MODELS, ()))
    monkeypatch.setattr(auto_routing, "guarded_openrouter_available", lambda: False)
    await service._auto_analyze()
    service.run_analysis.assert_not_awaited()
    assert service.last_auto_route["reason"] == "gpu_busy"


@pytest.mark.asyncio
async def test_auto_analysis_uses_budgeted_cloud_when_gpu_busy(monkeypatch):
    from muninn.history import auto_routing
    from muninn.history.service import HistoryService

    service = HistoryService.__new__(HistoryService)
    service._auto_since = lambda: 100.0
    service.memory = SimpleNamespace(_metadata=SimpleNamespace(
        list_history_threads=lambda *_args, **_kwargs: [
            {"thread_key": "test-thread", "turns_imported": 12}],
    ))
    service.run_analysis = AsyncMock(return_value={})
    service.last_auto_route = None
    monkeypatch.setattr(auto_routing, "probe_gpu", lambda: GpuState(13_600, 16_376, 90, time.time()))
    monkeypatch.setattr(auto_routing, "probe_ollama", lambda _base: (MODELS, ()))
    monkeypatch.setattr(auto_routing, "guarded_openrouter_available", lambda: True)
    await service._auto_analyze()
    assert service.run_analysis.await_args.kwargs == {
        "apply": True, "provider": "openrouter", "since": 100.0,
        "limit": 1, "concurrency": 1, "thread_key": "test-thread",
    }
